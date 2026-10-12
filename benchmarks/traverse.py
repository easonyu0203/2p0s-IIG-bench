"""Compares every history of each game with OpenSpiel.

    python benchmarks/traverse.py [--only REGEX] [--workers N] [--out FILE]

openspiel_test.py compares random histories of the larger games, which keeps
the tests fast. This script compares all of them, at each game's default
size: it walks OpenSpiel's tree and nashbench's together, for every deal of
cards or dice, and checks at every node that both agree on whether the game
is over, the returns, the seat to act, its legal actions and both seats'
observations, which are OpenSpiel's information-state tensors up to the
differences documented in openspiel_test.py. It also checks that
observations and OpenSpiel's information states determine each other, and
that every deal has OpenSpiel's probability. Each game appends a JSON line to
FILE with its node and information-state counts; later runs skip games that
FILE already has.
"""

import argparse
import collections
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

# Each worker runs JAX on one CPU thread; the GPU would only add transfers.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import jax
from open_spiel.python.observation import make_observation
import pyspiel

import nashbench

jax.config.update("jax_enable_x64", False)

# Name: (nashbench game, OpenSpiel game, every deal, depth of the split).
# A deal lists the chance outcomes in OpenSpiel's dealing order. The split
# depth is the number of player actions after which subtrees become tasks.
GAMES = {
    "kuhn_poker": (
        "kuhn_poker",
        "kuhn_poker",
        lambda: itertools.permutations(range(3), 2),
        0,
    ),
    "leduc_poker": (
        "leduc_poker",
        "leduc_poker",
        lambda: itertools.permutations(range(6), 3),
        0,
    ),
    "universal_poker": (
        "universal_poker",
        (
            "universal_poker(betting=limit,numPlayers=2,numRounds=4,"
            "blind=1 1,raiseSize=2 2 4 4,firstPlayer=1 1 1 1,"
            "maxRaises=2 2 2 2,numSuits=4,numRanks=3,numHoleCards=1,"
            "numBoardCards=0 1 1 1)"
        ),
        lambda: itertools.permutations(range(12), 5),
        0,
    ),
    "liars_dice": (
        "liars_dice",
        "liars_dice(numdice=2,dice_sides=5)",
        lambda: itertools.product(range(5), repeat=4),
        0,
    ),
    "goofspiel": (
        "goofspiel",
        (
            "turn_based_simultaneous_game(game="
            "goofspiel(players=2,num_cards=6,imp_info=true))"
        ),
        # OpenSpiel deals the last point card without a chance node.
        lambda: (p[:5] for p in itertools.permutations(range(6))),
        0,
    ),
    "oshi_zumo": (
        "oshi_zumo",
        (
            "turn_based_simultaneous_game(game="
            "oshi_zumo(coins=13,size=3,min_bid=1))"
        ),
        lambda: [()],
        2,
    ),
    "blotto": (
        "blotto",
        "turn_based_simultaneous_game(game=blotto(coins=10,fields=5))",
        lambda: [()],
        1,
    ),
    "battleship": (
        "battleship",
        (
            "battleship(board_height=6,board_width=6,ship_sizes=[2],"
            "ship_values=[1],num_shots=2,allow_repeated_shots=false)"
        ),
        lambda: [()],
        2,
    ),
    "phantom_ttt": ("phantom_ttt", "phantom_ttt", lambda: [()], 4),
    "phantom_ttt_abrupt": (
        "phantom_ttt_abrupt",
        "phantom_ttt(gameversion=abrupt)",
        lambda: [()],
        4,
    ),
    "dark_hex3": ("dark_hex3", "dark_hex", lambda: [()], 4),
    "dark_hex3_abrupt": (
        "dark_hex3_abrupt",
        "dark_hex(gameversion=adh)",
        lambda: [()],
        4,
    ),
}
CHUNK = 4096


@functools.cache
def _games(key) -> tuple[Any, pyspiel.Game]:
    name, os_name, _, _ = GAMES[key]
    return nashbench.make(name), pyspiel.load_game(os_name)


@functools.cache
def _functions(key):
    """Batched nashbench rules on padded chunks, compiled once."""
    game, _ = _games(key)

    def inspect(state):
        return {
            "observation": jax.vmap(game.observe, (None, 0))(
                state, np.arange(2)
            ),
            "legal_action_mask": game.legal_action_mask(
                state, state.current_seat
            ),
            "current_seat": state.current_seat,
            "done": state.done,
            "returns": game.returns(state),
        }

    return jax.jit(jax.vmap(inspect)), jax.jit(jax.vmap(game.apply_action))


def _pad(tree, size):
    """Pads the leading axis of every leaf to `size` by repeating row 0."""
    n = jax.tree.leaves(tree)[0].shape[0]
    if n == size:
        return tree
    index = np.concatenate([np.arange(n), np.zeros(size - n, int)])
    return jax.tree.map(lambda x: x[index], tree)


@functools.cache
def _initial_states(key):
    """The game's initial states and their probabilities, as NumPy arrays."""
    game, _ = _games(key)
    states, probs = game.initial_states()
    return jax.tree.map(np.asarray, states), np.asarray(probs)


def _root(key, deal):
    """The nashbench state after `deal`, as in openspiel_test._replay."""
    game, _ = _games(key)
    state = jax.tree.map(lambda x: x[0], _initial_states(key)[0])
    name = GAMES[key][0]
    chance = np.asarray(deal, np.int32)
    if name == "kuhn_poker":
        state = dataclasses.replace(state, cards=chance)
    elif name == "leduc_poker":
        state = dataclasses.replace(
            state, private_cards=chance[:2], public_card=chance[2]
        )
    elif name == "goofspiel":
        k = game.num_cards
        last = k * (k - 1) // 2 - chance.sum()
        state = dataclasses.replace(
            state, point_cards=np.append(chance, last).astype(np.int32)
        )
    elif name == "liars_dice":
        state = dataclasses.replace(state, dice=np.sort(chance.reshape(2, -1)))
    elif name == "universal_poker":
        state = dataclasses.replace(state, cards=chance)
    return state


def _deal_key(state):
    """A hashable identity of a nashbench state."""
    return b"".join(np.asarray(x).tobytes() for x in jax.tree.leaves(state))


@dataclasses.dataclass
class Node:
    """OpenSpiel's state at a node, and how much of its deal it has used."""

    state: pyspiel.State
    deal: tuple
    dealt: int  # Chance outcomes applied so far.
    probability: float  # Product of their probabilities.


def _settle(node):
    """Applies the deal's outcomes at chance nodes until a player acts."""
    state = node.state
    while state.is_chance_node():
        outcome = node.deal[node.dealt]
        probability = dict(state.chance_outcomes())[outcome]
        state.apply_action(outcome)
        node.dealt += 1
        node.probability *= probability
    return node


def _expected(key, nodes, size):
    """OpenSpiel's view of a chunk of nodes, as nashbench encodes it."""
    game, _ = _games(key)
    out = {
        "observation": np.zeros((size, 2, *game.observation_shape), np.float32),
        "legal_action_mask": np.zeros((size, game.num_actions), bool),
        "current_seat": np.zeros(size, np.int32),
        "done": np.zeros(size, bool),
        "returns": np.zeros((size, 2), np.float32),
    }
    information = [None] * size
    name = GAMES[key][0]
    for i, node in enumerate(nodes):
        state = node.state
        if state.is_terminal():
            out["done"][i] = True
            out["returns"][i] = state.returns()
            continue
        seat = state.current_player()
        out["current_seat"][i] = seat
        out["legal_action_mask"][i, state.legal_actions()] = True
        for s in range(2):
            _fill(key, node, s, out["observation"][i, s])
        information[i] = _information_state(name, node, seat)
    return out, information


@functools.cache
def _observer(key):
    """OpenSpiel's tensors of `key`, and where nashbench's copy starts.

    Returns the observer and the length of the seat's one-hot encoding that
    nashbench prepends, unless OpenSpiel's tensor already starts with it.
    """
    game, os_game = _games(key)
    name = GAMES[key][0]
    if name == "oshi_zumo":
        observer = make_observation(os_game)  # OpenSpiel's observation.
        extra = -2 + 2 * game.coins * (game.coins + 1)
    else:
        kind = pyspiel.IIGObservationType(perfect_recall=True)
        observer = make_observation(os_game, kind)
        extra = {"goofspiel": -2, "blotto": -2}.get(name, 0)
        if name == "universal_poker":
            extra = (game.num_rounds - 1) * game.num_cards
    assert observer is not None and observer.tensor is not None
    size = observer.tensor.size + extra
    return observer, game.observation_shape[0] - size


def _fill(key, node, seat, row):
    """Writes OpenSpiel's tensor of `seat` into `row`, as in the tests.

    Turn-based versions of games with simultaneous moves start their tensors
    with the seat to act, which nashbench omits. Oshi-Zumo has no
    information-state tensor; nashbench appends every bid of finished turns
    to its observation tensor. In limit poker, nashbench appends the public
    card of each round.
    """
    game, _ = _games(key)
    name = GAMES[key][0]
    observer, prefix = _observer(key)
    observer.set_from(node.state, seat)
    tensor = observer.tensor
    assert tensor is not None
    if prefix:
        row[seat] = 1
    if name in ("goofspiel", "blotto"):
        row[prefix:] = tensor[2:]
    elif name == "universal_poker":
        row[prefix : prefix + tensor.size] = tensor
        public = row[prefix + tensor.size :].reshape(-1, game.num_cards)
        for i, card in enumerate(node.deal[2 : node.dealt]):
            public[i, card] = 1
    elif name == "oshi_zumo":
        row[prefix : prefix + tensor.size - 2] = tensor[2:]
        history = node.state.history()
        num_turns = len(history) // 2
        bids = np.full((game.coins, 2), -1)
        bids[:num_turns] = np.reshape(history[: 2 * num_turns], (-1, 2))
        one_hot = bids.T[..., None] == np.arange(game.coins + 1)
        row[prefix + tensor.size - 2 :] = one_hot.ravel()
    else:
        row[prefix:] = tensor


def _information_state(name, node, seat):
    """OpenSpiel's information state of `seat`, as in the tests."""
    info = node.state.information_state_string(seat)
    if name == "dark_hex3" or name == "dark_hex3_abrupt":
        lines = info.split("\n")
        info = "\n".join(lines[:3] + lines[4:])
    if name == "universal_poker":
        info += str(list(node.deal[2 : node.dealt]))
    digest = hashlib.blake2b(f"{seat}|{info}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "little")


class Mismatch(Exception):
    """Nashbench and OpenSpiel disagree."""


def _compare(key, nodes, states, depth, stats):
    """Compares one chunk; returns its OpenSpiel and nashbench children."""
    inspect, _ = _functions(key)
    n = len(nodes)
    actual = jax.tree.map(np.asarray, inspect(_pad(states, CHUNK)))
    expected, information = _expected(key, nodes, CHUNK)
    done = expected["done"][:n]
    decision = ~done
    checks = [
        ("done", np.ones(n, bool)),
        ("returns", np.ones(n, bool)),
        ("current_seat", decision),
        ("legal_action_mask", decision),
        ("observation", decision),
    ]
    for field, where in checks:
        a, e = actual[field][:n], expected[field][:n]
        differs = (a != e).reshape(n, -1).any(-1) & where
        if differs.any():
            i = int(np.argmax(differs))
            raise Mismatch(
                f"{key}: {field} differs after OpenSpiel history "
                f"{nodes[i].state.history()}:\nnashbench: {a[i]}\n"
                f"OpenSpiel: {e[i]}"
            )
    stats["terminal"] += int(done.sum())
    stats["decision"] += int(decision.sum())
    # Observation of the seat to act, against its information state.
    seats = actual["current_seat"][:n]
    hashes = _hash_rows(actual["observation"][np.arange(n), seats])
    pairs = [(int(hashes[i]), information[i]) for i in np.flatnonzero(decision)]
    stats["pairs"].update(pairs)
    for node in nodes:
        if node.deal and node.dealt == len(node.deal):
            stats["deals"][node.deal].add(round(node.probability, 15))
    # Children of decision nodes, in OpenSpiel's order of legal actions.
    parents, actions, children = [], [], []
    for i in np.flatnonzero(decision):
        node = nodes[i]
        for action in node.state.legal_actions():
            parents.append(i)
            actions.append(action)
            child = Node(
                node.state.child(action),
                node.deal,
                node.dealt,
                node.probability,
            )
            children.append(_settle(child))
    return parents, actions, children


def _hash_rows(rows):
    """A 64-bit hash of each row's bytes.

    Each element's bits and position go through the splitmix64 finalizer
    before they are summed, so that no bits are lost to the trailing zeros
    of float32 values such as 1.0.
    """
    bits = rows.reshape(len(rows), -1).view(np.uint32).astype(np.uint64)
    position = np.arange(bits.shape[1], dtype=np.uint64) << np.uint64(32)
    z = (bits | position) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return z.sum(axis=1, dtype=np.uint64)


def _walk(key, roots, stop=None):
    """Compares every node below `roots`, depth first in chunks.

    Args:
        key: Name of the game in GAMES.
        roots: (deal, actions, state) triples at one depth: OpenSpiel's
            deal and player actions, and nashbench's state there, or None
            for the state right after the deal.
        stop: If set, nodes this many actions below the deal are returned
            instead of compared.

    Returns:
        The statistics, and the uncompared roots at depth `stop`.
    """
    _, os_game = _games(key)
    _, step = _functions(key)
    stats = {
        "terminal": 0,
        "decision": 0,
        "pairs": set(),
        "deals": collections.defaultdict(set),
    }
    nodes, states = [], []
    for deal, actions, state in roots:
        node = _settle(Node(os_game.new_initial_state(), tuple(deal), 0, 1.0))
        for action in actions:
            node.state.apply_action(action)
            node = _settle(node)
        nodes.append(node)
        states.append(_root(key, deal) if state is None else state)
    depth = len(roots[0][1])
    states = jax.tree.map(lambda *x: np.stack(x), *states)
    stack = [
        (
            nodes[i : i + CHUNK],
            jax.tree.map(lambda x, i=i: x[i : i + CHUNK], states),
            depth,
        )
        for i in range(0, len(nodes), CHUNK)
    ]
    frontier = []
    while stack:
        nodes, states, depth = stack.pop()
        if stop is not None and depth == stop:
            frontier.extend(
                (
                    node.deal,
                    tuple(_actions(node)),
                    jax.tree.map(lambda x, i=i: x[i], states),
                )
                for i, node in enumerate(nodes)
            )
            continue
        parents, actions, children = _compare(key, nodes, states, depth, stats)
        for start in range(0, len(children), CHUNK):
            index = np.asarray(parents[start : start + CHUNK])
            batch = jax.tree.map(lambda x, index=index: x[index], states)
            moves = np.zeros(CHUNK, np.int32)
            moves[: len(index)] = actions[start : start + CHUNK]
            batch = step(_pad(batch, CHUNK), moves)
            n = len(index)
            batch = jax.tree.map(lambda x, n=n: np.asarray(x)[:n], batch)
            stack.append((children[start : start + CHUNK], batch, depth + 1))
    return stats, frontier


def _actions(node):
    """The player actions in a node's OpenSpiel history."""
    return [
        a.action
        for a in node.state.full_history()
        if a.player != pyspiel.PlayerId.CHANCE
    ]


def _task(args) -> dict[str, Any]:
    key, roots = args
    try:
        stats, _ = _walk(key, roots)
    except Mismatch as error:
        return {"error": str(error)}
    pairs = np.array(sorted(stats["pairs"]), dtype=np.uint64).reshape(-1, 2)
    return {
        "terminal": stats["terminal"],
        "decision": stats["decision"],
        "pairs": pairs,
        "deals": {d: sorted(p) for d, p in stats["deals"].items()},
    }


def _check_game(key, workers, log):
    """Traverses one game; returns its record or raises Mismatch."""
    start = time.perf_counter()
    deals = [tuple(d) for d in GAMES[key][2]()]
    split = GAMES[key][3]
    # Nodes above the split are compared here; subtrees go to the workers.
    top, frontier = _walk(key, [(d, (), None) for d in deals], stop=split)
    # About 2,000 tasks keep the workers busy until the end.
    size = max(1, len(frontier) // 2000)
    tasks = [
        (key, frontier[i : i + size]) for i in range(0, len(frontier), size)
    ]
    totals = collections.Counter(
        terminal=top["terminal"], decision=top["decision"]
    )
    pairs = [np.array(sorted(top["pairs"]), dtype=np.uint64).reshape(-1, 2)]
    deal_probabilities = collections.defaultdict(set)
    for d, p in top["deals"].items():
        deal_probabilities[d].update(p)
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers) as pool:
        for k, result in enumerate(pool.imap_unordered(_task, tasks), 1):
            if "error" in result:
                pool.terminate()
                raise Mismatch(result["error"])
            totals["terminal"] += result["terminal"]
            totals["decision"] += result["decision"]
            pairs.append(result["pairs"])
            for d, p in result["deals"].items():
                deal_probabilities[d].update(p)
            if len(pairs) > 64:
                pairs = [np.unique(np.concatenate(pairs), axis=0)]
            if k % 100 == 0 or k == len(tasks):
                log(
                    f"{key}: {k}/{len(tasks)} tasks, "
                    f"{totals['terminal']:.3e} terminal histories, "
                    f"{time.perf_counter() - start:.0f}s"
                )
    pairs = np.unique(np.concatenate(pairs), axis=0)
    observations = len(np.unique(pairs[:, 0]))
    informations = len(np.unique(pairs[:, 1]))
    if not observations == informations == len(pairs):
        raise Mismatch(
            f"{key}: {len(pairs)} (observation, information state) pairs "
            f"for {observations} observations and {informations} "
            "information states"
        )
    _check_deals(key, deals, deal_probabilities)
    return {
        "game": key,
        "openspiel": GAMES[key][1],
        "deals": len(deals),
        "terminal_nodes": totals["terminal"],
        "decision_nodes": totals["decision"],
        "information_states": len(pairs),
        "seconds": time.perf_counter() - start,
        "workers": workers,
        "openspiel_version": getattr(pyspiel, "__version__", None),
    }


def _check_deals(key, deals, probabilities):
    """Each deal has one OpenSpiel probability, and nashbench's."""
    expected = collections.defaultdict(float)
    for deal in deals:
        # Games without chance have one deal, the empty one.
        values = probabilities.get(deal) or ({1.0} if deal == () else None)
        if not values or len(values) != 1:
            raise Mismatch(f"{key}: deal {deal} has probabilities {values}")
        expected[_deal_key(_root(key, deal))] += next(iter(values))
    states, probs = _initial_states(key)
    leaves = jax.tree.leaves(states)
    actual = collections.defaultdict(float)
    for i in range(len(probs)):
        actual[b"".join(x[i].tobytes() for x in leaves)] += float(probs[i])
    if expected.keys() != actual.keys() or any(
        not math.isclose(expected[k], actual[k], rel_tol=1e-9) for k in expected
    ):
        raise Mismatch(f"{key}: deal probabilities differ from OpenSpiel's")


def main():
    """Traverses the selected games and appends their records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=".*")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2)
    )
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/traverse.jsonl")
    )
    args = parser.parse_args()
    done = set()
    if args.out.exists():
        done = {json.loads(line)["game"] for line in args.out.open()}

    def log(message):
        print(message, flush=True)

    for key in GAMES:
        if key in done or not re.fullmatch(args.only, key):
            continue
        record = _check_game(key, args.workers, log)
        log(json.dumps(record))
        with args.out.open("a") as stream:
            stream.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
