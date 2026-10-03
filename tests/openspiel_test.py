"""Tests that nashbench games replicate their OpenSpiel counterparts.

Each test replays histories of the OpenSpiel game in nashbench, in seat
space, and compares every node. Small games are checked on every history;
larger games on random histories. Games with simultaneous moves are compared
with OpenSpiel's turn-based version of them.
"""

import collections
import dataclasses
import functools
import itertools
import random

import jax
import jax.numpy as jnp
import numpy as np
import pyspiel
import pytest

import nashbench

# Test ID: (nashbench game, its parameters, OpenSpiel game).
GAMES = {
    "kuhn_poker": ("kuhn_poker", {}, "kuhn_poker"),
    "leduc_poker": ("leduc_poker", {}, "leduc_poker"),
    "phantom_ttt": ("phantom_ttt", {}, "phantom_ttt"),
    "phantom_ttt_abrupt": (
        "phantom_ttt_abrupt",
        {},
        "phantom_ttt(gameversion=abrupt)",
    ),
    "dark_hex3": ("dark_hex3", {}, "dark_hex"),
    "dark_hex3_abrupt": ("dark_hex3_abrupt", {}, "dark_hex(gameversion=adh)"),
    **{
        f"goofspiel{k}": (
            "goofspiel",
            {"num_cards": k},
            (
                "turn_based_simultaneous_game(game="
                f"goofspiel(players=2,num_cards={k},imp_info=true))"
            ),
        )
        for k in range(3, 7)
    },
    **{
        f"liars_dice{n}x{s}": (
            "liars_dice",
            {"numdice": n, "dice_sides": s},
            f"liars_dice(numdice={n},dice_sides={s})",
        )
        for n, s in [(1, 3), (2, 2), (1, 6), (2, 5)]
    },
    **{
        f"oshi_zumo{c}x{s}": (
            "oshi_zumo",
            {"coins": c, "size": s},
            (
                "turn_based_simultaneous_game(game="
                f"oshi_zumo(coins={c},size={s},min_bid=1))"
            ),
        )
        for c, s in [(5, 3), (6, 1), (13, 3)]
    },
    **{
        f"blotto{c}x{f}": (
            "blotto",
            {"coins": c, "fields": f},
            (
                "turn_based_simultaneous_game(game="
                f"blotto(coins={c},fields={f}))"
            ),
        )
        for c, f in [(4, 3), (10, 5)]
    },
    **{
        f"battleship{h}x{w}_{size}_{shots}": (
            "battleship",
            {"height": h, "width": w, "ship_size": size, "num_shots": shots},
            (
                f"battleship(board_height={h},board_width={w},"
                f"ship_sizes=[{size}],ship_values=[1],num_shots={shots},"
                "allow_repeated_shots=false)"
            ),
        )
        for h, w, size, shots in [(2, 3, 1, 2), (2, 4, 3, 3), (6, 6, 2, 2)]
    },
    **{
        f"universal_poker{r}_{len(sizes)}_{m}": (
            "universal_poker",
            {"num_ranks": r, "raise_sizes": sizes, "max_raises": m},
            (
                "universal_poker(betting=limit,numPlayers=2,"
                f"numRounds={len(sizes)},blind=1 1,"
                f"raiseSize={' '.join(map(str, sizes))},"
                f"firstPlayer={' '.join(['1'] * len(sizes))},"
                f"maxRaises={' '.join([str(m)] * len(sizes))},numSuits=4,"
                f"numRanks={r},numHoleCards=1,"
                f"numBoardCards={' '.join(['0'] + ['1'] * (len(sizes) - 1))})"
            ),
        )
        for r, sizes, m in [
            (2, (2, 4), 1),
            (2, (1, 2, 3), 2),
            (3, (2, 2, 4, 4), 2),
        ]
    },
}
# Every deal (chance outcomes, in dealing order) of games checked on every
# history.
DEALS = {
    "kuhn_poker": list(itertools.permutations(range(3), 2)),
    "leduc_poker": list(itertools.permutations(range(6), 3)),
    "goofspiel3": list(itertools.permutations(range(3))),
    "goofspiel4": list(itertools.permutations(range(4))),
    "liars_dice1x3": list(itertools.product(range(3), repeat=2)),
    "liars_dice2x2": list(itertools.product(range(2), repeat=4)),
    "oshi_zumo5x3": [()],
    "oshi_zumo6x1": [()],
    "blotto4x3": [()],
    "battleship2x3_1_2": [()],
    "universal_poker2_2_1": list(itertools.permutations(range(8), 3)),
}
NUM_RANDOM_HISTORIES = 3000


def _histories(test_id, os_game):
    """Returns a list of (chance outcomes, actions) histories."""
    if test_id not in DEALS:
        rng = random.Random(0)
        histories = []
        for _ in range(NUM_RANDOM_HISTORIES):
            state, chance, actions = os_game.new_initial_state(), [], []
            while not state.is_terminal():
                if state.is_chance_node():
                    chance.append(rng.choice(state.legal_actions()))
                    state.apply_action(chance[-1])
                else:
                    actions.append(rng.choice(state.legal_actions()))
                    state.apply_action(actions[-1])
            histories.append((tuple(chance), tuple(actions)))
        return histories

    def extend(deal, state, actions, num_dealt=0):
        if state.is_chance_node():
            child = state.child(deal[num_dealt])
            yield from extend(deal, child, actions, num_dealt + 1)
        elif state.is_terminal():
            yield deal, actions
        else:
            for action in state.legal_actions():
                child = state.child(action)
                yield from extend(deal, child, (*actions, action), num_dealt)

    root = os_game.new_initial_state()
    return [h for deal in DEALS[test_id] for h in extend(deal, root, ())]


def _openspiel_states(os_game, chance, actions):
    """Yields the OpenSpiel state at every player or terminal node."""
    chance, actions = iter(chance), iter(actions)
    state = os_game.new_initial_state()
    while True:
        while state.is_chance_node():
            state.apply_action(next(chance))
        yield state
        if state.is_terminal():
            return
        state.apply_action(next(actions))


def _information_state_tensor(game, name, state, seat):
    """Returns OpenSpiel's information-state tensor of `seat`.

    Turn-based versions of games with simultaneous moves start their tensors
    with the seat to act, which nashbench omits. Oshi-Zumo has no
    information-state tensor; nashbench appends every bid of finished turns to
    its observation tensor. In limit poker, nashbench appends the public card
    of each round.
    """
    if name in ("goofspiel", "blotto"):
        return np.ravel(state.information_state_tensor(seat))[2:]
    if name == "universal_poker":
        public = np.zeros((game.num_rounds - 1, game.num_cards))
        for i, card in enumerate(_chance(state)[2:]):
            public[i, card] = 1
        tensor = state.information_state_tensor(seat)
        return np.concatenate([tensor, public.ravel()])
    if name != "oshi_zumo":
        return np.ravel(state.information_state_tensor(seat))
    history = state.history()
    num_turns = len(history) // 2
    bids = np.full((game.coins, 2), -1)
    bids[:num_turns] = np.reshape(history[: 2 * num_turns], (-1, 2))
    one_hot = bids.T[..., None] == np.arange(game.coins + 1)
    return np.concatenate([state.observation_tensor(seat)[2:], one_hot.ravel()])


def _information_state(state, player):
    info = state.information_state_string(player)
    name = state.get_game().get_type().short_name
    if name == "dark_hex":
        # Drop the total move count, which only the string reveals (see
        # docs/games/dark-hex.md).
        lines = info.split("\n")
        info = "\n".join(lines[:3] + lines[4:])
    if name == "universal_poker":
        # Add the order of the public cards, which the string lacks (see
        # docs/games/universal-poker.md).
        info += str(_chance(state)[2:])
    return player, info


def _chance(state):
    """Returns the chance outcomes of a state's history."""
    return [
        a.action
        for a in state.full_history()
        if a.player == pyspiel.PlayerId.CHANCE
    ]


@functools.cache
def _nodes(test_id):
    """Returns OpenSpiel's and nashbench's view of every node, by history."""
    name, params, os_name = GAMES[test_id]
    game = nashbench.make(name, **params)
    os_game = pyspiel.load_game(os_name)
    histories = _histories(test_id, os_game)
    num_nodes = max(len(actions) for _, actions in histories) + 1
    shape = (len(histories), num_nodes)
    expected = {
        "observation": np.zeros((*shape, 2, *game.observation_shape)),
        "legal_action_mask": np.zeros((*shape, 2, game.num_actions), bool),
        "current_seat": np.zeros(shape, int),
        "done": np.zeros(shape, bool),
        "returns": np.zeros((*shape, 2)),
    }
    is_node = np.zeros(shape, bool)
    acting = np.zeros((*shape, 2), bool)
    information_state = {}
    for i, (chance, actions) in enumerate(histories):
        for t, state in enumerate(_openspiel_states(os_game, chance, actions)):
            is_node[i, t] = True
            expected["done"][i, t] = state.is_terminal()
            expected["returns"][i, t] = state.returns()
            if state.is_terminal():
                continue
            expected["current_seat"][i, t] = state.current_player()
            for seat in range(2):
                tensor = _information_state_tensor(game, name, state, seat)
                # nashbench prepends the seat unless OpenSpiel already does.
                if tensor.size < expected["observation"][i, t, seat].size:
                    tensor = np.concatenate([np.eye(2)[seat], tensor])
                expected["observation"][i, t, seat] = tensor
                if state.current_player() == seat:
                    acting[i, t, seat] = True
                    legal = state.legal_actions()
                    expected["legal_action_mask"][i, t, seat, legal] = True
                    information_state[i, t, seat] = _information_state(
                        state, seat
                    )

    # Poker histories that end early lack public cards, which the rules then
    # don't use.
    num_chance = max(len(c) for c, _ in histories)
    chance = jnp.array(
        [c + (0,) * (num_chance - len(c)) for c, _ in histories], jnp.int32
    )
    actions = np.zeros(shape, np.int32)  # One padding action at the end.
    for i, (_, a) in enumerate(histories):
        actions[i, : len(a)] = a
    replay = functools.partial(_replay, game, name)
    actual = jax.jit(jax.vmap(replay))(chance, actions)
    actual = jax.tree.map(np.asarray, actual)
    return histories, is_node, acting, expected, actual, information_state


def _replay(game, name, chance, actions):
    """Returns nashbench's view of the nodes of one history."""
    state = jax.tree.map(lambda x: x[0], game.initial_states()[0])
    if name == "kuhn_poker":
        state = dataclasses.replace(state, cards=chance)
    elif name == "leduc_poker":
        state = dataclasses.replace(
            state, private_cards=chance[:2], public_card=chance[2]
        )
    elif name == "goofspiel":
        # OpenSpiel deals the last point card without a chance node.
        k = game.num_cards
        last = k * (k - 1) // 2 - chance[: k - 1].sum()
        cards = jnp.append(chance[: k - 1], last)
        state = dataclasses.replace(state, point_cards=cards)
    elif name == "liars_dice":
        dice = jnp.sort(chance.reshape(2, -1))
        state = dataclasses.replace(state, dice=dice)
    elif name == "universal_poker":
        state = dataclasses.replace(state, cards=chance)

    def step(state, action):
        seats = jnp.arange(2)
        node = {
            "observation": jax.vmap(game.observe, (None, 0))(state, seats),
            "legal_action_mask": jax.vmap(game.legal_action_mask, (None, 0))(
                state, seats
            ),
            "current_seat": state.current_seat,
            "done": state.done,
            "returns": game.returns(state),
        }
        return game.apply_action(state, action), node

    return jax.lax.scan(step, state, actions)[1]


@pytest.mark.parametrize("test_id", GAMES)
def test_matches_openspiel(test_id):
    """Observations, legal actions, turns, and returns match OpenSpiel."""
    histories, is_node, acting, expected, actual, _ = _nodes(test_id)
    is_decision = is_node & ~expected["done"]
    for field, where in [
        ("done", is_node),
        ("returns", is_node),
        ("current_seat", is_decision),
        ("legal_action_mask", is_decision[..., None] & acting),
        ("observation", is_decision[..., None].repeat(2, -1)),
    ]:
        differs = actual[field] != expected[field]
        differs = differs.reshape(*where.shape, -1).any(-1) & where
        if differs.any():
            i, t = np.argwhere(differs)[0][:2]
            pytest.fail(
                f"{field} differs after actions {histories[i][1][:t]} with "
                f"chance {histories[i][0]}:\n"
                f"nashbench: {actual[field][i, t]}\n"
                f"OpenSpiel: {expected[field][i, t]}"
            )


@pytest.mark.parametrize("test_id", GAMES)
def test_observation_identifies_information_state(test_id):
    """Observations and (seat, information state) determine each other."""
    _, _, _, _, actual, information_state = _nodes(test_id)
    observations = collections.defaultdict(set)
    information_states = collections.defaultdict(set)
    for (i, t, seat), info in information_state.items():
        observation = actual["observation"][i, t, seat].tobytes()
        observations[info].add(observation)
        information_states[observation].add(info)
    assert all(len(v) == 1 for v in observations.values())
    assert all(len(v) == 1 for v in information_states.values())
