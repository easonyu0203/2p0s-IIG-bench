"""Tests that nashbench games replicate their OpenSpiel counterparts.

Each test replays histories of the OpenSpiel game in nashbench, in seat
space, and compares every node. Small games are checked on every history;
larger games on random histories.
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
            f"goofspiel(players=2,num_cards={k},imp_info=true)",
        )
        for k in range(3, 7)
    },
}
# Every deal (chance outcomes, in dealing order) of games checked on every
# history.
DEALS = {
    "kuhn_poker": list(itertools.permutations(range(3), 2)),
    "leduc_poker": list(itertools.permutations(range(6), 3)),
    "goofspiel3": list(itertools.permutations(range(3))),
    "goofspiel4": list(itertools.permutations(range(4))),
}
NUM_RANDOM_HISTORIES = 3000


def _joint_actions(state):
    """Returns the legal (seat 0, seat 1) actions; one repeats if turn-based."""
    if state.is_simultaneous_node():
        return list(
            itertools.product(state.legal_actions(0), state.legal_actions(1))
        )
    return [(a, a) for a in state.legal_actions()]


def _apply(state, actions):
    if state.is_simultaneous_node():
        state.apply_actions(list(actions))
    else:
        state.apply_action(actions[0])


def _histories(test_id, os_game):
    """Returns a list of (chance outcomes, joint actions) histories."""
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
                    actions.append(rng.choice(_joint_actions(state)))
                    _apply(state, actions[-1])
            histories.append((tuple(chance), tuple(actions)))
        return histories

    def extend(deal, state, actions, num_dealt=0):
        if state.is_chance_node():
            child = state.child(deal[num_dealt])
            yield from extend(deal, child, actions, num_dealt + 1)
        elif state.is_terminal():
            yield deal, actions
        else:
            for joint in _joint_actions(state):
                child = state.clone()
                _apply(child, joint)
                yield from extend(deal, child, (*actions, joint), num_dealt)

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
        _apply(state, next(actions))


def _information_state(state, player):
    info = state.information_state_string(player)
    if state.get_game().get_type().short_name == "dark_hex":
        # Drop the total move count, which only the string reveals (see
        # docs/games/dark-hex.md).
        lines = info.split("\n")
        info = "\n".join(lines[:3] + lines[4:])
    return player, info


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
                tensor = np.ravel(state.information_state_tensor(seat))
                # nashbench prepends the seat unless OpenSpiel already does.
                if tensor.size < expected["observation"][i, t, seat].size:
                    tensor = np.concatenate([np.eye(2)[seat], tensor])
                expected["observation"][i, t, seat] = tensor
                if state.current_player() in (
                    seat,
                    pyspiel.PlayerId.SIMULTANEOUS,
                ):
                    acting[i, t, seat] = True
                    legal = state.legal_actions(seat)
                    expected["legal_action_mask"][i, t, seat, legal] = True
                    information_state[i, t, seat] = _information_state(
                        state, seat
                    )

    chance = jnp.array([c for c, _ in histories], jnp.int32)
    actions = np.zeros((*shape, 2), np.int32)  # One padding action at the end.
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
