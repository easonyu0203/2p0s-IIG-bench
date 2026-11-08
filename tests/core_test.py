"""Tests of the game API: how episodes start and step."""

import dataclasses
from typing import override

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nashbench
from nashbench import core


@core.pytree_dataclass
class DiceRaceState(core.GameState):
    """State of `DiceRace`.

    Attributes:
        key: PRNG key data that decides the rolls.
        totals: `[2]` int32 sum of each seat's rolls.
    """

    key: jax.Array
    totals: jax.Array


class DiceRace(core.Game[DiceRaceState]):
    """Seats take turns rolling a die; the first to reach 10 wins.

    Its rolls come from a key in its state, so it doesn't list its initial
    states. Its only action rolls.
    """

    num_actions = 1
    observation_shape = (4,)

    @override
    def sample_initial_state(self, key):
        return DiceRaceState(
            current_seat=jnp.int32(0),
            done=jnp.bool_(False),
            key=jax.random.key_data(key),
            totals=jnp.zeros(2, jnp.int32),
        )

    @override
    def apply_action(self, state, action):
        key, roll_key = jax.random.split(jax.random.wrap_key_data(state.key))
        roll = jax.random.randint(roll_key, (), 1, 7)
        totals = state.totals + roll * (jnp.arange(2) == state.current_seat)
        return dataclasses.replace(
            state,
            current_seat=1 - state.current_seat,
            done=(totals >= 10).any(),
            key=jax.random.key_data(key),
            totals=totals,
        )

    @override
    def observe(self, state, seat):
        return jnp.concatenate(
            [jax.nn.one_hot(seat, 2), state.totals / 10]
        ).astype(jnp.float32)

    @override
    def legal_action_mask(self, state, seat):
        return jnp.ones(1, bool)

    @override
    def returns(self, state):
        won = state.totals >= 10
        return won.astype(jnp.float32) - won[::-1]


def test_games_can_sample_initial_states_without_listing_them():
    """A game that samples its initial states plays batched episodes."""
    game = DiceRace()
    step = jax.jit(jax.vmap(nashbench.auto_reset(game)))
    state, timestep = jax.vmap(game.reset)(
        jax.random.split(jax.random.key(0), 64)
    )
    ended, rewards = 0, []
    for _ in range(40):
        state, timestep = step(state, jnp.zeros(64, jnp.int32))
        ended += int(timestep.done.sum())
        rewards.append(np.asarray(timestep.reward)[np.asarray(timestep.done)])
    rewards = np.concatenate(rewards)
    assert ended > 64
    np.testing.assert_array_equal(np.sort(rewards, 1), [[-1, 1]] * ended)
    # Each episode has its own rolls.
    assert len(np.unique(np.asarray(state.game_state.totals), axis=0)) > 1


def test_games_that_list_no_initial_states_have_no_sequence_form():
    """Exact evaluation of such a game says why it can't be done."""
    message = "DiceRace doesn't list its initial states"
    with pytest.raises(NotImplementedError, match=message):
        _ = DiceRace().sequence_form


def test_sample_initial_state_samples_initial_states_by_probability():
    """By default, initial states are sampled from the listed ones."""
    game = nashbench.make("kuhn_poker")
    states, probs = game.initial_states()
    keys = jax.random.split(jax.random.key(0), 6000)
    sampled = jax.vmap(game.sample_initial_state)(keys)
    listed = jax.tree.leaves(states)
    leaves = jax.tree.leaves(sampled)
    # Index each sample by the listed state it equals.
    matches = np.ones((len(keys), len(probs)), bool)
    for listed_leaf, leaf in zip(listed, leaves, strict=True):
        listed_leaf = np.asarray(listed_leaf).reshape(len(probs), -1)
        leaf = np.asarray(leaf).reshape(len(keys), -1)
        matches &= (leaf[:, None] == listed_leaf[None]).all(-1)
    assert (matches.sum(1) >= 1).all()
    frequencies = matches.argmax(1)
    counts = np.bincount(frequencies, minlength=len(probs))
    np.testing.assert_allclose(counts / len(keys), probs, atol=0.02)
