"""Oshi-Zumo, as in OpenSpiel's `oshi_zumo` with a minimum bid of 1."""

import dataclasses
import functools
from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import core
from nashbench.games import _oshi_zumo_tree


@core.pytree_dataclass
class OshiZumoState(core.GameState):
    """State of Oshi-Zumo.

    Attributes:
        bids: `[2, coins]` int32 coins each seat bid at each turn, then -1s.
            A game lasts at most `coins` turns.
        pending: `[]` int32 coins seat 0 bid in this turn, hidden until seat 1
            bids, or -1 if seat 0 is to bid.
    """

    bids: jax.Array
    pending: jax.Array


class OshiZumo(core.Game[OshiZumoState]):
    """Oshi-Zumo. Seats bid in turn, then see both bids."""

    def __init__(self, coins: int = 13, size: int = 3):
        """Creates the game.

        Args:
            coins: Coins each seat starts with.
            size: Positions between the middle and each edge. The wrestler
                starts in the middle of `2 * size + 3` positions.
        """
        self.coins = coins
        self.size = size
        self.num_actions = coins + 1
        self.observation_shape = (
            2 + 2 * (coins + 1) + 2 * size + 3 + 2 * coins * (coins + 1),
        )

    @override
    def initial_states(self):
        # No chance events: a single initial state.
        states = OshiZumoState(
            current_seat=jnp.zeros(1, jnp.int32),
            done=jnp.zeros(1, bool),
            bids=jnp.full((1, 2, self.coins), -1, jnp.int32),
            pending=jnp.full(1, -1, jnp.int32),
        )
        return states, np.ones(1)

    @override
    def apply_action(self, state, action):
        # Seat 0's bid is pending until seat 1 bids and ends the turn.
        ends = state.current_seat == 1
        turn = (state.bids[0] >= 0).sum()
        joint = jnp.stack([state.pending, action])
        bids = jnp.where(
            ends & (jnp.arange(self.coins) == turn), joint[:, None], state.bids
        )
        state = dataclasses.replace(
            state,
            current_seat=1 - state.current_seat,
            bids=bids,
            pending=jnp.where(ends, -1, action),
        )
        position = self._position(state)
        done = (
            (position == 0)
            | (position == 2 * self.size + 2)
            | (self._coins(state) == 0).all()
        )
        return dataclasses.replace(state, done=done)

    @override
    def observe(self, state, seat):
        # OpenSpiel has no information-state tensor. Its observation tensor,
        # the coins and position, is followed by every bid so far.
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                jax.nn.one_hot(self._coins(state), self.coins + 1).ravel(),
                jax.nn.one_hot(self._position(state), 2 * self.size + 3),
                jax.nn.one_hot(state.bids, self.coins + 1).ravel(),
            ]
        )

    @override
    def legal_action_mask(self, state, seat):
        # A seat bids at least 1 coin, or 0 once it has none.
        coins = self._coins(state)[seat]
        bid = jnp.arange(self.num_actions)
        return jnp.where(coins > 0, (bid >= 1) & (bid <= coins), bid == 0)

    @override
    def returns(self, state):
        # The seat that pushed the wrestler past the middle wins, whether or
        # not it fell off.
        margin = jnp.sign(self._position(state) - self.size - 1)
        margin = margin.astype(jnp.float32)
        return jnp.where(state.done, jnp.stack([margin, -margin]), 0.0)

    def _coins(self, state):
        """Returns the `[2]` coins each seat has left."""
        return self.coins - jnp.maximum(state.bids, 0).sum(1)

    def _position(self, state):
        """Returns the wrestler's position. Seat 0's wins push it up."""
        return self.size + 1 + jnp.sign(state.bids[0] - state.bids[1]).sum()

    @functools.cached_property
    def sequence_form(self):
        """The sequence form, built by enumerating the histories."""
        return _oshi_zumo_tree.sequence_form(self)
