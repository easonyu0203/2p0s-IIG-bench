"""Goofspiel with imperfect information, as in OpenSpiel's `goofspiel`."""

import dataclasses
import functools
import itertools
from typing import override

import jax
import jax.numpy as jnp

from nashbench import core
from nashbench.games import _goofspiel_tree


@core.pytree_dataclass
class GoofspielState(core.GameState):
    """State of Goofspiel.

    Attributes:
        point_cards: `[num_cards]` int32 point cards, in the order they are
            revealed.
        bids: `[2, num_cards]` int32 card each seat bid at each turn, then -1s.
        turn: `[]` int32 number of turns played.
    """

    point_cards: jax.Array
    bids: jax.Array
    turn: jax.Array


class Goofspiel(core.Game[GoofspielState]):
    """Goofspiel. Both seats bid at once, and see only who won each turn."""

    def __init__(self, num_cards: int = 6):
        """Creates the game with cards 1 to `num_cards`."""
        self.num_cards = num_cards
        self.num_actions = num_cards
        k = num_cards
        max_points = k * (k + 1) // 2
        self.observation_shape = (2 + 2 * (max_points + 1) + 3 * k + 2 * k * k,)

    @override
    def initial_states(self):
        k = self.num_cards
        point_cards = jnp.array(list(itertools.permutations(range(k))))
        n = len(point_cards)
        states = GoofspielState(
            current_seat=jnp.full(n, core.BOTH, jnp.int32),
            done=jnp.zeros(n, bool),
            point_cards=point_cards,
            bids=jnp.full((n, 2, k), -1, jnp.int32),
            turn=jnp.zeros(n, jnp.int32),
        )
        return states, jnp.full(n, 1 / n)

    @override
    def apply_action(self, state, action):
        k = self.num_cards
        bids = state.bids.at[:, state.turn].set(action)
        # The last turn has a single legal bid per seat, so it plays itself.
        turn = state.turn + 1
        last = k * (k - 1) // 2 - bids[:, : k - 1].sum(1)
        bids = jnp.where(turn == k - 1, bids.at[:, k - 1].set(last), bids)
        turn = jnp.where(turn == k - 1, k, turn)
        return dataclasses.replace(state, done=turn == k, bids=bids, turn=turn)

    @override
    def observe(self, state, seat):
        k = self.num_cards
        winner = _winners(state)
        points = _points(state, winner)
        max_points = k * (k + 1) // 2
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                jax.nn.one_hot(points[seat], max_points + 1),
                jax.nn.one_hot(points[1 - seat], max_points + 1),
                self.legal_action_mask(state, seat).astype(jnp.float32),
                jax.nn.one_hot(winner, 2).ravel(),
                jax.nn.one_hot(
                    jnp.where(
                        jnp.arange(k) <= state.turn, state.point_cards, -1
                    ),
                    k,
                ).ravel(),
                jax.nn.one_hot(state.bids[seat], k).ravel(),
            ]
        )

    @override
    def legal_action_mask(self, state, seat):
        return ~jax.nn.one_hot(
            state.bids[seat], self.num_cards, dtype=bool
        ).any(0)

    @override
    def returns(self, state):
        points = _points(state, _winners(state))
        margin = jnp.sign(points[0] - points[1]).astype(jnp.float32)
        return jnp.where(state.done, jnp.stack([margin, -margin]), 0.0)

    @functools.cached_property
    def sequence_form(self):
        """The sequence form, built from the regular structure of the tree."""
        return _goofspiel_tree.sequence_form(self)


def _winners(state):
    """Returns the `[num_cards]` seat that won each turn, or -1 if none."""
    b0, b1 = state.bids
    return jnp.where(b0 > b1, 0, jnp.where(b1 > b0, 1, -1))


def _points(state, winner):
    """Returns the `[2]` points of each seat: the values of the cards won."""
    value = state.point_cards + 1
    return jnp.stack([jnp.where(winner == s, value, 0).sum() for s in range(2)])
