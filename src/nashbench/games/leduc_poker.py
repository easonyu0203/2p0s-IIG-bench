"""Leduc poker, as in OpenSpiel's `leduc_poker`."""

import dataclasses
import itertools
from typing import override

import jax
import jax.numpy as jnp

from nashbench import core

FOLD, CALL, RAISE = 0, 1, 2


@core.pytree_dataclass
class LeducState(core.GameState):
    """State of Leduc poker.

    Attributes:
        private_cards: `[2]` int32 card of each seat. Card `c` has rank
            `c // 2` (J, Q, K) and suit `c % 2`.
        public_card: `[]` int32 card revealed in round 2.
        round: `[]` int32 betting round: 0 or 1.
        ante: `[2]` int32 chips each seat has put in the pot.
        sequences: `[2, 4]` int32 actions of each round so far, then -1s.
        folded: `[2]` bool.
    """

    private_cards: jax.Array
    public_card: jax.Array
    round: jax.Array
    ante: jax.Array
    sequences: jax.Array
    folded: jax.Array


class LeducPoker(core.Game[LeducState]):
    """Leduc poker."""

    num_actions = 3
    observation_shape = (30,)

    @override
    def initial_states(self):
        cards = jnp.array(list(itertools.permutations(range(6), 3)))
        n = len(cards)
        states = LeducState(
            current_seat=jnp.zeros(n, jnp.int32),
            done=jnp.zeros(n, bool),
            private_cards=cards[:, :2],
            public_card=cards[:, 2],
            round=jnp.zeros(n, jnp.int32),
            ante=jnp.ones((n, 2), jnp.int32),
            sequences=jnp.full((n, 2, 4), -1, jnp.int32),
            folded=jnp.zeros((n, 2), bool),
        )
        return states, jnp.full(n, 1 / n)

    @override
    def apply_action(self, state, action):
        seat = state.current_seat
        a = action[seat]
        # Calling matches the highest ante; raising then adds 2 or 4 chips.
        raise_size = jnp.where(state.round == 0, 2, 4)
        stakes = jnp.maximum(*state.ante) + jnp.where(a == RAISE, raise_size, 0)
        position = (state.sequences[state.round] >= 0).sum()
        # With two players, any call but the opening check ends the round.
        round_over = (a == CALL) & (position > 0)
        next_round = round_over & (state.round == 0)
        acting = jnp.arange(2) == seat
        slot = (jnp.arange(2)[:, None] == state.round) & (
            jnp.arange(4) == position
        )
        return dataclasses.replace(
            state,
            current_seat=jnp.where(next_round, 0, 1 - seat),
            done=(a == FOLD) | (round_over & (state.round == 1)),
            round=state.round + next_round,
            ante=jnp.where(acting & (a != FOLD), stakes, state.ante),
            sequences=jnp.where(slot, a, state.sequences),
            folded=state.folded | (acting & (a == FOLD)),
        )

    @override
    def observe(self, state, seat):
        public_card = jnp.where(state.round == 1, state.public_card, -1)
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                jax.nn.one_hot(state.private_cards[seat], 6),
                jax.nn.one_hot(public_card, 6),
                # Call is (1, 0) and raise is (0, 1). Folds end the game.
                jax.nn.one_hot(state.sequences - 1, 2).ravel(),
            ]
        )

    @override
    def legal_action_mask(self, state, seat):
        num_raises = (state.sequences[state.round] == RAISE).sum()
        return jnp.stack(
            [
                state.ante[seat] < state.ante[1 - seat],
                jnp.bool_(True),
                num_raises < 2,
            ]
        )

    @override
    def returns(self, state):
        rank = _hand_rank(state.private_cards, state.public_card)
        winners = jnp.where(
            state.folded[0] | state.folded[1], ~state.folded, rank >= rank[::-1]
        )
        pot_share = winners * (state.ante[0] + state.ante[1]) / winners.sum()
        return jnp.where(state.done, pot_share - state.ante, 0.0)


def _hand_rank(private_card, public_card):
    """Ranks hands like OpenSpiel: pairs, then high card, then low card."""
    low = jnp.minimum(private_card, public_card)
    high = jnp.maximum(private_card, public_card)
    return jnp.where(low // 2 == high // 2, 36 + low, high // 2 * 6 + low // 2)
