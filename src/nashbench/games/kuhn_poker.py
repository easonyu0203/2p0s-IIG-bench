"""Kuhn poker, as in OpenSpiel's `kuhn_poker`."""

import dataclasses
import itertools
from typing import override

import jax
import jax.numpy as jnp

from nashbench import core

PASS, BET = 0, 1


@core.pytree_dataclass
class KuhnState(core.GameState):
    """State of Kuhn poker.

    Attributes:
        cards: `[2]` int32 card of each seat: 0 (J), 1 (Q), or 2 (K).
        history: `[3]` int32 actions taken so far, then -1s.
    """

    cards: jax.Array
    history: jax.Array


class KuhnPoker(core.Game[KuhnState]):
    """Kuhn poker."""

    num_actions = 2
    observation_shape = (11,)

    @override
    def initial_states(self):
        cards = jnp.array(list(itertools.permutations(range(3), 2)))
        n = len(cards)
        states = KuhnState(
            current_seat=jnp.zeros(n, jnp.int32),
            done=jnp.zeros(n, bool),
            cards=cards,
            history=jnp.full((n, 3), -1, jnp.int32),
        )
        return states, jnp.full(n, 1 / n)

    @override
    def apply_action(self, state, action):
        num_actions = (state.history >= 0).sum()
        history = jnp.where(jnp.arange(3) == num_actions, action, state.history)
        # Only pass-bet continues after two actions.
        pass_bet = (history[0] == PASS) & (history[1] == BET)
        return dataclasses.replace(
            state,
            current_seat=(num_actions + 1) % 2,
            done=(num_actions == 2) | ((num_actions == 1) & ~pass_bet),
            history=history,
        )

    @override
    def observe(self, state, seat):
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                jax.nn.one_hot(state.cards[seat], 3),
                jax.nn.one_hot(state.history, 2).ravel(),
            ]
        )

    @override
    def legal_action_mask(self, state, seat):
        return jnp.ones(2, bool)

    @override
    def returns(self, state):
        # Seat 0 takes actions 0 and 2, seat 1 takes action 1.
        bet = state.history == BET
        bet0, bet1 = bet[0] | bet[2], bet[1]
        # With equal bets the higher card wins; otherwise the passer folded.
        winner = jnp.where(bet0 == bet1, state.cards[1] > state.cards[0], bet1)
        # The loser pays its ante, and its bet if both bet.
        stake = 1 + (bet0 & bet1)
        won = jnp.arange(2) == winner
        return jnp.where(state.done, jnp.where(won, stake, -stake), 0)
