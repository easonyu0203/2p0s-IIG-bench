"""Liar's Dice, as in OpenSpiel's `liars_dice`."""

import dataclasses
import functools
import itertools
from typing import override

import jax
import jax.numpy as jnp

from nashbench import core
from nashbench.games import _liars_dice_tree


@core.pytree_dataclass
class LiarsDiceState(core.GameState):
    """State of Liar's Dice.

    Attributes:
        dice: `[2, numdice]` int32 faces of each seat's dice, sorted. Face `f`
            shows `f + 1` pips.
        bids: `[num_bids]` bool, whether each bid has been made.
    """

    dice: jax.Array
    bids: jax.Array


class LiarsDice(core.Game[LiarsDiceState]):
    """Liar's Dice. Seats see only their own dice."""

    def __init__(self, numdice: int = 2, dice_sides: int = 5):
        """Creates the game.

        Args:
            numdice: Dice each seat rolls.
            dice_sides: Sides of each die.
        """
        self.numdice = numdice
        self.dice_sides = dice_sides
        self.num_bids = 2 * numdice * dice_sides
        self.num_actions = self.num_bids + 1  # The last action calls liar.
        self.observation_shape = (2 + numdice * dice_sides + self.num_actions,)

    @override
    def initial_states(self):
        rolls = itertools.product(
            range(self.dice_sides), repeat=2 * self.numdice
        )
        dice = jnp.sort(jnp.array(list(rolls)).reshape(-1, 2, self.numdice))
        n = len(dice)
        states = LiarsDiceState(
            current_seat=jnp.zeros(n, jnp.int32),
            done=jnp.zeros(n, bool),
            dice=dice,
            bids=jnp.zeros((n, self.num_bids), bool),
        )
        return states, jnp.full(n, 1 / n)

    @override
    def apply_action(self, state, action):
        seat = state.current_seat
        bid = action[seat]
        liar = bid == self.num_bids
        return dataclasses.replace(
            state,
            current_seat=jnp.where(liar, seat, 1 - seat),
            done=liar,
            bids=state.bids | (jnp.arange(self.num_bids) == bid),
        )

    @override
    def observe(self, state, seat):
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                jax.nn.one_hot(state.dice[seat], self.dice_sides).ravel(),
                state.bids,
                state.done[None],  # Whether liar was called.
            ]
        ).astype(jnp.float32)

    @override
    def legal_action_mask(self, state, seat):
        # A bid must beat the last one, and liar needs a bid to call.
        action = jnp.arange(self.num_actions)
        return (action > _last_bid(state)) & (
            (action < self.num_bids) | state.bids.any()
        )

    @override
    def returns(self, state):
        # Bid `b` claims at least `b // dice_sides + 1` dice show face
        # `b % dice_sides`. The caller is the seat to act.
        quantity, face = jnp.divmod(_last_bid(state), self.dice_sides)
        wild = self.dice_sides - 1
        matches = ((state.dice == face) | (state.dice == wild)).sum()
        caller_wins = matches <= quantity
        won = (jnp.arange(2) == state.current_seat) == caller_wins
        return jnp.where(state.done, jnp.where(won, 1.0, -1.0), 0.0)

    @functools.cached_property
    def sequence_form(self):
        """The sequence form, built from the regular structure of the tree."""
        return _liars_dice_tree.sequence_form(self)


def _last_bid(state):
    """Returns the highest bid made, or -1 if none."""
    return jnp.where(state.bids, jnp.arange(state.bids.size), -1).max()
