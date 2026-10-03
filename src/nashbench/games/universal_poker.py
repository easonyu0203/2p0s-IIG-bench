"""Limit poker, as in OpenSpiel's `universal_poker`."""

import collections
from collections.abc import Sequence
import dataclasses
import functools
import itertools
from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import core
from nashbench.games import _universal_poker_tree

FOLD, CALL, RAISE = 0, 1, 2
NUM_SUITS = 4


@core.pytree_dataclass
class UniversalPokerState(core.GameState):
    """State of limit poker.

    Attributes:
        cards: `[num_rounds + 1]` int32 private card of each seat, then the
            public card of each round after the first. Card `c` has rank
            `c // 4` and suit `c % 4`.
        round: `[]` int32 betting round, from 0.
        spent: `[2]` int32 chips each seat has put in the pot.
        sequences: `[num_rounds, max_raises + 2]` int32 actions of each round
            so far, then -1s.
        folded: `[2]` bool.
    """

    cards: jax.Array
    round: jax.Array
    spent: jax.Array
    sequences: jax.Array
    folded: jax.Array


class UniversalPoker(core.Game[UniversalPokerState]):
    """Limit poker with one private card per seat and four suits.

    Each seat antes 1 chip. Seat 0 acts first in every round, and each round
    after the first reveals one public card.
    """

    num_actions = 3

    def __init__(
        self,
        num_ranks: int = 3,
        raise_sizes: Sequence[int] = (2, 2, 4, 4),
        max_raises: int = 2,
    ):
        """Creates the game.

        Args:
            num_ranks: Ranks of the deck, of four suits each.
            raise_sizes: Chips of a raise in each round, one to four rounds.
            max_raises: Raises allowed in each round.
        """
        self.num_ranks = num_ranks
        self.raise_sizes = tuple(raise_sizes)
        self.max_raises = max_raises
        self.num_rounds = len(self.raise_sizes)
        self.num_cards = NUM_SUITS * num_ranks
        # OpenSpiel's bound on the game length, which sizes its tensor: an
        # action to end, the deals, two checks per round, and 20 raises.
        self._max_game_length = 3 * self.num_rounds + 22
        self.observation_shape = (
            2
            + (self.num_rounds + 1) * self.num_cards
            + 3 * self._max_game_length,
        )
        # The strength of each hand, by the ranks of its private card and
        # the public cards.
        hands = itertools.product(range(num_ranks), repeat=self.num_rounds)
        keys = [_strength_key(hand) for hand in hands]
        strength = {key: i for i, key in enumerate(sorted(set(keys)))}
        self._strength = np.array([strength[key] for key in keys]).reshape(
            [num_ranks] * self.num_rounds
        )

    @override
    def initial_states(self):
        deals = itertools.permutations(
            range(self.num_cards), self.num_rounds + 1
        )
        deals = np.array(list(deals))
        n = len(deals)
        states = UniversalPokerState(
            current_seat=jnp.zeros(n, jnp.int32),
            done=jnp.zeros(n, bool),
            cards=jnp.asarray(deals, jnp.int32),
            round=jnp.zeros(n, jnp.int32),
            spent=jnp.ones((n, 2), jnp.int32),
            sequences=jnp.full(
                (n, self.num_rounds, self.max_raises + 2), -1, jnp.int32
            ),
            folded=jnp.zeros((n, 2), bool),
        )
        return states, np.full(n, 1 / n)

    @override
    def apply_action(self, state, action):
        seat = state.current_seat
        # Calling matches the highest stake; raising then adds the raise.
        raise_size = jnp.asarray(self.raise_sizes)[state.round]
        stake = jnp.maximum(*state.spent) + (action == RAISE) * raise_size
        position = (state.sequences[state.round] >= 0).sum()
        # With two players, any call but the opening check ends the round.
        round_over = (action == CALL) & (position > 0)
        last_round = state.round == self.num_rounds - 1
        next_round = round_over & ~last_round
        acting = jnp.arange(2) == seat
        folds = action == FOLD
        rounds, positions = jnp.indices(state.sequences.shape)
        slot = (rounds == state.round) & (positions == position)
        return dataclasses.replace(
            state,
            current_seat=jnp.where(next_round, 0, 1 - seat),
            done=folds | (round_over & last_round),
            round=state.round + next_round,
            spent=jnp.where(acting & ~folds, stake, state.spent),
            sequences=jnp.where(slot, action, state.sequences),
            folded=state.folded | (acting & folds),
        )

    @override
    def observe(self, state, seat):
        n, length = self.num_cards, self._max_game_length
        public = jnp.where(
            jnp.arange(self.num_rounds - 1) < state.round, state.cards[2:], -1
        )
        public = jax.nn.one_hot(public, n)
        # The index of each action in OpenSpiel's action sequence, which also
        # holds the deals: two private cards, then a public card before each
        # round after the first.
        played = (state.sequences >= 0).sum(1)
        first = 2 + jnp.arange(self.num_rounds) + jnp.cumsum(played) - played
        index = first[:, None] + jnp.arange(self.max_raises + 2)
        # One-hot over (fold, call, raise) at each index.
        a = self.num_actions
        code = jnp.where(state.sequences >= 0, a * index + state.sequences, -1)
        actions = jax.nn.one_hot(code, a * length, dtype=bool).any((0, 1))
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                jax.nn.one_hot(state.cards[seat], n),
                public.sum(0),
                actions.reshape(length, a)[:, CALL:].ravel(),
                jnp.zeros(length),  # Raise sizes, which OpenSpiel sets to 0.
                # Each round's public card, unlike OpenSpiel; see the docs.
                public.ravel(),
            ]
        ).astype(jnp.float32)

    @override
    def legal_action_mask(self, state, seat):
        num_raises = (state.sequences[state.round] == RAISE).sum()
        return jnp.stack(
            [
                state.spent[seat] < state.spent[1 - seat],
                jnp.bool_(True),
                num_raises < self.max_raises,
            ]
        )

    @override
    def returns(self, state):
        # Each seat's hand: its private card, then the public cards.
        public = jnp.broadcast_to(state.cards[2:], (2, self.num_rounds - 1))
        hands = jnp.concatenate([state.cards[:2, None], public], 1)
        strength = jnp.asarray(self._strength)[tuple(hands.T // NUM_SUITS)]
        winners = jnp.where(
            state.folded.any(), ~state.folded, strength >= strength[::-1]
        )
        pot_share = winners * state.spent.sum() / winners.sum()
        return jnp.where(state.done, pot_share - state.spent, 0.0)

    @functools.cached_property
    def sequence_form(self):
        """The sequence form, built from the public betting tree."""
        return _universal_poker_tree.sequence_form(self)


def _strength_key(ranks):
    """Returns a key that orders hands of these ranks by strength.

    Hands compare by the sizes of their groups of equal ranks (four of a
    kind, three of a kind, two pairs, a pair), then by the ranks of the
    groups, the larger groups first.
    """
    groups = sorted(
        ((n, r) for r, n in collections.Counter(ranks).items()), reverse=True
    )
    return tuple(n for n, _ in groups), tuple(r for _, r in groups)
