"""Rules shared by 3x3 phantom games: Phantom Tic-Tac-Toe and Dark Hex 3.

A seat sees only its own stones and the cells it has tried. Trying an
occupied cell reveals the opponent's stone there. In the classical version
the seat tries again; in the abrupt version its turn ends.
"""

import abc
import dataclasses
import functools
from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import core
from nashbench.games import _phantom_tree

NUM_CELLS = 9


@core.pytree_dataclass
class PhantomState(core.GameState):
    """State of a phantom game.

    Attributes:
        board: `[9]` int32 seat owning each cell, or -1 if empty.
        view: `[2, 9]` int32 board as each seat knows it, -1 if unknown.
        history: `[2, 9]` int32 cells each seat has tried, then -1s.
    """

    board: jax.Array
    view: jax.Array
    history: jax.Array


class PhantomGame(core.Game[PhantomState]):
    """A 3x3 phantom game; subclasses define winning and the view encoding."""

    num_actions = NUM_CELLS

    def __init__(self, abrupt: bool):
        """Creates the classical version, or the abrupt one if `abrupt`."""
        self.abrupt = abrupt
        stones = (np.arange(2**NUM_CELLS)[:, None] >> np.arange(NUM_CELLS)) & 1
        stones = stones.astype(bool).reshape(-1, 3, 3)
        # Whether a seat wins, by seat and by its stones as a bitmask.
        self._win_table = np.stack([self._wins(stones, s) for s in range(2)])

    @abc.abstractmethod
    def _wins(self, stones: np.ndarray, seat: int) -> np.ndarray:
        """Returns whether `seat` wins with `stones`, `[..., 3, 3]` bool."""

    @abc.abstractmethod
    def _encode_view(self, view: jax.Array) -> jax.Array:
        """Returns OpenSpiel's tensor encoding of one seat's `view`."""

    @override
    def initial_states(self):
        # No chance events: a single initial state.
        states = PhantomState(
            current_seat=jnp.zeros(1, jnp.int32),
            done=jnp.zeros(1, bool),
            board=jnp.full((1, NUM_CELLS), -1, jnp.int32),
            view=jnp.full((1, 2, NUM_CELLS), -1, jnp.int32),
            history=jnp.full((1, 2, NUM_CELLS), -1, jnp.int32),
        )
        return states, jnp.ones(1)

    @override
    def apply_action(self, state, action):
        seat = state.current_seat
        cell = action[seat]
        cells = jnp.arange(NUM_CELLS)
        placed = state.board[cell] < 0
        board = jnp.where((cells == cell) & placed, seat, state.board)
        own = (jnp.arange(2) == seat)[:, None]
        num_tries = (state.history[seat] >= 0).sum()
        return dataclasses.replace(
            state,
            # A blocked try ends the turn only in the abrupt version.
            current_seat=jnp.where(placed | self.abrupt, 1 - seat, seat),
            done=self._has_won(board, seat) | (board >= 0).all(),
            board=board,
            view=jnp.where(own & (cells == cell), board[cell], state.view),
            history=jnp.where(own & (cells == num_tries), cell, state.history),
        )

    @override
    def observe(self, state, seat):
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                self._encode_view(state.view[seat]),
                jax.nn.one_hot(state.history[seat], NUM_CELLS).ravel(),
            ]
        )

    @override
    def legal_action_mask(self, state, seat):
        return state.view[seat] < 0

    def _has_won(self, board, seat):
        """Returns whether `seat` has won on `board`."""
        stones = jnp.where(board == seat, 1 << jnp.arange(NUM_CELLS), 0).sum()
        return jnp.asarray(self._win_table)[seat, stones]

    @functools.cached_property
    def sequence_form(self):
        """The sequence form, built by traversing the tree on device."""
        return _phantom_tree.sequence_form(self)

    @override
    def returns(self, state):
        won = jnp.stack(
            [
                self._has_won(state.board, 0),
                self._has_won(state.board, 1),
            ]
        ).astype(jnp.float32)
        return won - won[::-1]
