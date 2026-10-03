"""Battleship, as in OpenSpiel's `battleship` with one ship per seat."""

import dataclasses
import functools
from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import core
from nashbench.games import _battleship_tree


@core.pytree_dataclass
class BattleshipState(core.GameState):
    """State of Battleship.

    Attributes:
        placements: `[2]` int32 action that placed each seat's ship, or -1.
        shots: `[2, num_shots]` int32 cells each seat shot, in order, then
            -1s.
    """

    placements: jax.Array
    shots: jax.Array


class Battleship(core.Game[BattleshipState]):
    """Battleship. Seats see their own ship, and where their shots hit."""

    def __init__(
        self,
        height: int = 6,
        width: int = 6,
        ship_size: int = 2,
        num_shots: int = 2,
    ):
        """Creates the game.

        Args:
            height: Rows of each seat's board.
            width: Columns of each seat's board.
            ship_size: Cells of each seat's ship.
            num_shots: Shots of each seat.
        """
        self.height = height
        self.width = width
        self.ship_size = ship_size
        self.num_shots = num_shots
        num_cells = height * width
        self.num_actions = 3 * num_cells
        self.observation_shape = (
            9 + height + width + 2 * num_shots * (5 + height + width),
        )
        # The cells each action places the ship on; none if not a placement.
        # Action `num_cells + c` places it horizontally from cell `c`, and
        # `2 * num_cells + c` vertically. One-cell ships only go horizontally.
        row, col = np.divmod(np.arange(num_cells), width)
        ship = np.zeros((self.num_actions, num_cells), bool)
        for direction in range(1 + (ship_size > 1)):
            rows = row + direction * np.arange(ship_size)[:, None]
            cols = col + (1 - direction) * np.arange(ship_size)[:, None]
            fits = (rows < height).all(0) & (cols < width).all(0)
            for c in np.flatnonzero(fits):
                action = (direction + 1) * num_cells + c
                ship[action, rows[:, c] * width + cols[:, c]] = True
        self._ship = ship

    @override
    def initial_states(self):
        states = BattleshipState(
            current_seat=jnp.zeros(1, jnp.int32),
            done=jnp.zeros(1, bool),
            placements=jnp.full((1, 2), -1, jnp.int32),
            shots=jnp.full((1, 2, self.num_shots), -1, jnp.int32),
        )
        return states, np.ones(1)

    @override
    def apply_action(self, state, action):
        seat = state.current_seat
        own = jnp.arange(2) == seat
        # Seats place their ships in turn, then shoot in turn.
        placing = state.placements[seat] < 0
        taken = (state.shots[seat] >= 0).sum()
        slot = own[:, None] & (jnp.arange(self.num_shots) == taken)
        state = dataclasses.replace(
            state,
            current_seat=1 - seat,
            placements=jnp.where(own & placing, action, state.placements),
            shots=jnp.where(slot & ~placing, action, state.shots),
        )
        done = self._sunk(state).any() | (state.shots >= 0).all()
        return dataclasses.replace(state, done=done)

    @override
    def observe(self, state, seat):
        h, w, n = self.height, self.width, self.height * self.width
        # OpenSpiel's tensor: whether the game ended, the seat, the seat to
        # act, the seat's placement, then every shot in order.
        placement = state.placements[seat]
        cell = jnp.where(placement >= 0, placement % n, -1)
        vertical = placement >= 2 * n
        shots = state.shots.T.ravel()  # Seat 0's first shot, seat 1's, ...
        shooter = jnp.arange(2 * self.num_shots) % 2
        # The outcome of the seat's own shots: water, hit, or sunk.
        hit, sinks = self._hits(state), self._sinks(state)
        outcome = jnp.stack([~hit, hit & ~sinks, sinks], -1)
        outcome = outcome.transpose(1, 0, 2).reshape(-1, 3)
        outcome &= ((shooter == seat) & (shots >= 0))[:, None]
        return jnp.concatenate(
            [
                jax.nn.one_hot(seat, 2),
                state.done[None],
                jax.nn.one_hot(seat, 2),
                jax.nn.one_hot(
                    jnp.where(state.done, -1, state.current_seat), 2
                ),
                jax.nn.one_hot(jnp.where(cell >= 0, vertical, -1), 2),
                _cell(cell, h, w),
                jnp.concatenate(
                    [
                        jax.nn.one_hot(jnp.where(shots >= 0, shooter, -1), 2),
                        _cell(shots, h, w),
                        outcome,
                    ],
                    -1,
                ).ravel(),
            ]
        ).astype(jnp.float32)

    @override
    def legal_action_mask(self, state, seat):
        n, num_actions = self.height * self.width, self.num_actions
        placements = jnp.asarray(self._ship.any(1))
        shot = jax.nn.one_hot(state.shots[seat], num_actions, dtype=bool)
        shots = (jnp.arange(num_actions) < n) & ~shot.any(0)
        return jnp.where(state.placements[seat] < 0, placements, shots)

    @override
    def returns(self, state):
        # A seat wins 1 by sinking the other's ship.
        sunk = self._sunk(state).astype(jnp.float32)
        return sunk[::-1] - sunk

    def _ships(self, state):
        """Returns the `[2, num_cells]` cells of each seat's ship."""
        return jnp.asarray(self._ship)[jnp.maximum(state.placements, 0)]

    def _hits(self, state):
        """Returns `[2, num_shots]`, whether each shot hit the other ship."""
        ships = self._ships(state)[::-1]
        hits = jnp.take_along_axis(ships, jnp.maximum(state.shots, 0), 1)
        return hits & (state.shots >= 0)

    def _sinks(self, state):
        """Returns `[2, num_shots]`, whether each shot sank the other ship."""
        hits = self._hits(state)
        return hits & (jnp.cumsum(hits, 1) == self.ship_size)

    def _sunk(self, state):
        """Returns `[2]`, whether each seat's ship has sunk."""
        return self._sinks(state)[::-1].any(1)

    @functools.cached_property
    def sequence_form(self):
        """The sequence form, built from the regular structure of the tree."""
        return _battleship_tree.sequence_form(self)


def _cell(cell, height, width):
    """Returns one-hot rows and columns of cells, or zeros for -1."""
    row, col = jnp.divmod(cell, width)
    return jnp.concatenate(
        [
            jax.nn.one_hot(jnp.where(cell >= 0, row, -1), height),
            jax.nn.one_hot(jnp.where(cell >= 0, col, -1), width),
        ],
        -1,
    )
