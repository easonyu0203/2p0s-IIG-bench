"""Phantom Tic-Tac-Toe, as in OpenSpiel's `phantom_ttt`."""

from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from nashbench.games import _phantom


class PhantomTTT(_phantom.PhantomGame):
    """Phantom Tic-Tac-Toe. Seat 0 plays x and moves first."""

    observation_shape = (110,)

    @override
    def _wins(self, stones, seat):
        del seat  # Both seats win with a line.
        return (
            stones.all(-1).any(-1)
            | stones.all(-2).any(-1)
            | np.diagonal(stones, axis1=-2, axis2=-1).all(-1)
            | np.diagonal(np.flip(stones, -1), axis1=-2, axis2=-1).all(-1)
        )

    @override
    def _encode_view(self, view):
        # OpenSpiel's cell states: 0 empty, 1 o (seat 1), 2 x (seat 0).
        cell_state = jnp.where(view < 0, 0, 2 - view)
        return jax.nn.one_hot(cell_state, 3).T.ravel()
