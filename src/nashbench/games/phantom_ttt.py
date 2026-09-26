"""Phantom Tic-Tac-Toe, as in OpenSpiel's `phantom_ttt`."""

from typing import override

import jax
import jax.numpy as jnp

from nashbench.games import _phantom


class PhantomTTT(_phantom.PhantomGame):
    """Phantom Tic-Tac-Toe. Seat 0 plays x and moves first."""

    observation_shape = (110,)

    @override
    def _has_won(self, board, seat):
        stones = (board == seat).reshape(3, 3)
        return (
            stones.all(axis=0).any()
            | stones.all(axis=1).any()
            | jnp.diag(stones).all()
            | jnp.diag(jnp.fliplr(stones)).all()
        )

    @override
    def _encode_view(self, view):
        # OpenSpiel's cell states: 0 empty, 1 o (seat 1), 2 x (seat 0).
        cell_state = jnp.where(view < 0, 0, 2 - view)
        return jax.nn.one_hot(cell_state, 3).T.ravel()
