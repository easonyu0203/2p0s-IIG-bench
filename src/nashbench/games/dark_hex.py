"""Dark Hex on a 3x3 board, as in OpenSpiel's `dark_hex`."""

from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from nashbench.games import _phantom


class DarkHex3(_phantom.PhantomGame):
    """Dark Hex 3.

    Seat 0 (black) moves first and connects the top and bottom rows; seat 1
    (white) connects the left and right columns.
    """

    observation_shape = (164,)

    @override
    def _wins(self, stones, seat):
        # Transposing preserves hex adjacency and swaps the two directions.
        if seat == 1:
            stones = stones.swapaxes(-1, -2)
        reached = stones & (np.arange(3) == 0)[:, None]
        for _ in range(_phantom.NUM_CELLS):
            reached = stones & (reached | _neighbors(reached))
        return reached[..., -1, :].any(-1)

    @override
    def _encode_view(self, view):
        # OpenSpiel's cell states, offset by 4: 3 white, 4 empty, 5 black.
        cell_state = jnp.where(view < 0, 4, 5 - 2 * view)
        return jax.nn.one_hot(cell_state, 9).ravel()


def _neighbors(cells: np.ndarray) -> np.ndarray:
    """Returns the cells adjacent to any of `cells`, `[..., 3, 3]` bool."""
    # Cell (r, c) neighbors (r-1, c), (r-1, c+1), (r, c-1), (r, c+1),
    # (r+1, c-1), and (r+1, c).
    p = np.pad(cells, [(0, 0)] * (cells.ndim - 2) + [(1, 1), (1, 1)])
    return (
        p[..., :-2, 1:-1]
        | p[..., :-2, 2:]
        | p[..., 1:-1, :-2]
        | p[..., 1:-1, 2:]
        | p[..., 2:, :-2]
        | p[..., 2:, 1:-1]
    )
