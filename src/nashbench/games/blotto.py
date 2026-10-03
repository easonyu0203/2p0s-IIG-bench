"""Colonel Blotto, as in OpenSpiel's `blotto`."""

import dataclasses
import itertools
from typing import override

import jax
import jax.numpy as jnp
import numpy as np

from nashbench import core


@core.pytree_dataclass
class BlottoState(core.GameState):
    """State of Blotto.

    Attributes:
        allocations: `[2]` int32 allocation each seat chose, or -1.
    """

    allocations: jax.Array


class Blotto(core.Game[BlottoState]):
    """Blotto. Seats split their coins over the fields at the same time."""

    observation_shape = (3,)

    def __init__(self, coins: int = 10, fields: int = 5):
        """Creates the game.

        Args:
            coins: Coins each seat splits.
            fields: Fields to win.
        """
        self.coins = coins
        self.fields = fields
        # Coins per field of each action, in OpenSpiel's lexicographic order.
        self._allocations = np.array(
            [
                a
                for a in itertools.product(range(coins + 1), repeat=fields)
                if sum(a) == coins
            ]
        )
        self.num_actions = len(self._allocations)

    @override
    def initial_states(self):
        states = BlottoState(
            current_seat=jnp.zeros(1, jnp.int32),
            done=jnp.zeros(1, bool),
            allocations=jnp.full((1, 2), -1, jnp.int32),
        )
        return states, np.ones(1)

    @override
    def apply_action(self, state, action):
        seat = state.current_seat
        return dataclasses.replace(
            state,
            current_seat=1 - seat,
            done=seat == 1,
            allocations=jnp.where(
                jnp.arange(2) == seat, action, state.allocations
            ),
        )

    @override
    def observe(self, state, seat):
        # OpenSpiel's tensor is whether the game has ended.
        return jnp.append(jax.nn.one_hot(seat, 2), state.done)

    @override
    def legal_action_mask(self, state, seat):
        return jnp.ones(self.num_actions, bool)

    @override
    def returns(self, state):
        coins = jnp.asarray(self._allocations)[
            jnp.maximum(state.allocations, 0)
        ]
        fields = jnp.sign(coins[0] - coins[1])  # 1 where seat 0 wins.
        margin = jnp.sign(fields.sum()).astype(jnp.float32)
        return jnp.where(state.done, jnp.stack([margin, -margin]), 0.0)
