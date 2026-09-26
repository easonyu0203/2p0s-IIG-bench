"""The policy interface that evaluation expects."""

from typing import Protocol

import jax


class Policy(Protocol):
    """A policy that plays both seats of a game.

    A policy maps one player's observation to a distribution over actions.
    Observations encode the seat, so a single policy defines how both seats
    play. Evaluation calls policies under `jax.jit` and `jax.vmap`, so they
    must be JAX-transformable; close over any parameters.
    """

    def __call__(
        self, observation: jax.Array, legal_action_mask: jax.Array
    ) -> jax.Array:
        """Returns action probabilities for one observation.

        Args:
            observation: `[*observation_shape]` observation of one player.
            legal_action_mask: `[num_actions]` bool mask of legal actions.

        Returns:
            `[num_actions]` probabilities that sum to 1 and are 0 for illegal
            actions.
        """
        ...


def uniform_random(
    observation: jax.Array, legal_action_mask: jax.Array
) -> jax.Array:
    """Plays a uniformly random legal action. A `Policy`."""
    del observation  # Unused.
    return legal_action_mask / legal_action_mask.sum()
