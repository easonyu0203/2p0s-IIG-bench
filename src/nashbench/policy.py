"""The policy interface that evaluation expects."""

from collections.abc import Sequence
from typing import Protocol

import jax
import numpy as np
import numpy.typing as npt


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


class Mixture:
    """Policies that a seat samples from, once per episode.

    At the start of each episode, a seat samples one policy, with probability
    proportional to its weight, and plays it for the whole episode. Seats
    sample independently and privately. A policy can itself be a mixture.

    Attributes:
        policies: The policies, as a tuple.
        weights: `[len(policies)]` float64 weights, divided by their sum.
    """

    def __init__(
        self,
        policies: Sequence["Policy | Mixture"],
        weights: npt.ArrayLike | None = None,
    ):
        """Initializes the mixture.

        Args:
            policies: The policies.
            weights: `[len(policies)]` finite, nonnegative weights with a
                positive sum. Defaults to equal weights.
        """
        self.policies = tuple(policies)
        n = len(self.policies)
        weights = np.asarray(np.ones(n) if weights is None else weights, float)
        if not (
            weights.shape == (n,)
            and np.isfinite(weights).all()
            and (weights >= 0).all()
            and weights.sum() > 0
        ):
            raise ValueError(
                f"weights must be {n} finite, nonnegative numbers with a "
                f"positive sum, not {weights}."
            )
        self.weights = weights / weights.sum()


def uniform_random(
    observation: jax.Array, legal_action_mask: jax.Array
) -> jax.Array:
    """Plays a uniformly random legal action. A `Policy`."""
    del observation  # Unused.
    return legal_action_mask / legal_action_mask.sum()
