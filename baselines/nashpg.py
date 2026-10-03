"""NashPG: PPO, regularized toward a magnet that is refined iteratively.

    uv run baselines/nashpg.py --game leduc_poker --decisions 10000000

NashPG (https://arxiv.org/abs/2510.18183) adds KL(π ‖ magnet) to PPO's loss,
and replaces the magnet with the current actor every `magnet_interval`
updates.
"""

import dataclasses
import functools
from typing import Any, NamedTuple

import common
import jax
import jax.numpy as jnp
import ppo
import tyro

import nashbench


@dataclasses.dataclass(frozen=True)
class Config(ppo.Config):
    """NashPG's hyperparameters."""

    num_envs: int = 64
    num_steps: int = 64
    learning_rate: float = 3e-4
    entropy_coef: float = 0.1
    value_coef: float = 1.0
    magnet_coef: float = 0.2
    """Weight of KL(π ‖ magnet) in the loss."""
    magnet_interval: int = 1000
    """Updates between replacing the magnet with the actor."""


class State(NamedTuple):
    """The state of training.

    Attributes:
        ppo_state: PPO's state.
        magnet: The magnet's actor parameters.
        updates: `[]` updates so far.
    """

    ppo_state: ppo.State
    magnet: Any
    updates: jax.Array


def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state of training, whose magnet is the actor."""
    state = ppo.init(config, game)
    return State(state, state.params["actor"], jnp.zeros((), jnp.int32))


def update(
    config: Config, game: nashbench.Game, state: State
) -> tuple[State, jax.Array]:
    """PPO's update, with NashPG's loss and magnet."""
    loss = functools.partial(nashpg_loss, config, state.magnet)
    ppo_state, episodes = ppo.update(config, game, state.ppo_state, loss)
    updates = state.updates + 1
    magnet = jax.tree.map(
        lambda m, p: jnp.where(updates % config.magnet_interval == 0, p, m),
        state.magnet,
        ppo_state.params["actor"],
    )
    return State(ppo_state, magnet, updates), episodes


def nashpg_loss(
    config: Config,
    magnet: Any,
    params: Any,
    window: common.Step,
    advantage: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Returns PPO's loss plus `magnet_coef` × KL(π ‖ magnet)."""
    log_probs, magnet_log_probs = (
        common.log_policy(actor, window.observation, window.legal_action_mask)
        for actor in (params["actor"], magnet)
    )
    kl = jnp.sum(jnp.exp(log_probs) * (log_probs - magnet_log_probs), 1)
    return ppo.ppo_loss(
        config, params, window, advantage, weight
    ) + config.magnet_coef * common.weighted_mean(kl, weight)


if __name__ == "__main__":
    common.run(
        "nashpg",
        tyro.cli(Config),
        init,
        common.repeat(update, ppo.decisions_per_update),
        lambda state: common.policy(state.ppo_state.params["actor"]),
    )
