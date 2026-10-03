"""MMD: magnetic mirror descent.

    uv run baselines/mmd.py --game leduc_poker --decisions 10000000

MMD (https://arxiv.org/abs/2206.05825) regularizes PPO toward a uniform
magnet, through an entropy bonus, and toward the previous policy, through
KL(π ‖ π_old).

We follow the implementation of Rudolph et al. (2025,
https://arxiv.org/abs/2502.08938) in
https://github.com/nathanlct/IIG-RL-Benchmark: it keeps PPO's clipping,
estimates the KL from the sampled actions, and keeps the coefficients
constant.
"""

import dataclasses
import functools
from typing import Any

import common
import jax
import jax.numpy as jnp
import ppo
import tyro

import nashbench


@dataclasses.dataclass(frozen=True)
class Config(ppo.Config):
    """MMD's hyperparameters."""

    entropy_coef: float = 0.05
    kl_coef: float = 0.05
    """Weight of KL(π ‖ π_old) in the loss."""


def update(
    config: Config, game: nashbench.Game, state: ppo.State
) -> tuple[ppo.State, jax.Array]:
    """PPO's update, with MMD's loss."""
    return ppo.update(config, game, state, functools.partial(mmd_loss, config))


def mmd_loss(
    config: Config,
    params: Any,
    window: common.Step,
    advantage: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Returns PPO's loss plus `kl_coef` × KL(π ‖ π_old).

    With r = π(a) / π_old(a), the KL is E_old[r log r − (r − 1)], estimated
    from the sampled actions (http://joschu.net/blog/kl-approx.html).
    """
    log_probs = common.log_policy(
        params["actor"], window.observation, window.legal_action_mask
    )
    log_ratio = (
        jnp.take_along_axis(log_probs, window.action[:, None], 1)[:, 0]
        - window.log_prob
    )
    ratio = jnp.exp(log_ratio)
    kl = ratio * log_ratio - (ratio - 1)
    return ppo.ppo_loss(
        config, params, window, advantage, weight
    ) + config.kl_coef * common.weighted_mean(kl, weight)


if __name__ == "__main__":
    common.run(
        "mmd",
        tyro.cli(Config),
        ppo.init,
        common.repeat(update, ppo.decisions_per_update),
        lambda state: common.policy(ppo.actor(state)),
    )
