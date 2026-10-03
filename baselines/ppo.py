"""PPO in self-play.

    uv run baselines/ppo.py --game leduc_poker --decisions 10000000

PPO (https://arxiv.org/abs/1707.06347), in which one actor plays both players
and both players' decisions train it.
"""

from collections.abc import Callable
import dataclasses
import functools
from typing import Any, NamedTuple

import common
import jax
import jax.numpy as jnp
import optax
import tyro

import nashbench


@dataclasses.dataclass(frozen=True)
class Config(common.Config):
    """PPO's hyperparameters."""

    num_envs: int = 256
    num_steps: int = 32
    """Steps of each environment per update."""
    epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 2.5e-4
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5


class State(NamedTuple):
    """The state of training.

    Attributes:
        params: The actor's and the critic's parameters.
        opt_state: The optimizer's state.
        envs: The environments.
        key: PRNG key of the next update.
    """

    params: Any
    opt_state: Any
    envs: common.Envs
    key: jax.Array


def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state of training."""
    return new(config, game, jax.random.key(config.seed))


@functools.partial(jax.jit, static_argnums=(0, 1, 3))
def new(
    config: Config,
    game: nashbench.Game,
    key: jax.Array,
    conditions: int = 0,
    params: Any = None,
) -> State:
    """Returns a state with new environments, and new networks if no `params`.

    Decisions hold `conditions` inputs after the observation, which new
    networks take.
    """
    params_key, envs_key, key = jax.random.split(key, 3)
    if params is None:
        params = common.init_actor_critic(config, game, params_key, conditions)
    envs = common.reset(game, envs_key, config.num_envs, conditions)
    return State(params, _optimizer(config).init(params), envs, key)


def actor(state: State) -> Any:
    """Returns the actor's parameters."""
    return state.params["actor"]


def decisions_per_update(config: Config) -> int:
    """Returns the player decisions of an update."""
    return config.num_envs * config.num_steps


def _optimizer(config):
    # Updating one flat vector, rather than every array, saves GPU kernels.
    return optax.flatten(
        optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate, eps=1e-5),
        )
    )


type Loss = Callable[[Any, common.Step, jax.Array, jax.Array], jax.Array]


def update(
    config: Config,
    game: nashbench.Game,
    state: State,
    loss: Loss | None = None,
    opponent: common.Opponent | None = None,
) -> tuple[State, jax.Array]:
    """Plays `num_steps` steps in each environment, then trains on them.

    Args:
        config: The configuration.
        game: The game.
        state: The state of training.
        loss: `(params, decisions, advantages, weights) -> loss`. Defaults
            to PPO's loss.
        opponent: Player 1's actors, if the actor doesn't play both players.

    Returns:
        The next state, and the number of episodes that ended.
    """
    key, rollout_key, train_key = jax.random.split(state.key, 3)
    envs, window, advantage, complete = common.rollout(
        game,
        state.params,
        state.envs,
        rollout_key,
        config.num_steps,
        config.gae_lambda,
        opponent,
    )
    params, opt_state = learn(
        config,
        state.params,
        state.opt_state,
        (window, advantage, complete),
        train_key,
        loss,
    )
    return State(params, opt_state, envs, key), window.done.sum()


def learn(
    config: Config,
    params: Any,
    opt_state: Any,
    rollout: tuple[common.Step, jax.Array, jax.Array],
    key: jax.Array,
    loss: Loss | None = None,
) -> tuple[Any, Any]:
    """Trains for `epochs` epochs on a rollout's decisions.

    Args:
        config: The configuration.
        params: The actor's and the critic's parameters.
        opt_state: The optimizer's state.
        rollout: The steps, advantages, and completeness of `common.rollout`,
            or other weights of decisions that `loss` takes.
        key: PRNG key.
        loss: As in `update`.

    Returns:
        The next parameters and optimizer state.
    """
    if loss is None:
        loss = functools.partial(ppo_loss, config)
    window, advantage, complete = rollout
    optimizer = _optimizer(config)
    samples = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]),
        (window, advantage, complete.astype(jnp.float32)),
    )

    def epoch(carry, key):
        # Sorting random bits once is faster than `jax.random.permutation`,
        # which sorts twice to make ties negligible.
        permutation = jnp.argsort(jax.random.bits(key, (len(samples[1]),)))
        minibatches = jax.tree.map(
            lambda x: x[permutation].reshape(
                config.minibatches, -1, *x.shape[1:]
            ),
            samples,
        )
        return jax.lax.scan(minibatch, carry, minibatches)[0], None

    def minibatch(carry, batch):
        params, opt_state = carry
        updates, opt_state = optimizer.update(
            jax.grad(loss)(params, *batch), opt_state
        )
        return (optax.apply_updates(params, updates), opt_state), None

    return jax.lax.scan(
        epoch, (params, opt_state), jax.random.split(key, config.epochs)
    )[0]


def ppo_loss(
    config: Config,
    params: Any,
    window: common.Step,
    advantage: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Returns PPO's loss, a weighted mean over decisions."""
    return actor_loss(
        config, params["actor"], window, advantage, weight
    ) + config.value_coef * value_loss(
        config, params["critic"], window, advantage, weight
    )


def actor_loss(
    config: Config,
    actor: Any,
    window: common.Step,
    advantage: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Returns PPO's clipped surrogate loss, minus the entropy bonus."""

    def mean(x):
        return common.weighted_mean(x, weight)

    log_probs = common.log_policy(
        actor, window.observation, window.legal_action_mask
    )
    log_prob = jnp.take_along_axis(log_probs, window.action[:, None], 1)[:, 0]
    entropy = mean(-jnp.sum(jnp.exp(log_probs) * log_probs, 1))
    advantage -= mean(advantage)
    advantage /= jnp.sqrt(mean(jnp.square(advantage))) + 1e-8
    ratio = jnp.exp(log_prob - window.log_prob)
    surrogate = jnp.minimum(
        ratio * advantage,
        jnp.clip(ratio, 1 - config.clip, 1 + config.clip) * advantage,
    )
    return -mean(surrogate) - config.entropy_coef * entropy


def value_loss(
    config: Config,
    critic: Any,
    window: common.Step,
    advantage: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Returns PPO's clipped value loss."""
    value = common.value(critic, window.observation)
    target = advantage + window.value
    clipped = window.value + jnp.clip(
        value - window.value, -config.clip, config.clip
    )
    return 0.5 * common.weighted_mean(
        jnp.maximum(jnp.square(value - target), jnp.square(clipped - target)),
        weight,
    )


if __name__ == "__main__":
    common.run(
        "ppo",
        tyro.cli(Config),
        init,
        common.repeat(update, decisions_per_update),
        lambda state: common.policy(actor(state)),
    )
