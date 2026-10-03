"""R-NaD: regularized Nash dynamics.

    uv run baselines/rnad.py --game leduc_poker --decisions 10000000

R-NaD (https://arxiv.org/abs/2206.15378) solves the game regularized toward
a policy π_reg with NeuRD and v-trace, moves π_reg to the solution, and
repeats. We follow OpenSpiel's implementation.
"""

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
    """R-NaD's hyperparameters."""

    num_envs: int = 16
    num_steps: int = 64
    """Steps of each environment per learner step."""
    learning_rate: float = 3e-4
    clip_gradient: float = 10_000
    """Bound on each update, after Adam."""
    target_network_avg: float = 0.001
    """Rate at which the target network follows the network."""
    reg_interval: int = 1000
    """Learner steps between updates of π_reg."""
    eta: float = 0.2
    """Weight of the reward transform."""
    c_vtrace: float = 1.0
    nerd_beta: float = 2.0
    """Bound on centered logits, beyond which NeuRD stops pushing them."""
    nerd_clip: float = 10_000


class State(NamedTuple):
    """The state of training.

    Attributes:
        params: The actor's and the critic's parameters.
        target: The target network's parameters.
        reg: The actor's parameters of π_reg.
        prev_reg: Those of the π_reg before it.
        opt_state: The optimizer's state.
        steps: `[]` learner steps so far.
        key: PRNG key.
    """

    params: Any
    target: Any
    reg: Any
    prev_reg: Any
    opt_state: Any
    steps: jax.Array
    key: jax.Array


@functools.partial(jax.jit, static_argnums=(0, 1))
def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state, whose networks are all the same."""
    params_key, key = jax.random.split(jax.random.key(config.seed))
    params = common.init_actor_critic(config, game, params_key)
    actor = params["actor"]
    return State(
        params,
        params,
        actor,
        actor,
        _optimizer(config).init(params),
        jnp.zeros((), jnp.int32),
        key,
    )


def _optimizer(config):
    # OpenSpiel's Adam has no momentum.
    return optax.chain(
        optax.scale_by_adam(b1=0.0, b2=0.999, eps=10e-8),
        optax.scale(-config.learning_rate),
        optax.clip(config.clip_gradient),
    )


def update(
    config: Config, game: nashbench.Game, state: State
) -> tuple[State, jax.Array]:
    """Plays new episodes, and takes a learner step on those that end.

    Returns:
        The next state, and the number of episodes that ended.
    """
    key, rollout_key = jax.random.split(state.key)
    window = _rollout(config, game, state, rollout_key)
    # π_reg is blended from the previous one over the first half of an
    # iteration, and updated to the target network at its end.
    progress = state.steps % config.reg_interval
    alpha = jnp.minimum(2 * progress / config.reg_interval, 1.0)
    grads = jax.grad(_loss, 1)(config, state.params, state, alpha, window)
    updates, opt_state = _optimizer(config).update(grads, state.opt_state)
    params = optax.apply_updates(state.params, updates)
    target = jax.tree.map(
        lambda t, p: t + config.target_network_avg * (p - t),
        state.target,
        params,
    )
    ends = progress == config.reg_interval - 1
    reg, prev_reg = jax.tree.map(
        lambda new, old: jnp.where(ends, new, old),
        (target["actor"], state.reg),
        (state.reg, state.prev_reg),
    )
    state = State(
        params, target, reg, prev_reg, opt_state, state.steps + 1, key
    )
    return state, window.done.sum()


def _rollout(config, game, state, key):
    """Plays `num_steps` steps of `num_envs` new episodes with the actor.

    Returns:
        The `[num_steps, num_envs]` decisions, with the target network's
        values, which are valid if their episode ends.
    """
    reset_key, key = jax.random.split(key)
    env_state, timestep = jax.vmap(game.reset)(
        jax.random.split(reset_key, config.num_envs)
    )
    step = jax.vmap(nashbench.auto_reset(game))

    def collect(carry, key):
        env_state, timestep = carry
        log_probs = common.log_policy(
            state.params["actor"],
            timestep.observation,
            timestep.legal_action_mask,
        )
        action = jax.random.categorical(key, log_probs)
        env_state, next_timestep = step(env_state, action)
        decision = {
            "observation": timestep.observation,
            "legal_action_mask": timestep.legal_action_mask,
            "player": timestep.current_player,
            "action": action,
            "log_prob": jnp.take_along_axis(log_probs, action[:, None], 1)[
                :, 0
            ],
            "reward": next_timestep.reward,
            "done": next_timestep.done,
        }
        return (env_state, next_timestep), decision

    _, steps = jax.lax.scan(
        collect,
        (env_state, timestep),
        jax.random.split(key, config.num_steps),
        unroll=common.UNROLL,
    )
    ends = jnp.cumsum(steps["done"][::-1], 0)[::-1] > 0
    return common.Step(
        **steps,
        value=common.value(state.target["critic"], steps["observation"]),
        valid=ends,
    )


def _loss(config, params, state, alpha, window):
    """Returns R-NaD's loss: the critic's and NeuRD's, of both players."""
    mask = window.legal_action_mask
    logits = common.mlp(params["actor"], window.observation)

    def log_policy(actor):
        log_probs = common.log_policy(actor, window.observation, mask)
        return jnp.where(mask, log_probs, 0.0)

    log_probs = log_policy(params["actor"])
    policy = jax.lax.stop_gradient(jnp.exp(log_probs) * mask)
    log_ratio = jax.lax.stop_gradient(
        log_probs
        - alpha * log_policy(state.reg)
        - (1 - alpha) * log_policy(state.prev_reg)
    )
    value = common.value(params["critic"], window.observation)
    centered = logits - jnp.sum(logits * mask, -1, keepdims=True) / jnp.sum(
        mask, -1, keepdims=True
    )

    def player_loss(player):
        target, q = _v_trace(config, window, policy, log_ratio, player)
        acts = window.valid & (window.player == player)
        advantage = q - jnp.sum(policy * q, -1, keepdims=True)
        advantage = jnp.clip(advantage, -config.nerd_clip, config.nerd_clip)
        # NeuRD stops pushing logits that are beyond ±nerd_beta.
        force = jnp.where(
            centered > -config.nerd_beta, jnp.minimum(advantage, 0), 0
        ) + jnp.where(centered < config.nerd_beta, jnp.maximum(advantage, 0), 0)
        return _mean(jnp.square(value - target), acts) - _mean(
            jnp.sum(mask * centered * force, -1), acts
        )

    # Batching the players runs their v-traces in one scan.
    return jax.vmap(player_loss)(jnp.arange(2)).sum()


def _mean(x, mask):
    """Returns the mean of `x` where `mask`, or 0."""
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1)


def _v_trace(config, window, policy, log_ratio, player):
    """Returns v-trace's value targets and NeuRD's action values.

    This is OpenSpiel's `v_trace` for `player`: undiscounted, with λ = 1 and
    ρ = ∞, and with rewards transformed by -η log(π / π_reg), which `player`
    receives for its own actions and the other player's negated.

    Returns:
        `[T, N]` value targets and `[T, N, num_actions]` action values, which
        are 0 except at `player`'s valid decisions.
    """
    one_hot = jax.nn.one_hot(window.action, policy.shape[-1])
    mu = jnp.exp(window.log_prob)
    ratio = jnp.sum(policy * one_hot, -1) / mu
    sign = jnp.where(window.player == player, 1.0, -1.0)
    entropy_reward = -config.eta * jnp.sum(policy * log_ratio, -1) * sign
    log_ratio_reward = -config.eta * log_ratio * sign[..., None]
    acts = window.player == player
    zeros = jnp.zeros_like(mu[0])
    # The reward since `player`'s next decision, corrected and not; the
    # value and target of that decision; and the importance weight to it.
    init = (zeros, zeros, zeros, zeros, jnp.ones_like(zeros))

    def body(carry, x):
        r, cs, mu, v, entropy_r, log_r, one_hot, acts, done, valid = x
        # Later steps belong to another episode.
        carry = jax.tree.map(
            lambda c, i: jnp.where(done | ~valid, i, c), carry, init
        )
        reward, uncorrected, next_value, next_target, weight = carry
        uncorrected = r + uncorrected + entropy_r
        discounted = r + reward
        target = (
            v
            + cs * weight * (uncorrected + next_value - v)
            + jnp.minimum(config.c_vtrace, cs * weight)
            * (next_target - next_value)
        )
        q = (
            v[:, None]
            + log_r
            + one_hot * ((discounted + weight * next_target - v) / mu)[:, None]
        )
        ours = (zeros, zeros, v, target, jnp.ones_like(zeros))
        theirs = (
            entropy_r + cs * discounted,
            uncorrected,
            next_value,
            next_target,
            cs * weight,
        )
        carry = jax.tree.map(lambda a, b: jnp.where(acts, a, b), ours, theirs)
        out = acts & valid
        return carry, (
            jnp.where(out, target, 0.0),
            jnp.where(out[:, None], q, 0.0),
        )

    _, (target, q) = jax.lax.scan(
        body,
        init,
        (
            window.reward[..., player],
            ratio,
            mu,
            window.value,
            entropy_reward,
            log_ratio_reward,
            one_hot,
            acts,
            window.done,
            window.valid,
        ),
        reverse=True,
        unroll=common.UNROLL,
    )
    return target, q


if __name__ == "__main__":
    common.run(
        "rnad",
        tyro.cli(Config),
        init,
        common.repeat(
            update, lambda config: config.num_envs * config.num_steps
        ),
        # As in OpenSpiel, the target network is the evaluated policy.
        lambda state: common.policy(state.target["actor"]),
    )
