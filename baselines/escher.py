"""ESCHER: regret minimization with a history value function.

    uv run baselines/escher.py --game leduc_poker --decisions 10000000

ESCHER (https://arxiv.org/abs/2206.04122) estimates regrets with a learned
history value function instead of importance sampling.

We follow the paper's Algorithm 2, not the authors' code, which steps a copy
of the state with every action and weights regrets by the traverser's
sampling probability, because the paper's version needs only sampled
episodes. As in DREAM, one regret network plays both seats and each
iteration updates both.
"""

import dataclasses
import functools
from typing import Any, NamedTuple

import common
import dream
import jax
import jax.numpy as jnp
import optax
import tyro

import nashbench


@dataclasses.dataclass(frozen=True)
class Config(common.Config):
    """ESCHER's hyperparameters, from the paper and the authors' code."""

    traversals: int = 1000
    """Episodes per seat per iteration that estimate regrets."""
    value_traversals: int = 1000
    """Episodes per iteration that train the value network."""
    value_noise: float = 0.01
    """Weight of uniform exploration in episodes that train it."""
    learning_rate: float = 1e-3
    capacity: int = 100_000
    """Capacity of the regret and the average policy reservoirs."""
    regret_batches: int = 5000
    regret_batch_size: int = 2048
    value_batches: int = 5000
    value_batch_size: int = 2048
    average_batches: int = 10_000
    average_batch_size: int = 10_000


class Return(NamedTuple):
    """A sampled return of an action at a history.

    Attributes:
        history: `[2 * observation_shape[0]]` observations of both seats.
        action: `[]` action taken.
        target: `[]` seat 0's return after the action.
    """

    history: jax.Array
    action: jax.Array
    target: jax.Array


class State(NamedTuple):
    """The state of training.

    Attributes:
        network: The regret network, or None before the first iteration.
        regrets: A uniform sample of all estimated regrets.
        policies: A uniform sample of the policies of non-traversers.
        average: The average policy network, or None before it is trained.
        iteration: Iterations so far.
        key: PRNG key.
    """

    network: Any
    regrets: common.Buffer
    policies: common.Buffer
    average: Any
    iteration: int
    key: jax.Array


def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state of training."""
    n, a = game.observation_shape[0], game.num_actions
    sample = dream.Sample(
        jnp.zeros(n), jnp.zeros(a, bool), jnp.zeros(a), jnp.float32(0)
    )
    buffer = common.empty_buffer(sample, config.capacity)
    return State(None, buffer, buffer, None, 0, jax.random.key(config.seed))


def _optimizer(config):
    return optax.adam(config.learning_rate)


def train(
    config: Config, game: nashbench.Game, state: State, decisions: int
) -> tuple[State, int, int]:
    """Runs iterations until they make at least `decisions` decisions.

    Then trains the average policy network.

    Returns:
        The next state, and the numbers of player decisions and of episodes
        that ended.
    """
    made, episodes = 0, 0
    while made < decisions:
        key, value_key, collect_key, fit_key = jax.random.split(state.key, 4)
        iteration = state.iteration + 1
        value, more = _fit_values(config, game, state.network, value_key)
        regrets, policies, even_more = _collect(
            config,
            game,
            state.network,
            value,
            state.regrets,
            state.policies,
            iteration,
            collect_key,
        )
        network = _fit_regrets(config, game, regrets, iteration, fit_key)
        state = State(network, regrets, policies, None, iteration, key)
        made += int(more + even_more)
        episodes += config.value_traversals + 2 * config.traversals
    key, average_key = jax.random.split(state.key)
    average = _fit_average(
        config, game, state.policies, state.iteration, average_key
    )
    return state._replace(average=average, key=key), made, episodes


@functools.partial(jax.jit, static_argnums=(0, 1))
def _fit_values(config, game, network, key):
    """Returns a new value network, trained on episodes of the policy.

    Both seats explore a little, so that every history is reached. The
    target of each action is the return after it, with importance weights
    that correct for exploration at later decisions.

    Returns:
        The network, and the number of decisions in the episodes.
    """
    play_key, init_key, fit_key = jax.random.split(key, 3)
    epsilon = jnp.full((config.value_traversals, 2), config.value_noise)
    steps = dream.play(game, network, epsilon, play_key)

    def backward(next_value, step):
        target = step.reward[:, 0] + next_value
        policy = jnp.take_along_axis(step.policy, step.action[:, None], 1)
        weight = policy[:, 0] / step.behaviour
        return jnp.where(step.valid, weight * target, next_value), target

    _, target = jax.lax.scan(
        backward, jnp.zeros(config.value_traversals), steps, reverse=True
    )
    returns = Return(steps.history, steps.action, target)
    returns = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), returns)
    valid = steps.valid.reshape(-1)
    buffer = common.add_latest(
        common.empty_buffer(jax.tree.map(lambda x: x[0], returns), len(valid)),
        returns,
        valid,
    )

    def loss(value, batch):
        q = common.mlp(value, batch.history)
        q = jnp.take_along_axis(q, batch.action[:, None], 1)[:, 0]
        return jnp.mean(jnp.square(q - batch.target))

    n = game.observation_shape[0]
    value = common.init_network(config, init_key, 2 * n, game.num_actions, 1.0)
    tx = _optimizer(config)
    value = dream.fit(
        tx,
        loss,
        value,
        tx.init(value),
        buffer,
        config.value_batches,
        config.value_batch_size,
        fit_key,
    )[0]
    return value, steps.valid.sum()


@functools.partial(jax.jit, static_argnums=(0, 1))
def _collect(config, game, network, value, regrets, policies, iteration, key):
    """Samples episodes and adds their regrets and policies.

    Returns:
        The regrets, the policies, and the number of decisions.
    """
    play_key, regrets_key, policies_key = jax.random.split(key, 3)
    traverser, epsilon = dream.traversers(config, 1.0)
    steps = dream.play(game, network, epsilon, play_key)
    q = dream.values(value, steps.history, steps.legal_action_mask, traverser)
    regret = q - jnp.sum(steps.policy * q, -1, keepdims=True)
    regret *= steps.legal_action_mask
    traverses = steps.seat == traverser
    weight = jnp.full(traverses.shape, iteration, jnp.float32)
    flat = functools.partial(
        jax.tree.map, lambda x: x.reshape(-1, *x.shape[2:])
    )
    regrets = common.add_to_reservoir(
        regrets,
        flat(
            dream.Sample(
                steps.observation, steps.legal_action_mask, regret, weight
            )
        ),
        (steps.valid & traverses).reshape(-1),
        regrets_key,
    )
    policies = common.add_to_reservoir(
        policies,
        flat(
            dream.Sample(
                steps.observation, steps.legal_action_mask, steps.policy, weight
            )
        ),
        (steps.valid & ~traverses).reshape(-1),
        policies_key,
    )
    return regrets, policies, steps.valid.sum()


def _reweighted(buffer, iteration):
    """Divides weights by the iteration, which keeps the loss's scale."""
    return buffer._replace(
        items=buffer.items._replace(weight=buffer.items.weight / iteration)
    )


@functools.partial(jax.jit, static_argnums=(0, 1))
def _fit_regrets(config, game, regrets, iteration, key):
    """Returns a new regret network, trained on all regrets."""
    init_key, key = jax.random.split(key)
    network = common.init_network(
        config, init_key, game.observation_shape[0], game.num_actions, 1.0
    )
    tx = _optimizer(config)
    return dream.fit(
        tx,
        dream.regression_loss,
        network,
        tx.init(network),
        _reweighted(regrets, iteration),
        config.regret_batches,
        config.regret_batch_size,
        key,
    )[0]


@functools.partial(jax.jit, static_argnums=(0, 1))
def _fit_average(config, game, policies, iteration, key):
    """Returns a new average policy network, trained on all policies.

    As in the authors' code, its loss is the weighted squared error of its
    action probabilities.
    """
    init_key, key = jax.random.split(key)
    average = common.init_network(
        config, init_key, game.observation_shape[0], game.num_actions, 0.01
    )

    def loss(average, batch):
        probs = jnp.exp(
            common.log_policy(
                average, batch.observation, batch.legal_action_mask
            )
        )
        error = jnp.mean(jnp.square(probs - batch.target), 1)
        return jnp.mean(batch.weight * error)

    tx = _optimizer(config)
    return dream.fit(
        tx,
        loss,
        average,
        tx.init(average),
        _reweighted(policies, iteration),
        config.average_batches,
        config.average_batch_size,
        key,
    )[0]


def strategy(state: State) -> nashbench.Policy:
    """Returns the average policy network."""
    if state.average is None:
        return nashbench.uniform_random
    return common.policy(state.average)


if __name__ == "__main__":
    common.run("escher", tyro.cli(Config), init, train, strategy)
