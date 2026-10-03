"""DREAM: deep regret minimization with advantage baselines.

    uv run baselines/dream.py --game leduc_poker --decisions 10000000

DREAM (https://arxiv.org/abs/2006.10410) is outcome-sampling MCCFR with
neural networks, whose learned history baseline reduces variance. The
defaults follow the paper's Leduc poker setup.

Unlike the paper, which alternates seats, one advantage network plays both
seats and each iteration updates both, for fairness with the other
baselines, which share one network across seats.
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
    """DREAM's hyperparameters."""

    traversals: int = 900
    """Episodes per seat per iteration."""
    epsilon: float = 0.5
    """Weight of uniform exploration in the traverser's policy."""
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    advantage_capacity: int = 2_000_000
    advantage_batches: int = 3000
    """Gradient steps that train each advantage network."""
    advantage_batch_size: int = 2048
    baseline_capacity: int = 200_000
    baseline_batches: int = 1000
    """Gradient steps of the baseline per iteration."""
    baseline_batch_size: int = 512


class Step(NamedTuple):
    """A decision in an episode.

    Attributes:
        observation: `[*observation_shape]` observation of the seat to act.
        legal_action_mask: `[num_actions]` its legal actions.
        history: `[2 * observation_shape[0]]` observations of seats 0 and 1.
        seat: `[]` seat to act.
        action: `[]` action taken.
        policy: `[num_actions]` the current policy.
        behaviour: `[]` probability that the sampling policy took the action.
        reward: `[2]` reward of each seat on this step.
        valid: `[]` whether the episode had not ended, so this is a decision.
    """

    observation: jax.Array
    legal_action_mask: jax.Array
    history: jax.Array
    seat: jax.Array
    action: jax.Array
    policy: jax.Array
    behaviour: jax.Array
    reward: jax.Array
    valid: jax.Array


class Sample(NamedTuple):
    """A target for a network that maps observations to action values.

    Attributes:
        observation: `[*observation_shape]` observation.
        legal_action_mask: `[num_actions]` legal actions.
        target: `[num_actions]` target of each action.
        weight: `[]` weight in the loss.
    """

    observation: jax.Array
    legal_action_mask: jax.Array
    target: jax.Array
    weight: jax.Array


class Transition(NamedTuple):
    """A decision, and the next one, for expected SARSA.

    Attributes:
        history: `[2 * observation_shape[0]]` observations of both seats.
        legal_action_mask: `[num_actions]` legal actions.
        action: `[]` action taken.
        reward: `[]` reward of seat 0.
        next_history: Observations of both seats at the next decision.
        next_legal_action_mask: Legal actions at the next decision.
        next_policy: `[num_actions]` policy at the next decision.
        last: `[]` whether the episode ended, so there is no next decision.
    """

    history: jax.Array
    legal_action_mask: jax.Array
    action: jax.Array
    reward: jax.Array
    next_history: jax.Array
    next_legal_action_mask: jax.Array
    next_policy: jax.Array
    last: jax.Array


class State(NamedTuple):
    """The state of training.

    Attributes:
        networks: The advantage network of each iteration so far.
        advantages: A uniform sample of all sampled advantages.
        baseline: The baseline's parameters, which value histories for
            seat 0.
        baseline_opt_state: The baseline's optimizer state.
        transitions: The baseline's latest transitions.
        key: PRNG key.
    """

    networks: list[Any]
    advantages: common.Buffer
    baseline: Any
    baseline_opt_state: Any
    transitions: common.Buffer
    key: jax.Array


def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state of training."""
    baseline_key, key = jax.random.split(jax.random.key(config.seed))
    n, a = game.observation_shape[0], game.num_actions
    baseline = common.init_network(config, baseline_key, 2 * n, a, 1.0)
    sample = Sample(
        jnp.zeros(n), jnp.zeros(a, bool), jnp.zeros(a), jnp.float32(0)
    )
    transition = Transition(
        jnp.zeros(2 * n),
        jnp.zeros(a, bool),
        jnp.int32(0),
        jnp.float32(0),
        jnp.zeros(2 * n),
        jnp.zeros(a, bool),
        jnp.zeros(a),
        jnp.bool(False),
    )
    return State(
        networks=[],
        advantages=common.empty_buffer(sample, config.advantage_capacity),
        baseline=baseline,
        baseline_opt_state=optimizer(config).init(baseline),
        transitions=common.empty_buffer(transition, config.baseline_capacity),
        key=key,
    )


def optimizer(config: Config) -> optax.GradientTransformation:
    """Returns the optimizer of every network."""
    # Updating one flat vector, rather than every array, saves GPU kernels.
    return optax.flatten(
        optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )
    )


def train(
    config: Config, game: nashbench.Game, state: State, decisions: int
) -> tuple[State, int, int]:
    """Runs iterations until they make at least `decisions` decisions.

    Returns:
        The next state, and the numbers of player decisions and of episodes
        that ended.
    """
    made, episodes = 0, 0
    while made < decisions:
        key, collect_key, fit_key, baseline_key = jax.random.split(state.key, 4)
        iteration = len(state.networks) + 1
        advantages, transitions, more = _collect(
            config,
            game,
            state.networks[-1] if state.networks else None,
            state.baseline,
            state.advantages,
            state.transitions,
            iteration,
            collect_key,
        )
        network = _fit_advantages(config, game, advantages, iteration, fit_key)
        baseline, baseline_opt_state = _fit_baseline(
            config,
            state.baseline,
            state.baseline_opt_state,
            transitions,
            baseline_key,
        )
        state = State(
            [*state.networks, network],
            advantages,
            baseline,
            baseline_opt_state,
            transitions,
            key,
        )
        made += int(more)
        episodes += 2 * config.traversals
    return state, made, episodes


def current_policy(network, observation, legal_action_mask) -> jax.Array:
    """Returns the policy that regret matching gives an advantage network.

    Plays uniformly at random if `network` is None.
    """
    if network is None:
        total = legal_action_mask.sum(-1, keepdims=True)
        return legal_action_mask / jnp.maximum(total, 1)
    advantages = common.mlp(network, observation)
    positive = jnp.where(legal_action_mask, jnp.maximum(advantages, 0), 0)
    total = positive.sum(-1, keepdims=True)
    # With no positive advantage, the best action has probability 1.
    best = jnp.argmax(jnp.where(legal_action_mask, advantages, -jnp.inf), -1)
    return jnp.where(
        total > 0,
        positive / jnp.where(total > 0, total, 1),
        jax.nn.one_hot(best, advantages.shape[-1]),
    )


def play(
    game: nashbench.Game, network, epsilon: jax.Array, key: jax.Array
) -> Step:
    """Plays episodes to the end with an advantage network's policy.

    Args:
        game: The game.
        network: The advantage network, or None to play uniformly.
        epsilon: `[N, 2]` weight of uniform exploration of each seat in each
            of `N` episodes.
        key: PRNG key.

    Returns:
        `[T, N]` steps, where `T` bounds the decisions in an episode.
    """
    num_episodes = len(epsilon)
    reset_key, key = jax.random.split(key)
    env = jnp.arange(num_episodes)

    def step(carry, key):
        state, timestep = carry
        game_state = state.game_state
        seat = game_state.current_seat
        valid = ~timestep.done
        mask = timestep.legal_action_mask & valid[:, None]
        policy = current_policy(network, timestep.observation, mask)
        uniform = mask / jnp.maximum(mask.sum(1, keepdims=True), 1)
        explore = epsilon[env, seat][:, None]
        behaviour = (1 - explore) * policy + explore * uniform
        action = jax.random.categorical(key, jnp.log(behaviour))
        history = jnp.concatenate(
            [
                jax.vmap(game.observe)(game_state, jnp.full_like(seat, s))
                for s in (0, 1)
            ],
            1,
        )
        next_state, next_timestep = jax.vmap(game.step)(state, action)
        # `seat` is its own inverse, so it maps player rewards to seats.
        reward = jnp.take_along_axis(next_timestep.reward, state.seat, 1)
        decision = Step(
            observation=timestep.observation,
            legal_action_mask=mask,
            history=history,
            seat=seat,
            action=action,
            policy=policy,
            behaviour=behaviour[env, action],
            reward=reward,
            valid=valid,
        )
        return (next_state, next_timestep), decision

    start = jax.vmap(game.reset)(jax.random.split(reset_key, num_episodes))
    # A seat decides at most once per level of its information sets.
    num_steps = len(game.sequence_form.levels)
    return jax.lax.scan(step, start, jax.random.split(key, num_steps))[1]


def traversers(config: Config, epsilon: float) -> tuple[jax.Array, jax.Array]:
    """Returns the traverser of each episode, half seat 0 and half seat 1.

    Also returns the `[N, 2]` exploration of the seats: `epsilon` for the
    traverser and none for the other seat.
    """
    traverser = jnp.arange(2 * config.traversals) % 2
    return traverser, epsilon * jax.nn.one_hot(traverser, 2)


def values(network, history, legal_action_mask, traverser) -> jax.Array:
    """Returns a history network's action values for the traverser.

    The network values histories for seat 0; seat 1's values are their
    negatives, as the game is zero-sum.
    """
    q = common.mlp(network, history) * (1 - 2 * traverser)[:, None]
    return jnp.where(legal_action_mask, q, 0)


@functools.partial(jax.jit, static_argnums=(0, 1))
def _collect(
    config, game, network, baseline, advantages, transitions, iteration, key
):
    """Samples episodes and adds their advantages and transitions.

    Returns:
        The advantages, the transitions, and the number of decisions.
    """
    play_key, reservoir_key = jax.random.split(key)
    traverser, epsilon = traversers(config, config.epsilon)
    steps = play(game, network, epsilon, play_key)
    traverses = steps.valid & (steps.seat == traverser)
    q = values(baseline, steps.history, steps.legal_action_mask, traverser)

    def backward(next_value, x):
        step, q = x
        # Baseline-corrected values of the actions and of the history.
        sampled = jax.nn.one_hot(step.action, q.shape[-1])
        reward = jnp.take_along_axis(step.reward, traverser[:, None], 1)[:, 0]
        q_action = jnp.sum(sampled * q, 1)
        correction = (reward + next_value - q_action) / step.behaviour
        u = q + sampled * correction[:, None]
        value = jnp.sum(step.policy * u, 1)
        advantage = (u - value[:, None]) * step.legal_action_mask
        return jnp.where(step.valid, value, next_value), advantage

    _, advantage = jax.lax.scan(
        backward, jnp.zeros(len(traverser)), (steps, q), reverse=True
    )
    # The traverser's sampling probability up to each decision, relative to
    # uniform sampling, which DREAM divides each advantage's weight by.
    num_legal = steps.legal_action_mask.sum(-1)
    factor = jnp.where(traverses, steps.behaviour * num_legal, 1.0)
    reach = jnp.cumprod(factor, 0) / factor
    samples = Sample(
        steps.observation, steps.legal_action_mask, advantage, iteration / reach
    )
    flat = functools.partial(
        jax.tree.map, lambda x: x.reshape(-1, *x.shape[2:])
    )
    advantages = common.add_to_reservoir(
        advantages, flat(samples), traverses.reshape(-1), reservoir_key
    )

    def shift(x):
        return jnp.concatenate([x[1:], jnp.zeros_like(x[:1])])

    transition = Transition(
        history=steps.history,
        legal_action_mask=steps.legal_action_mask,
        action=steps.action,
        reward=steps.reward[..., 0],
        next_history=shift(steps.history),
        next_legal_action_mask=shift(steps.legal_action_mask),
        next_policy=shift(steps.policy),
        last=~shift(steps.valid),
    )
    transitions = common.add_latest(
        transitions, flat(transition), steps.valid.reshape(-1)
    )
    return advantages, transitions, steps.valid.sum()


def fit(
    optimizer: optax.GradientTransformation,
    loss,
    params: Any,
    opt_state: Any,
    buffer: common.Buffer,
    batches: int,
    batch_size: int,
    key: jax.Array,
) -> tuple[Any, Any]:
    """Takes `batches` gradient steps on batches sampled from `buffer`."""

    def body(i, carry):
        params, opt_state = carry
        batch = common.sample(buffer, batch_size, jax.random.fold_in(key, i))
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    return jax.lax.fori_loop(0, batches, body, (params, opt_state))


def regression_loss(network, batch: Sample) -> jax.Array:
    """Returns the weighted mean squared error of the network's outputs."""
    prediction = common.mlp(network, batch.observation)
    error = jnp.where(batch.legal_action_mask, prediction - batch.target, 0)
    return jnp.mean(batch.weight * jnp.mean(jnp.square(error), 1))


@functools.partial(jax.jit, static_argnums=(0, 1))
def _fit_advantages(config, game, advantages, iteration, key):
    """Returns a new advantage network, trained on all advantages."""
    init_key, key = jax.random.split(key)
    network = common.init_network(
        config, init_key, game.observation_shape[0], game.num_actions, 1.0
    )
    # Dividing by the iteration keeps the loss's scale, as in ESCHER's code.
    advantages = advantages._replace(
        items=advantages.items._replace(
            weight=advantages.items.weight / iteration
        )
    )
    tx = optimizer(config)
    return fit(
        tx,
        regression_loss,
        network,
        tx.init(network),
        advantages,
        config.advantage_batches,
        config.advantage_batch_size,
        key,
    )[0]


@functools.partial(jax.jit, static_argnums=0)
def _fit_baseline(config, baseline, opt_state, transitions, key):
    """Trains the baseline further by expected SARSA."""

    def loss(baseline, batch):
        q = common.mlp(baseline, batch.history)
        q = jnp.take_along_axis(q, batch.action[:, None], 1)[:, 0]
        next_q = jnp.where(
            batch.next_legal_action_mask,
            common.mlp(baseline, batch.next_history),
            0,
        )
        next_value = jnp.sum(batch.next_policy * next_q, 1)
        target = batch.reward + jnp.where(batch.last, 0, next_value)
        return jnp.mean(jnp.square(q - jax.lax.stop_gradient(target)))

    return fit(
        optimizer(config),
        loss,
        baseline,
        opt_state,
        transitions,
        config.baseline_batches,
        config.baseline_batch_size,
        key,
    )


def policy(network) -> nashbench.Policy:
    """Returns the policy of an advantage network, for evaluation."""
    return functools.partial(current_policy, network)


def strategy(state: State) -> nashbench.Policy | nashbench.Mixture:
    """Returns the networks of all iterations, iteration t with weight t."""
    if not state.networks:
        return nashbench.uniform_random
    return nashbench.Mixture(
        [policy(network) for network in state.networks],
        range(1, len(state.networks) + 1),
    )


if __name__ == "__main__":
    common.run("dream", tyro.cli(Config), init, train, strategy)
