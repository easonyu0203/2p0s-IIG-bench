"""Networks, rollouts, and the training loop that baselines share."""

from collections.abc import Callable
import dataclasses
import functools
import itertools
import math
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

import nashbench

# Loops over environment steps unroll this many steps per iteration, which
# spreads an iteration's fixed GPU cost, such as copying its carry, over
# several small steps.
UNROLL = 4


@dataclasses.dataclass(frozen=True)
class Config:
    """Settings of every training run."""

    game: str = "kuhn_poker"
    decisions: int = 10_000_000
    """Player decisions to train for."""
    seed: int = 0
    layers: int = 2
    """Hidden layers of each network."""
    width: int = 256
    """Units per hidden layer."""
    evaluations_per_decade: int = 4
    """Exact evaluations per tenfold increase in player decisions."""


type Train[C, S] = Callable[
    [C, nashbench.Game, S, int], tuple[S, int, int | jax.Array]
]


def run[C: Config, S](
    name: str,
    config: C,
    init: Callable[[C, nashbench.Game], S],
    train: Train[C, S],
    strategy: Callable[[S], nashbench.Policy | nashbench.Mixture],
) -> None:
    """Trains until `config.decisions` player decisions, logging to W&B.

    Logs the exact exploitability of the strategy at the start, once
    training passes each point 10^(k / evaluations_per_decade), and at the
    end.

    Args:
        name: Name of the algorithm.
        config: The run's configuration.
        init: Returns the initial state of training.
        train: Trains for at least the given number of player decisions.
            Returns the next state, and the numbers of player decisions and
            of episodes that ended.
        strategy: Returns the strategy of a state, which is evaluated.
    """
    # Later runs, such as with other seeds, reuse compiled programs.
    jax.config.update(
        "jax_compilation_cache_dir", str(Path.home() / ".cache" / "jax")
    )
    game = nashbench.make(config.game)
    state = init(config, game)
    decisions, episodes = 0, 0
    with wandb.init(
        project="nashbench", group=name, config=dataclasses.asdict(config)
    ) as logger:
        while True:
            exploitability = nashbench.exploitability(game, strategy(state))
            print(
                f"{decisions} decisions: exploitability {exploitability}",
                flush=True,
            )
            logger.log(
                {"episodes": int(episodes), "exploitability": exploitability},
                step=decisions,
            )
            if decisions >= config.decisions:
                break
            k = math.log10(max(decisions, 1)) * config.evaluations_per_decade
            point = 10 ** ((math.floor(k) + 1) / config.evaluations_per_decade)
            state, made, ended = train(
                config,
                game,
                state,
                math.ceil(min(point, config.decisions) - decisions),
            )
            decisions += made
            # Episodes stay on the device until they are logged.
            episodes = episodes + ended


def repeat[C, S](
    update: Callable[[C, nashbench.Game, S], tuple[S, jax.Array]],
    decisions_per_update: Callable[[C], int],
) -> Train[C, S]:
    """Returns `train` for `run`, which repeats an update on the device.

    Args:
        update: Returns the next state, and the number of episodes that
            ended. `jax.jit` compiles it.
        decisions_per_update: Returns the player decisions of an update.
    """

    @functools.partial(jax.jit, static_argnums=(0, 1))
    def loop(config, game, state, num_updates):
        def body(_, carry):
            state, episodes = carry
            state, ended = update(config, game, state)
            return state, episodes + ended

        return jax.lax.fori_loop(
            0, num_updates, body, (state, jnp.zeros((), jnp.int32))
        )

    def train(config, game, state, decisions):
        per_update = decisions_per_update(config)
        num_updates = -(-decisions // per_update)
        state, episodes = loop(config, game, state, num_updates)
        return state, num_updates * per_update, episodes

    return train


def init_actor_critic(
    config: Config, game: nashbench.Game, key: jax.Array, conditions=0
):
    """Returns an actor's and a critic's dense networks.

    Their inputs are an observation and `conditions` more units.
    """
    actor_key, critic_key = jax.random.split(key)
    inputs = game.observation_shape[0] + conditions
    return {
        "actor": init_network(
            config, actor_key, inputs, game.num_actions, 0.01
        ),
        "critic": init_network(config, critic_key, inputs, 1, 1.0),
    }


@functools.partial(jax.jit, static_argnums=(0, 2, 3, 4))
def init_network(
    config: Config,
    key: jax.Array,
    inputs: int,
    outputs: int,
    output_scale: float,
):
    """Returns a dense network with the hidden layers of `config`.

    Every baseline uses this network.
    """
    hidden = [config.width] * config.layers
    return init_mlp(key, [inputs, *hidden, outputs], output_scale)


def init_mlp(key: jax.Array, sizes: list[int], output_scale: float):
    """Returns a dense network's layers, with `sizes` units.

    Weights are orthogonal, with gain √2 in hidden layers and `output_scale`
    in the output layer. Biases are 0.
    """
    scales = [np.sqrt(2)] * (len(sizes) - 2) + [output_scale]
    keys = jax.random.split(key, len(scales))
    return [
        (jax.nn.initializers.orthogonal(scale)(k, (n, m)), jnp.zeros(m))
        for k, scale, (n, m) in zip(
            keys, scales, itertools.pairwise(sizes), strict=True
        )
    ]


def mlp(layers, x: jax.Array) -> jax.Array:
    """Returns a dense network's output, with tanh hidden layers."""
    for w, b in layers[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


def log_policy(actor, observation, legal_action_mask) -> jax.Array:
    """Returns the actor's log-probabilities of actions."""
    logits = mlp(actor, observation)
    # Not -inf, which makes the gradient of the entropy NaN.
    logits = jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    return jax.nn.log_softmax(logits)


def value(critic, observation) -> jax.Array:
    """Returns the critic's value of observations."""
    return mlp(critic, observation)[..., 0]


def policy(actor) -> nashbench.Policy:
    """Returns an actor as a policy, for evaluation."""

    def policy(observation, legal_action_mask):
        return jnp.exp(log_policy(actor, observation, legal_action_mask))

    return policy


def weighted_mean(x: jax.Array, weight: jax.Array) -> jax.Array:
    """Returns the mean of `x` with weights."""
    return jnp.sum(weight * x) / jnp.sum(weight)


class Step(NamedTuple):
    """A player's decision and the environment step it starts.

    Attributes:
        observation: `[*observation_shape]` observation of the player to act.
        legal_action_mask: `[num_actions]` legal actions of that player.
        player: `[]` player to act.
        action: `[]` action taken.
        log_prob: `[]` log-probability of the action.
        value: `[]` critic's value of the observation.
        reward: `[2]` reward of each player on this step.
        done: `[]` whether this step ends the episode.
        valid: `[]` whether this is a decision of the actor being trained,
            rather than padding or an opponent's.
    """

    observation: jax.Array
    legal_action_mask: jax.Array
    player: jax.Array
    action: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    valid: jax.Array


class Envs(NamedTuple):
    """Environments between rollouts.

    Attributes:
        state: `[N]` environment states.
        timestep: `[N]` timesteps of the next decisions.
        pending: `[N]` decision of each environment whose return depends on
            steps after the last rollout, if `valid`.
        opponent: `[N]` index of the opponent's actor in each episode, or -1
            until it is sampled.
    """

    state: nashbench.State
    timestep: nashbench.TimeStep
    pending: Step
    opponent: jax.Array


class Opponent(NamedTuple):
    """Actors that player 1 samples one of at the start of each episode.

    Attributes:
        actors: The actors' parameters, stacked.
        weights: `[K]` probability of each actor.
    """

    actors: Any
    weights: jax.Array


def play_opponent(
    opponent: Opponent,
    index: jax.Array,
    key: jax.Array,
    observation: jax.Array,
    legal_action_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Returns each environment's opponent actor and its log-probabilities.

    Args:
        opponent: The opponent.
        index: `[N]` index of each environment's actor, or -1 to sample one,
            at the start of an episode.
        key: PRNG key.
        observation: `[N, *observation_shape]` observations.
        legal_action_mask: `[N, num_actions]` legal actions.
    """
    sampled = jax.random.choice(
        key, opponent.weights.size, index.shape, p=opponent.weights
    )
    index = jnp.where(index < 0, sampled, index)
    log_probs = jax.vmap(log_policy, (0, None, None))(
        opponent.actors, observation, legal_action_mask
    )
    return index, log_probs[index, jnp.arange(index.size)]


def reset(
    game: nashbench.Game, key: jax.Array, num_envs: int, conditions: int = 0
) -> Envs:
    """Starts `num_envs` environments.

    Decisions hold observations followed by `conditions` more units.
    """
    state, timestep = jax.vmap(game.reset)(jax.random.split(key, num_envs))
    zeros = jnp.zeros(num_envs)
    pending = Step(
        observation=jnp.zeros(
            (num_envs, game.observation_shape[0] + conditions)
        ),
        legal_action_mask=jnp.zeros_like(timestep.legal_action_mask),
        player=jnp.zeros_like(timestep.current_player),
        action=jnp.zeros_like(timestep.current_player),
        log_prob=zeros,
        value=zeros,
        reward=jnp.zeros_like(timestep.reward),
        done=jnp.zeros_like(timestep.done),
        valid=jnp.zeros_like(timestep.done),
    )
    # Strongly typed, as rollouts return it, so `jit` compiles training once.
    opponent = jnp.full(num_envs, -1, jnp.int32)
    return Envs(state, timestep, pending, opponent)


def rollout(
    game: nashbench.Game,
    params: Any,
    envs: Envs,
    key: jax.Array,
    num_steps: int,
    gae_lambda: float,
    opponent: Opponent | None = None,
) -> tuple[Envs, Step, jax.Array, jax.Array]:
    """Plays `num_steps` steps in each environment with an actor-critic.

    The actor plays both players, or only player 0 against an opponent.

    Returns:
        The environments; the `[num_steps + 1, N]` steps, starting with the
        decisions pending from the last rollout; their advantages; and
        whether each is a decision of the actor whose advantage is complete.
    """
    step = jax.vmap(nashbench.auto_reset(game))

    def collect(carry, key):
        state, timestep, opponent_index = carry
        action_key, opponent_key = jax.random.split(key)
        observation = timestep.observation
        mask = timestep.legal_action_mask
        log_probs = log_policy(params["actor"], observation, mask)
        learns = jnp.ones_like(timestep.done)
        if opponent is not None:
            opponent_index, opponent_log_probs = play_opponent(
                opponent, opponent_index, opponent_key, observation, mask
            )
            learns = timestep.current_player == 0
            log_probs = jnp.where(
                learns[:, None], log_probs, opponent_log_probs
            )
        action = jax.random.categorical(action_key, log_probs)
        state, next_timestep = step(state, action)
        decision = {
            "observation": observation,
            "legal_action_mask": mask,
            "player": timestep.current_player,
            "action": action,
            "log_prob": jnp.take_along_axis(log_probs, action[:, None], 1)[
                :, 0
            ],
            "reward": next_timestep.reward,
            "done": next_timestep.done,
            "valid": learns,
        }
        opponent_index = jnp.where(next_timestep.done, -1, opponent_index)
        return (state, next_timestep, opponent_index), decision

    (state, timestep, opponent_index), steps = jax.lax.scan(
        collect,
        (envs.state, envs.timestep, envs.opponent),
        jax.random.split(key, num_steps),
        unroll=UNROLL,
    )
    # The critic doesn't affect actions, so it evaluates all steps at once.
    steps = Step(**steps, value=value(params["critic"], steps["observation"]))
    window = jax.tree.map(
        lambda p, s: jnp.concatenate([p[None], s]), envs.pending, steps
    )
    advantage, complete, pending = advantages(
        window,
        timestep.current_player,
        value(params["critic"], timestep.observation),
        gae_lambda,
    )
    envs = Envs(state, timestep, pending, opponent_index)
    return envs, window, advantage, complete


def advantages(
    window: Step,
    last_player: jax.Array,
    last_value: jax.Array,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array, Step]:
    """Returns the advantage of each decision.

    A decision's reward is its player's reward on every step from the
    decision until the player's next decision in the episode, whoever acts.
    Advantages are undiscounted GAE along each player's decisions. The player
    to act after the window bootstraps from `last_value`; the other player's
    last decision is pending until that player decides again or the episode
    ends.

    Args:
        window: `[T, N]` steps of `N` environments, after the decisions
            pending from the previous window.
        last_player: `[N]` player to act after the window.
        last_value: `[N]` value of that player's observation.
        gae_lambda: λ of GAE.

    Returns:
        `[T, N]` advantages; `[T, N]` whether each step is a decision whose
        advantage is complete; and the `[N]` pending decisions, whose rewards
        are their player's rewards so far.
    """

    def body(carry, step):
        # Of each player: the value and advantage of its next decision, its
        # reward since this step, and whether they are known.
        next_value, next_advantage, reward, complete = carry
        # A step that ends an episode is followed by another episode.
        done = step.done[:, None]
        next_value = jnp.where(done, 0.0, next_value)
        next_advantage = jnp.where(done, 0.0, next_advantage)
        reward = jnp.where(done, 0.0, reward) + step.reward
        complete |= done
        value = step.value[:, None]
        advantage = reward + next_value - value + gae_lambda * next_advantage
        acts = step.valid[:, None] & (step.player[:, None] == jnp.arange(2))
        carry = (
            jnp.where(acts, value, next_value),
            jnp.where(acts, advantage, next_advantage),
            jnp.where(acts, 0.0, reward),
            complete | acts,
        )
        outputs = (advantage, complete, reward)
        return carry, jax.tree.map(
            lambda x: jnp.where(step.player == 0, x[:, 0], x[:, 1]), outputs
        )

    last = last_player[:, None] == jnp.arange(2)
    zeros = jnp.zeros(last.shape)
    carry = (jnp.where(last, last_value[:, None], 0.0), zeros, zeros, last)
    _, (advantage, complete, reward) = jax.lax.scan(
        body, carry, window, reverse=True, unroll=UNROLL
    )
    complete &= window.valid
    # At most one per environment: the last of the player not to act next.
    is_pending = window.valid & ~complete
    index = jnp.argmax(is_pending, 0), jnp.arange(is_pending.shape[1])
    last = jax.tree.map(lambda x: x[index], window)
    reward = jnp.where(is_pending, reward, 0.0).sum(0)
    pending = last._replace(
        reward=jax.nn.one_hot(last.player, 2) * reward[:, None],
        valid=is_pending.any(0),
    )
    return advantage, complete, pending


def gradient_step(
    learns: jax.Array,
    optimizer: optax.GradientTransformation,
    loss: Callable[[Any, Any], jax.Array],
    params: Any,
    opt_state: Any,
    batch: Any,
) -> tuple[Any, Any]:
    """Takes a gradient step on `loss(params, batch)`, if `learns`."""
    grads = jax.grad(loss)(params, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    # Selecting, unlike `lax.cond`, doesn't make the GPU wait for the host to
    # read `learns`.
    return jax.tree.map(
        lambda new, old: jnp.where(learns, new, old),
        (optax.apply_updates(params, updates), new_opt_state),
        (params, opt_state),
    )


class Buffer(NamedTuple):
    """A fixed number of items.

    Attributes:
        items: The items, stacked.
        added: `[]` number of items ever added.
    """

    items: Any
    added: jax.Array


def empty_buffer(item: Any, capacity: int) -> Buffer:
    """Returns an empty buffer for `capacity` items like `item`."""
    items = jax.tree.map(
        lambda x: jnp.zeros((capacity, *x.shape), x.dtype), item
    )
    return Buffer(items, jnp.int32(0))


def add_latest(buffer: Buffer, items: Any, valid: jax.Array) -> Buffer:
    """Adds the valid items, replacing the oldest once the buffer is full."""
    capacity = _capacity(buffer)
    index = (buffer.added + jnp.cumsum(valid) - 1) % capacity
    return _write(buffer, items, valid, jnp.where(valid, index, capacity))


def add_to_reservoir(
    buffer: Buffer, items: Any, valid: jax.Array, key: jax.Array
) -> Buffer:
    """Adds the valid items, keeping a uniform sample of all those added.

    This is reservoir sampling (Vitter's algorithm R). Items of one call that
    pick the same slot replace each other in an unspecified order.
    """
    capacity = _capacity(buffer)
    count = buffer.added + jnp.cumsum(valid)
    slot = jnp.where(
        count <= capacity,
        count - 1,
        jax.random.randint(key, count.shape, 0, count),
    )
    return _write(buffer, items, valid, jnp.where(valid, slot, capacity))


def _capacity(buffer):
    return len(jax.tree.leaves(buffer.items)[0])


def _write(buffer, items, valid, index):
    """Writes items at `index`, unless out of range, and counts valid ones."""
    items = jax.tree.map(
        lambda b, x: b.at[index].set(x, mode="drop"), buffer.items, items
    )
    return Buffer(items, buffer.added + valid.sum())


def sample(buffer: Buffer, batch_size: int, key: jax.Array) -> Any:
    """Returns items sampled uniformly, with replacement."""
    size = jnp.clip(buffer.added, 1, _capacity(buffer))
    index = jax.random.randint(key, (batch_size,), 0, size)
    return jax.tree.map(lambda x: x[index], buffer.items)
