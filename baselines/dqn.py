"""DQN best responses, which NFSP and PSRO train.

DQN (https://doi.org/10.1038/nature14236), with the defaults of OpenSpiel's
NFSP Leduc poker example. Unlike single-agent DQN, a transition runs from a
player's decision to its next one, so that it includes the rewards that
arrive while the other player acts.
"""

import dataclasses
import functools
from typing import Any, NamedTuple

import common
import jax
import jax.numpy as jnp
import optax

import nashbench


@dataclasses.dataclass(frozen=True)
class Config(common.Config):
    """DQN's hyperparameters."""

    num_envs: int = 128
    num_steps: int = 8
    """Steps of each environment per update."""
    replay_capacity: int = 200_000
    min_buffer_size: int = 1000
    """Items a buffer needs before its network learns."""
    batch_size: int = 128
    learn_every: int = 64
    """Player decisions per gradient step."""
    learning_rate: float = 0.01
    target_update_every: int = 300
    """Gradient steps between copies of the Q-network to its target."""
    epsilon_start: float = 0.06
    epsilon_end: float = 0.001
    epsilon_decay: int = 20_000_000
    """Player decisions over which ε decays."""


class Transition(NamedTuple):
    """A player's decision and what followed, until its next decision.

    Attributes:
        observation: `[*observation_shape]` observation of the player.
        action: `[]` action taken.
        reward: `[]` the player's reward until its next decision.
        next_observation: `[*observation_shape]` observation at its next
            decision, if not `done`.
        next_legal_action_mask: `[num_actions]` legal actions there.
        done: `[]` whether the episode ended before the next decision.
    """

    observation: jax.Array
    action: jax.Array
    reward: jax.Array
    next_observation: jax.Array
    next_legal_action_mask: jax.Array
    done: jax.Array


class Players(NamedTuple):
    """Each player's last decision in each environment's episode.

    Attributes:
        observation: `[N, 2, *observation_shape]` the player's observation.
        action: `[N, 2]` its action.
        reward: `[N, 2]` its reward since.
        pending: `[N, 2]` whether the player has decided in the episode.
    """

    observation: jax.Array
    action: jax.Array
    reward: jax.Array
    pending: jax.Array


class Learner(NamedTuple):
    """A Q-network and what trains it.

    Attributes:
        q: The Q-network's parameters.
        target: The target network's parameters.
        opt_state: The optimizer's state.
        replay: The latest transitions.
        gradient_steps: `[]` gradient steps so far.
    """

    q: Any
    target: Any
    opt_state: Any
    replay: common.Buffer
    gradient_steps: jax.Array


class Envs(NamedTuple):
    """Environments between updates.

    Attributes:
        state: `[N]` environment states.
        timestep: `[N]` timesteps of the next decisions.
        players: Each player's last decision.
        opponent: `[N]` index of the opponent's actor in each episode, or -1
            until it is sampled.
    """

    state: nashbench.State
    timestep: nashbench.TimeStep
    players: Players
    opponent: jax.Array


class State(NamedTuple):
    """The state of training.

    Attributes:
        learner: The Q-network and what trains it.
        envs: The environments.
        updates: `[]` updates so far.
        key: PRNG key.
    """

    learner: Learner
    envs: Envs
    updates: jax.Array
    key: jax.Array


def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state of training."""
    return new(config, game, jax.random.key(config.seed))


@functools.partial(jax.jit, static_argnums=(0, 1))
def new(config: Config, game: nashbench.Game, key: jax.Array) -> State:
    """Returns a state with a new network and environments."""
    learner_key, envs_key, key = jax.random.split(key, 3)
    n = config.num_envs
    state, timestep = jax.vmap(game.reset)(jax.random.split(envs_key, n))
    envs = Envs(state, timestep, players(game, n), jnp.full(n, -1, jnp.int32))
    return State(learner(config, game, learner_key), envs, jnp.int32(0), key)


def players(game: nashbench.Game, num_envs: int) -> Players:
    """Returns players that haven't decided yet."""
    return Players(
        observation=jnp.zeros((num_envs, 2, *game.observation_shape)),
        action=jnp.zeros((num_envs, 2), jnp.int32),
        reward=jnp.zeros((num_envs, 2)),
        pending=jnp.zeros((num_envs, 2), bool),
    )


def learner(config: Config, game: nashbench.Game, key: jax.Array) -> Learner:
    """Returns a new Q-network, with an empty replay buffer."""
    q = common.init_network(
        config, key, game.observation_shape[0], game.num_actions, 1.0
    )
    observation = jnp.zeros(game.observation_shape)
    mask = jnp.zeros(game.num_actions, bool)
    transition = Transition(
        observation,
        jnp.int32(0),
        jnp.float32(0),
        observation,
        mask,
        jnp.bool(False),
    )
    return Learner(
        q=q,
        target=q,
        opt_state=optax.adam(config.learning_rate).init(q),
        replay=common.empty_buffer(transition, config.replay_capacity),
        gradient_steps=jnp.int32(0),
    )


def decisions_per_update(config: Config) -> int:
    """Returns the player decisions of an update."""
    return config.num_envs * config.num_steps


def actor(state: State) -> Any:
    """Returns an actor that plays the Q-network's greedy actions."""
    *hidden, (w, b) = state.learner.q
    # Scaled values, as logits, put all probability on their maximum.
    return [*hidden, (w * 1e6, b * 1e6)]


def update(
    config: Config,
    game: nashbench.Game,
    state: State,
    opponent: common.Opponent | None = None,
) -> tuple[State, jax.Array]:
    """Plays `num_steps` steps in each environment, then trains on them.

    Args:
        config: The configuration.
        game: The game.
        state: The state of training.
        opponent: Player 1's actors, if the Q-network doesn't play both
            players.

    Returns:
        The next state, and the number of episodes that ended.
    """
    key, rollout_key, learn_key = jax.random.split(state.key, 3)
    envs, (transitions, ended) = jax.lax.scan(
        functools.partial(
            _step,
            game,
            state.learner.q,
            epsilon(config, state.updates),
            opponent,
        ),
        state.envs,
        jax.random.split(rollout_key, config.num_steps),
        unroll=common.UNROLL,
    )
    transitions, valid = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), transitions
    )
    learner = state.learner._replace(
        replay=common.add_latest(state.learner.replay, transitions, valid)
    )
    learner = jax.lax.fori_loop(
        0,
        decisions_per_update(config) // config.learn_every,
        lambda i, learner: learn(
            config, learner, jax.random.fold_in(learn_key, i)
        ),
        learner,
    )
    state = State(learner, envs, state.updates + 1, key)
    return state, ended.sum()


def _step(game, q, epsilon, opponent, envs, key):
    """Plays one step in each environment.

    Returns:
        The environments; the transitions that the step completes, and
        whether each is valid; and which episodes ended.
    """
    action_key, opponent_key = jax.random.split(key)
    timestep = envs.timestep
    observation, mask = timestep.observation, timestep.legal_action_mask
    log_probs = jnp.log(epsilon_greedy(q, observation, mask, epsilon))
    index, learners = envs.opponent, (True, True)
    if opponent is not None:
        index, opponent_log_probs = common.play_opponent(
            opponent, index, opponent_key, observation, mask
        )
        is_learner = (timestep.current_player == 0)[:, None]
        log_probs = jnp.where(is_learner, log_probs, opponent_log_probs)
        learners = (True, False)
    action = jax.random.categorical(action_key, log_probs)
    state, next_timestep = jax.vmap(nashbench.auto_reset(game))(
        envs.state, action
    )
    players, transitions = record(
        envs.players, timestep, action, next_timestep, learners
    )
    index = jnp.where(next_timestep.done, -1, index)
    envs = Envs(state, next_timestep, players, index)
    return envs, (transitions, next_timestep.done)


def record(
    players: Players,
    timestep: nashbench.TimeStep,
    action: jax.Array,
    next_timestep: nashbench.TimeStep,
    learners: tuple[bool, bool] = (True, True),
) -> tuple[Players, tuple[Transition, jax.Array]]:
    """Records a step, and returns the transitions that it completes.

    A player's decision completes at its next decision in the episode, or at
    the end of the episode.

    Args:
        players: Each player's last decision.
        timestep: `[N]` timesteps of the step's decisions.
        action: `[N]` actions taken.
        next_timestep: `[N]` timesteps after the step.
        learners: Whether each player's transitions are valid.

    Returns:
        Each player's last decision, and `[3 * N]` transitions with whether
        each is valid.
    """
    learns = jnp.array(learners)
    # Selecting, rather than indexing by player, fuses into fewer kernels.
    first = timestep.current_player == 0
    acts = jnp.stack([first, ~first], 1)

    def own(x):
        """Returns the acting player's entries of `[N, 2, ...]` `x`."""
        return jnp.where(
            first.reshape(-1, *[1] * (x.ndim - 2)), x[:, 0], x[:, 1]
        )

    # The decision completes the player's last one.
    completed = (
        Transition(
            own(players.observation),
            own(players.action),
            own(players.reward),
            timestep.observation,
            timestep.legal_action_mask,
            jnp.zeros(action.size, bool),
        ),
        own(players.pending & learns),
    )
    players = Players(
        observation=jnp.where(
            acts[..., None], timestep.observation[:, None], players.observation
        ),
        action=jnp.where(acts, action[:, None], players.action),
        reward=jnp.where(acts, 0.0, players.reward) + next_timestep.reward,
        pending=players.pending | acts,
    )
    # The end of an episode completes both players' last decisions.
    done = next_timestep.done[:, None]
    ended = (
        Transition(
            players.observation,
            players.action,
            players.reward,
            jnp.zeros_like(players.observation),
            jnp.zeros(
                (action.size, 2, timestep.legal_action_mask.shape[1]), bool
            ),
            jnp.ones_like(players.pending),
        ),
        done & players.pending & learns,
    )
    transitions = jax.tree.map(
        lambda c, e: jnp.concatenate([c, e.reshape(-1, *e.shape[2:])]),
        completed,
        ended,
    )
    players = players._replace(
        reward=jnp.where(done, 0.0, players.reward),
        pending=players.pending & ~done,
    )
    return players, transitions


def epsilon(config: Config, updates: jax.Array) -> jax.Array:
    """Returns ε after `updates` updates."""
    progress = updates * (decisions_per_update(config) / config.epsilon_decay)
    return config.epsilon_start + jnp.minimum(progress, 1) * (
        config.epsilon_end - config.epsilon_start
    )


def epsilon_greedy(
    q: Any,
    observation: jax.Array,
    legal_action_mask: jax.Array,
    epsilon: jax.Array,
) -> jax.Array:
    """Returns the Q-network's ε-greedy action probabilities."""
    values = common.mlp(q, observation)
    values = jnp.where(legal_action_mask, values, -jnp.inf)
    greedy = jax.nn.one_hot(values.argmax(-1), values.shape[-1])
    uniform = legal_action_mask / legal_action_mask.sum(-1, keepdims=True)
    return (1 - epsilon) * greedy + epsilon * uniform


def learn(config: Config, learner: Learner, key: jax.Array) -> Learner:
    """Takes a gradient step, once the replay buffer has enough items.

    Copies the Q-network to its target every `target_update_every` steps.
    """
    learns = learner.replay.added >= max(
        config.batch_size, config.min_buffer_size
    )
    q, opt_state = common.gradient_step(
        learns,
        optax.adam(config.learning_rate),
        functools.partial(_q_loss, learner.target),
        learner.q,
        learner.opt_state,
        common.sample(learner.replay, config.batch_size, key),
    )
    gradient_steps = learner.gradient_steps + learns
    copies = learns & (gradient_steps % config.target_update_every == 0)
    target = jax.tree.map(
        lambda t, p: jnp.where(copies, p, t), learner.target, q
    )
    return Learner(q, target, opt_state, learner.replay, gradient_steps)


def _q_loss(target, q, batch):
    """Returns the mean squared TD error, bootstrapping from `target`."""
    next_values = common.mlp(target, batch.next_observation)
    next_values = jnp.where(
        batch.next_legal_action_mask, next_values, -jnp.inf
    ).max(1)
    returns = batch.reward + jnp.where(batch.done, 0.0, next_values)
    values = common.mlp(q, batch.observation)
    values = jnp.take_along_axis(values, batch.action[:, None], 1)[:, 0]
    return jnp.mean(jnp.square(values - returns))
