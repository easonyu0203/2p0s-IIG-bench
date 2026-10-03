"""NFSP: neural fictitious self-play.

    uv run baselines/nfsp.py --game leduc_poker --decisions 10000000

NFSP (https://arxiv.org/abs/1603.01121) mixes a DQN best response with an
average policy that imitates it, which is the evaluated strategy. The
defaults and the imitation of action probabilities follow OpenSpiel's Leduc
poker example.

Unlike OpenSpiel, both networks use Adam, as in Rudolph et al. (2025), and
one network of each plays both seats, as in the other baselines.
"""

import dataclasses
import functools
from typing import Any, NamedTuple

import common
import dqn
import jax
import jax.numpy as jnp
import optax
import tyro

import nashbench


@dataclasses.dataclass(frozen=True)
class Config(dqn.Config):
    """NFSP's hyperparameters, and DQN's for the best response."""

    reservoir_capacity: int = 2_000_000
    anticipatory: float = 0.1
    """Probability that a player plays its best response in an episode."""
    sl_learning_rate: float = 0.01
    """Learning rate of the average policy."""


class Behaviour(NamedTuple):
    """A decision of the best response, which the average policy imitates.

    Attributes:
        observation: `[*observation_shape]` observation of the player.
        legal_action_mask: `[num_actions]` legal actions.
        probs: `[num_actions]` the best response's action probabilities.
    """

    observation: jax.Array
    legal_action_mask: jax.Array
    probs: jax.Array


class Envs(NamedTuple):
    """Environments between updates.

    Attributes:
        state: `[N]` environment states.
        timestep: `[N]` timesteps of the next decisions.
        players: Each player's last decision.
        best_response: `[N, 2]` whether each player plays its best response
            in the episode.
    """

    state: nashbench.State
    timestep: nashbench.TimeStep
    players: dqn.Players
    best_response: jax.Array


class State(NamedTuple):
    """The state of training.

    Attributes:
        learner: The best response's Q-network and what trains it.
        average: The average policy's parameters.
        average_opt_state: The average policy's optimizer state.
        reservoir: A uniform sample of the best response's behaviour.
        envs: The environments.
        updates: `[]` updates so far.
        key: PRNG key.
    """

    learner: dqn.Learner
    average: Any
    average_opt_state: Any
    reservoir: common.Buffer
    envs: Envs
    updates: jax.Array
    key: jax.Array


@functools.partial(jax.jit, static_argnums=(0, 1))
def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state of training."""
    learner_key, average_key, envs_key, modes_key, key = jax.random.split(
        jax.random.key(config.seed), 5
    )
    average = common.init_network(
        config, average_key, game.observation_shape[0], game.num_actions, 0.01
    )
    n = config.num_envs
    state, timestep = jax.vmap(game.reset)(jax.random.split(envs_key, n))
    best_response = jax.random.bernoulli(modes_key, config.anticipatory, (n, 2))
    envs = Envs(state, timestep, dqn.players(game, n), best_response)
    behaviour = Behaviour(
        jnp.zeros(game.observation_shape),
        jnp.zeros(game.num_actions, bool),
        jnp.zeros(game.num_actions),
    )
    return State(
        learner=dqn.learner(config, game, learner_key),
        average=average,
        average_opt_state=optax.adam(config.sl_learning_rate).init(average),
        reservoir=common.empty_buffer(behaviour, config.reservoir_capacity),
        envs=envs,
        updates=jnp.int32(0),
        key=key,
    )


def update(
    config: Config, game: nashbench.Game, state: State
) -> tuple[State, jax.Array]:
    """Plays `num_steps` steps in each environment, then trains on them.

    Each network takes a gradient step per `learn_every` player decisions.

    Returns:
        The next state, and the number of episodes that ended.
    """
    key, rollout_key, reservoir_key, learn_key = jax.random.split(state.key, 4)
    envs, (transitions, behaviours, ended) = jax.lax.scan(
        functools.partial(
            _step,
            config,
            game,
            state.learner.q,
            state.average,
            dqn.epsilon(config, state.updates),
        ),
        state.envs,
        jax.random.split(rollout_key, config.num_steps),
        unroll=common.UNROLL,
    )
    (transitions, valid), (behaviours, played) = jax.tree.map(
        lambda x: x.reshape(-1, *x.shape[2:]), (transitions, behaviours)
    )
    state = state._replace(
        learner=state.learner._replace(
            replay=common.add_latest(state.learner.replay, transitions, valid)
        ),
        reservoir=common.add_to_reservoir(
            state.reservoir, behaviours, played, reservoir_key
        ),
        envs=envs,
    )
    state = jax.lax.fori_loop(
        0,
        dqn.decisions_per_update(config) // config.learn_every,
        lambda i, state: _learn(
            config, state, jax.random.fold_in(learn_key, i)
        ),
        state,
    )
    return state._replace(updates=state.updates + 1, key=key), ended.sum()


def _step(config, game, q, average, epsilon, envs, key):
    """Plays one step in each environment.

    Returns:
        The environments; the transitions that the step completes, and
        whether each is valid; the decisions, and whether the best response
        made each; and which episodes ended.
    """
    action_key, modes_key = jax.random.split(key)
    timestep = envs.timestep
    observation, mask = timestep.observation, timestep.legal_action_mask
    env, player = jnp.arange(config.num_envs), timestep.current_player
    best_response = dqn.epsilon_greedy(q, observation, mask, epsilon)
    plays_best_response = envs.best_response[env, player]
    probs = jnp.where(
        plays_best_response[:, None],
        best_response,
        jnp.exp(common.log_policy(average, observation, mask)),
    )
    action = jax.random.categorical(action_key, jnp.log(probs))
    state, next_timestep = jax.vmap(nashbench.auto_reset(game))(
        envs.state, action
    )
    players, transitions = dqn.record(
        envs.players, timestep, action, next_timestep
    )
    # A new episode samples each player's policy again.
    best_responses = jnp.where(
        next_timestep.done[:, None],
        jax.random.bernoulli(
            modes_key, config.anticipatory, envs.best_response.shape
        ),
        envs.best_response,
    )
    envs = Envs(state, next_timestep, players, best_responses)
    behaviour = Behaviour(observation, mask, best_response)
    return envs, (
        transitions,
        (behaviour, plays_best_response),
        next_timestep.done,
    )


def _learn(config, state, key):
    """Takes a gradient step of each network that has enough data."""
    q_key, average_key = jax.random.split(key)
    average, average_opt_state = common.gradient_step(
        state.reservoir.added >= max(config.batch_size, config.min_buffer_size),
        optax.adam(config.sl_learning_rate),
        _average_loss,
        state.average,
        state.average_opt_state,
        common.sample(state.reservoir, config.batch_size, average_key),
    )
    return state._replace(
        learner=dqn.learn(config, state.learner, q_key),
        average=average,
        average_opt_state=average_opt_state,
    )


def _average_loss(average, batch):
    """Returns the cross-entropy of the average policy to the best response."""
    log_probs = common.log_policy(
        average, batch.observation, batch.legal_action_mask
    )
    return -jnp.mean(jnp.sum(batch.probs * log_probs, 1))


if __name__ == "__main__":
    common.run(
        "nfsp",
        tyro.cli(Config),
        init,
        common.repeat(update, dqn.decisions_per_update),
        lambda state: common.policy(state.average),
    )
