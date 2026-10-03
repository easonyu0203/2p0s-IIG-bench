"""PSRO, with PPO (the default) or DQN best responses.

    uv run baselines/psro.py --game leduc_poker --decisions 10000000
    uv run baselines/psro.py dqn --game leduc_poker --decisions 10000000

PSRO (https://arxiv.org/abs/1711.00832) grows a population with best
responses to a Nash equilibrium of the population's meta-game. As in
OpenSpiel's `psro_v2` with `symmetric_game=True`, one population plays both
seats.

Unlike OpenSpiel, which copies a member, each best response starts from new
weights, because PPO started from a nearly deterministic member explores too
little. DQN best responses play their greedy actions, as in Rudolph et al.
(2025).
"""

import dataclasses
import functools
import sys
from typing import Any, NamedTuple

import common
import dqn
import jax
import jax.numpy as jnp
import numpy as np
import ppo
from scipy import optimize
import tyro

import nashbench


@dataclasses.dataclass(frozen=True)
class _Settings:
    oracle_decisions: int = 400_000
    """Player decisions that train each best response."""
    sims_per_entry: int = 1000
    """Episodes that estimate each payoff."""


@dataclasses.dataclass(frozen=True)
class Config(_Settings, ppo.Config):
    """PSRO's hyperparameters, and PPO's for best responses."""


@dataclasses.dataclass(frozen=True)
class DQNConfig(_Settings, dqn.Config):
    """PSRO's hyperparameters, and DQN's for best responses.

    The defaults follow OpenSpiel's PSRO with DQN, which Rudolph et al. use:
    a gradient step per 10 of the best response's own decisions, and a
    target copy every 1000.
    """

    batch_size: int = 32
    learn_every: int = 20
    target_update_every: int = 100


def _oracle(config) -> Any:
    """Returns the module that trains best responses: `ppo` or `dqn`."""
    return dqn if isinstance(config, dqn.Config) else ppo


class State(NamedTuple):
    """The state of training.

    Attributes:
        population: The members' actor parameters.
        payoffs: `[K, K]` mean return of each member against each member.
        meta_strategy: `[K]` probability of each member.
        key: PRNG key.
    """

    population: list[Any]
    payoffs: np.ndarray
    meta_strategy: np.ndarray
    key: jax.Array


def init(config: Config | DQNConfig, game: nashbench.Game) -> State:
    """Returns the initial state, whose population is an untrained policy."""
    oracle_key, key = jax.random.split(jax.random.key(config.seed))
    member = _oracle(config).actor(_new_oracle(config, game, oracle_key))
    return State([member], np.zeros((1, 1)), np.ones(1), key)


def train(
    config: Config | DQNConfig,
    game: nashbench.Game,
    state: State,
    decisions: int,
) -> tuple[State, int, int | jax.Array]:
    """Runs PSRO iterations until they make at least `decisions` decisions.

    Returns:
        The next state, and the numbers of player decisions and of episodes
        that ended.
    """
    made, episodes = 0, 0
    while made < decisions:
        state, more_decisions, more_episodes = _iteration(config, game, state)
        made += more_decisions
        episodes += more_episodes
    return state, made, episodes


def _iteration(config, game, state):
    """Trains a best response to the meta-strategy, and adds it.

    Returns:
        The next state, and the numbers of player decisions and of episodes
        that ended, in training and in estimating payoffs.
    """
    key, oracle_key, payoffs_key = jax.random.split(state.key, 3)
    (oracle, _), decisions, episodes = _train_oracle(
        config,
        game,
        (_new_oracle(config, game, oracle_key), _opponent(state)),
        config.oracle_decisions,
    )
    member = _oracle(config).actor(oracle)
    k = len(state.population)
    keys = jax.random.split(payoffs_key, k)
    returns, made = np.array(
        [
            payoff(game, member, opponent, key, config.sims_per_entry)
            for opponent, key in zip(state.population, keys, strict=True)
        ]
    ).T
    payoffs = np.pad(state.payoffs, (0, 1))
    # A member's payoff against itself is 0, as players switch seats.
    payoffs[k, :k], payoffs[:k, k] = returns, -returns
    population = [*state.population, member]
    state = State(population, payoffs, nash(payoffs), key)
    decisions += int(made.sum())
    return state, decisions, episodes + k * config.sims_per_entry


def _new_oracle(config, game, key):
    """Returns the state of a best response with new weights."""
    return _oracle(config).new(config, game, key)


def _pad(index):
    """Pads `index` to a power of two, by repeating its last element.

    Arrays then take few shapes, which JAX compiles programs for.
    """
    padding = (1 << (len(index) - 1).bit_length()) - len(index)
    return np.pad(index, (0, padding), mode="edge")


def _stack(population, index):
    """Returns the members at `index`, stacked."""
    return jax.tree.map(
        lambda *x: jnp.stack(x), *(population[i] for i in index)
    )


def _opponent(state):
    """Returns the members that the meta-strategy plays, as an opponent.

    Padding members have weight 0.
    """
    support = np.flatnonzero(state.meta_strategy)
    index = _pad(support)
    weights = np.pad(
        state.meta_strategy[support], (0, len(index) - len(support))
    )
    return common.Opponent(
        _stack(state.population, index), jnp.asarray(weights)
    )


def _oracle_update(config, game, carry):
    """Trains the best response for one update against the population."""
    oracle, opponent = carry
    oracle, episodes = _oracle(config).update(
        config, game, oracle, opponent=opponent
    )
    return (oracle, opponent), episodes


_train_oracle = common.repeat(
    _oracle_update,
    lambda config: _oracle(config).decisions_per_update(config),
)


@functools.partial(jax.jit, static_argnums=(0, 4))
def payoff(game, actor, opponent, key, num_episodes):
    """Returns the mean return of `actor` against `opponent`.

    Also returns the number of player decisions in the episodes.
    """
    reset_key, key = jax.random.split(key)
    state, timestep = jax.vmap(game.reset)(
        jax.random.split(reset_key, num_episodes)
    )

    def play(carry):
        state, timestep, key, returns, decisions = carry
        key, action_key = jax.random.split(key)
        log_probs = jnp.where(
            (timestep.current_player == 0)[:, None],
            common.log_policy(
                actor, timestep.observation, timestep.legal_action_mask
            ),
            common.log_policy(
                opponent, timestep.observation, timestep.legal_action_mask
            ),
        )
        action = jax.random.categorical(action_key, log_probs)
        decisions += jnp.sum(~timestep.done)
        state, timestep = jax.vmap(game.step)(state, action)
        returns += timestep.reward[:, 0]
        return state, timestep, key, returns, decisions

    _, _, _, returns, decisions = jax.lax.while_loop(
        lambda carry: ~carry[1].done.all(),
        play,
        (state, timestep, key, jnp.zeros(num_episodes), jnp.int32(0)),
    )
    return returns.mean(), decisions


def nash(payoffs):
    """Returns a Nash equilibrium of the symmetric zero-sum meta-game.

    Maximizes v such that the strategy's payoff against each member is at
    least v.
    """
    k = len(payoffs)
    result = optimize.linprog(
        c=np.r_[np.zeros(k), -1],
        A_ub=np.c_[-payoffs.T, np.ones(k)],
        b_ub=np.zeros(k),
        A_eq=np.r_[np.ones(k), 0][None],
        b_eq=[1],
        bounds=[(0, None)] * k + [(None, None)],
    )
    strategy = np.maximum(result.x[:k], 0)
    return strategy / strategy.sum()


def _strategy(state):
    """Returns the population, mixed by the meta-strategy."""
    policies = [common.policy(actor) for actor in state.population]
    return nashbench.Mixture(policies, state.meta_strategy)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] not in (["ppo"], ["dqn"], ["-h"], ["--help"]):
        args = ["ppo", *args]  # PPO best responses by default.
    config = tyro.extras.subcommand_cli_from_dict(
        {"ppo": Config, "dqn": DQNConfig}, args=args
    )
    name = "psro-dqn" if isinstance(config, DQNConfig) else "psro-ppo"
    common.run(name, config, init, train, _strategy)
