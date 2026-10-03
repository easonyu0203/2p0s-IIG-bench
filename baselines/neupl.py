"""NeuPL, with PPO best responses.

    uv run baselines/neupl.py --game leduc_poker --decisions 10000000

NeuPL (https://arxiv.org/abs/2202.07415) represents a population with one
network, conditioned on each policy's row of an interaction graph: the
mixture of policies it best-responds to. We follow Algorithm 5 with the
PSRO-Nash graph (Algorithm 3), except:

- Best responses use PPO, not MPO, as in the other baselines, so the critic
  is a value function rather than an action-value function.
- The payoff estimator φ(σ_i, σ_j) regresses episode outcomes, not the
  critic's values, because values averaged over decisions are biased when a
  player's number of decisions varies.
"""

import dataclasses
import functools
from typing import NamedTuple

import common
import jax
import jax.numpy as jnp
import numpy as np
import ppo
import psro
import tyro

import nashbench


@dataclasses.dataclass(frozen=True)
class Config(ppo.Config):
    """NeuPL's hyperparameters, and PPO's."""

    population: int = 8
    """Policies, including the sink."""
    graph_interval: int = 400_000
    """Player decisions between updates of the interaction graph."""
    evaluation_fraction: float = 0.3
    """Probability that an episode is an evaluation match (the paper's ε)."""


class State(NamedTuple):
    """The state of training.

    Attributes:
        ppo_state: PPO's state, whose actor takes a policy's row of `graph`
            after the observation, the critic the opponent's row after that,
            and `params["payoff"]` the two rows.
        graph: `[N, N]` mixture that each policy best-responds to.
        meta_strategy: `[N]` probability of each policy.
        until_update: Player decisions until the next graph update.
    """

    ppo_state: ppo.State
    graph: np.ndarray
    meta_strategy: np.ndarray
    until_update: int


def init(config: Config, game: nashbench.Game) -> State:
    """Returns the initial state, whose policies best-respond to the sink."""
    n = config.population
    actor_key, critic_key, payoff_key, ppo_key = jax.random.split(
        jax.random.key(config.seed), 4
    )
    inputs = game.observation_shape[0]
    params = {
        "actor": common.init_network(
            config, actor_key, inputs + n, game.num_actions, 0.01
        ),
        "critic": common.init_network(
            config, critic_key, inputs + 2 * n, 1, 1.0
        ),
        "payoff": common.init_network(config, payoff_key, 2 * n, 1, 1.0),
    }
    graph = np.zeros((n, n))
    graph[1:, 0] = 1
    return State(
        ppo.new(config, game, ppo_key, 2 * n, params),
        graph,
        np.eye(n)[0],
        config.graph_interval,
    )


def train(
    config: Config, game: nashbench.Game, state: State, decisions: int
) -> tuple[State, int, int | jax.Array]:
    """Trains for at least `decisions` decisions, updating the graph on time.

    Returns:
        The next state, and the numbers of player decisions and of episodes
        that ended.
    """
    made, episodes = 0, 0
    while made < decisions:
        (ppo_state, *_), more, ended = _train_ppo(
            config,
            game,
            (
                state.ppo_state,
                jnp.asarray(state.graph),
                jnp.asarray(_learners(state.graph)),
            ),
            min(decisions - made, state.until_update),
        )
        made += more
        episodes += ended
        state = state._replace(
            ppo_state=ppo_state, until_update=state.until_update - more
        )
        if state.until_update <= 0:
            payoffs = _payoffs(ppo_state.params["payoff"], state.graph)
            state = State(ppo_state, *_solve(payoffs), config.graph_interval)
    return state, made, episodes


def _learners(graph):
    """Returns each policy's probability of learning in a training match.

    It is uniform over distinct policies other than the sink.
    """
    _, first = np.unique(graph[1:], axis=0, return_index=True)
    learners = np.zeros(len(graph))
    learners[first + 1] = 1
    return learners / learners.sum()


def _payoffs(estimator, graph):
    """Returns the estimated `[N, N]` payoff of each policy against each."""
    n = len(graph)
    pairs = jnp.concatenate(
        [jnp.repeat(graph, n, 0), jnp.tile(graph, (n, 1))], 1
    )
    return np.asarray(common.mlp(estimator, pairs), np.float64).reshape(n, n)


def _solve(payoffs):
    """Returns the interaction graph and the meta-strategy of `payoffs`.

    Row k of the graph is a Nash equilibrium over the policies before k, and
    the meta-strategy is one over all policies.
    """
    n = len(payoffs)
    strategies = np.zeros((n + 1, n))
    for k in range(1, n + 1):
        strategies[k, :k] = psro.nash(payoffs[:k, :k])
    return strategies[:n], strategies[n]


def _population(actor, graph):
    """Returns each policy's actor, stacked.

    A row conditions the network by adding row @ W to the first layer's
    bias. The sink's output layer is 0, so it plays uniformly at random.
    """
    (w, b), *layers = actor
    inputs = len(w) - len(graph)

    def policy(row, learns):
        *hidden, (w_out, b_out) = [
            (w[:inputs], b + row @ w[inputs:]),
            *layers,
        ]
        return [*hidden, (w_out * learns, b_out * learns)]

    return jax.vmap(policy)(graph, jnp.arange(len(graph)) > 0)


def _member(actors, k):
    """Returns the `k`th of stacked actors."""
    return jax.tree.map(lambda x: x[k], actors)


def _rollout(config, game, params, graph, learners, envs, key):
    """Plays `num_steps` steps in each environment, as `common.rollout`.

    At the start of each episode, policy i becomes player 0 and j player 1,
    in an evaluation or a training match; `envs.opponent` holds
    (evaluation × N + i) × N + j. Observations hold the rows of the player
    to act and of the other player.

    Returns:
        The environments, and the window, advantages, and `[T + 1, N, 2]`
        weights of decisions for the critic and for the actor.
    """
    n = len(graph)
    step = jax.vmap(nashbench.auto_reset(game))

    def observe(timestep, match):
        """Returns observations, and the policy to act."""
        first = timestep.current_player == 0
        i, j = match // n % n, match % n
        policy, other = jnp.where(first, i, j), jnp.where(first, j, i)
        observation = jnp.concatenate(
            [timestep.observation, graph[policy], graph[other]], 1
        )
        return observation, policy

    def collect(carry, key):
        state, timestep, match = carry
        keys = jax.random.split(key, 5)
        evaluation = jax.random.bernoulli(
            keys[0], config.evaluation_fraction, match.shape
        )
        learner = jax.random.choice(keys[1], n, match.shape, p=learners)
        opponent = jax.random.categorical(keys[2], jnp.log(graph[learner]))
        i, j = jnp.where(
            evaluation,
            jax.random.randint(keys[3], (2, *match.shape), 0, n),
            jnp.stack([learner, opponent]),
        )
        match = jnp.where(match < 0, (evaluation * n + i) * n + j, match)
        observation, policy = observe(timestep, match)
        mask = timestep.legal_action_mask
        log_probs = jnp.where(
            (policy == 0)[:, None],
            jnp.log(jax.vmap(nashbench.uniform_random)(observation, mask)),
            common.log_policy(params["actor"], observation[:, :-n], mask),
        )
        action = jax.random.categorical(keys[4], log_probs)
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
            "valid": jnp.ones_like(timestep.done),
        }
        next_match = jnp.where(next_timestep.done, -1, match)
        return (state, next_timestep, next_match), (decision, match)

    (state, timestep, match), (steps, matches) = jax.lax.scan(
        collect,
        (envs.state, envs.timestep, envs.opponent),
        jax.random.split(key, config.num_steps),
        unroll=common.UNROLL,
    )
    steps = common.Step(
        **steps, value=common.value(params["critic"], steps["observation"])
    )
    window = jax.tree.map(
        lambda p, s: jnp.concatenate([p[None], s]), envs.pending, steps
    )
    advantage, complete, pending = common.advantages(
        window,
        timestep.current_player,
        common.value(params["critic"], observe(timestep, match)[0]),
        config.gae_lambda,
    )
    # A pending decision belongs to the episode that continues.
    matches = jnp.concatenate([envs.opponent[None], matches])
    learns = (window.player == 0) & (matches < n * n)
    weights = jnp.stack([complete, complete & learns], -1)
    envs = common.Envs(state, timestep, pending, match)
    return envs, (window, advantage, weights)


def _loss(config, params, window, advantage, weight):
    """Returns PPO's loss, plus the payoff estimator's squared error.

    The actor takes the decisions of `weight[:, 1]`, and the critic those of
    `weight[:, 0]`. The estimator takes both players' outcomes of the
    decisions that end episodes.
    """
    n = config.population
    own, other = jnp.split(window.observation[:, -2 * n :], 2, 1)
    payoff = jnp.concatenate(
        [
            common.mlp(params["payoff"], jnp.concatenate(rows, 1))
            for rows in ((own, other), (other, own))
        ],
        1,
    )
    outcome = jnp.where(
        (window.player == 0)[:, None], window.reward, window.reward[:, ::-1]
    )
    payoff_loss = 0.5 * common.weighted_mean(
        window.done * jnp.square(payoff - outcome).sum(1), weight[:, 0]
    )
    actor_window = window._replace(observation=window.observation[:, :-n])
    return ppo.actor_loss(
        config, params["actor"], actor_window, advantage, weight[:, 1]
    ) + config.value_coef * (
        ppo.value_loss(
            config, params["critic"], window, advantage, weight[:, 0]
        )
        + payoff_loss
    )


def _update(config, game, carry):
    """Plays and trains the policies for one PPO update."""
    ppo_state, graph, learners = carry
    key, rollout_key, train_key = jax.random.split(ppo_state.key, 3)
    envs, rollout = _rollout(
        config,
        game,
        ppo_state.params,
        graph,
        learners,
        ppo_state.envs,
        rollout_key,
    )
    params, opt_state = ppo.learn(
        config,
        ppo_state.params,
        ppo_state.opt_state,
        rollout,
        train_key,
        functools.partial(_loss, config),
    )
    ppo_state = ppo.State(params, opt_state, envs, key)
    return (ppo_state, graph, learners), rollout[0].done.sum()


_train_ppo = common.repeat(_update, ppo.decisions_per_update)


def _strategy(state):
    """Returns the policies, mixed by the meta-strategy."""
    actors = _population(
        state.ppo_state.params["actor"], jnp.asarray(state.graph)
    )
    support = np.flatnonzero(state.meta_strategy)
    policies = [common.policy(_member(actors, k)) for k in support]
    return nashbench.Mixture(policies, state.meta_strategy[support])


if __name__ == "__main__":
    common.run("neupl", tyro.cli(Config), init, train, _strategy)
