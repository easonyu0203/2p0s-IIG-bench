# nashbench

nashbench is a benchmark for algorithms that learn Nash equilibria in
two-player zero-sum imperfect-information games. Its games are chosen to be
as large as possible while their **exact exploitability** stays feasible to
compute.

-   **Games in JAX**, which run under `jax.jit` and `jax.vmap`. Their rules,
    actions, and observations match
    [OpenSpiel](https://github.com/google-deepmind/open_spiel).
-   **One policy interface**: a function from an observation and its legal
    actions to action probabilities, which plays both seats.
-   **Exact exploitability** of that policy, from TBD after warmup for
    poker to TBD on a GPU for games with 10^10 histories.
    It also evaluates a policy per seat, and populations of policies.

## Benchmarks

nashbench includes games with 30 to more than 14 billion terminal histories.
After warmup, exact exploitability takes TBD for poker, Goofspiel, Liar's
Dice, and Oshi-Zumo, and TBD for the phantom games on a GPU. For per-game
tree sizes, evaluation times, simulation throughput, and measurement
conditions, see
[Benchmark results](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/benchmarks.md).

## Install

```sh
uv add nashbench  # Or: pip install nashbench
```

## Quickstart

```python
import jax
import jax.numpy as jnp
import nashbench

game = nashbench.make("leduc_poker")

# A policy maps one player's observation to action probabilities.
params = jax.random.normal(
    jax.random.key(0), (*game.observation_shape, game.num_actions)
)


def policy(observation, legal_action_mask):
    logits = observation @ params  # Your network goes here.
    return jax.nn.softmax(jnp.where(legal_action_mask, logits, -jnp.inf))


# Play one game. The same policy acts for both players.
key = jax.random.key(1)
state, timestep = game.reset(key)
while not timestep.done:
    key, subkey = jax.random.split(key)
    probs = policy(timestep.observation, timestep.legal_action_mask)
    action = jax.random.categorical(subkey, jnp.log(probs))
    state, timestep = game.step(state, action)
print(timestep.reward)  # [2]: sums to zero.

# Compute the policy's exact exploitability: 0 at a Nash equilibrium.
print(nashbench.exploitability(game, policy))
```

Before you train, read [Core concepts](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/core-concepts.md). It explains
players, seats, observations, rewards, and how to play many games in
parallel.

## Documentation

-   [Core concepts](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/core-concepts.md): players, seats, the step API,
    observations, and rewards.
-   [Evaluation](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/evaluation.md): the policy interface, mixtures,
    and how exploitability is computed.
-   [Games](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/index.md): the rules, actions, and observation
    layout of each game.
-   [Benchmarks](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/benchmarks.md): game sizes, exact evaluation times, simulation
    throughput, and measurement conditions.
-   [API reference](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/api-reference.md): every name that `nashbench` exports
    and the source containing its docstring.
-   [Contributing](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/CONTRIBUTING.md): set up, check, and add a game.

## License

Apache 2.0. See [LICENSE](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/LICENSE).
