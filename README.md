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
-   **Exact exploitability** of that policy, from milliseconds after warmup
    for poker to under 10 seconds on a GPU for games with 10^10 histories.

## Benchmarks

nashbench includes games with 30 to more than 14 billion terminal histories.
After warmup, exact exploitability takes milliseconds for poker and Goofspiel
and under 10 seconds for the largest games on a GPU. For per-game tree sizes,
evaluation times, simulation throughput, and measurement conditions, see
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
    probs = jax.vmap(policy)(timestep.observation, timestep.legal_action_mask)
    action = jax.random.categorical(subkey, jnp.log(probs))  # One per player.
    state, timestep = game.step(state, action)
print(timestep.reward)  # [2]: sums to zero.

# Compute the policy's exact exploitability: 0 at a Nash equilibrium.
print(nashbench.exploitability(game, policy))
```

Before you train, read [Conventions](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/conventions.md). It explains
how players, seats, observations, and rewards work, and how to play many
games in parallel.

## Documentation

-   [Conventions](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/conventions.md): players, seats, the step API,
    observations, and rewards.
-   [Games](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/index.md): the rules, actions, and observation
    layout of each game.
-   [Benchmarks](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/benchmarks.md): game sizes, exact evaluation times, simulation
    throughput, and measurement conditions.
-   [Evaluation](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/evaluation.md): the policy interface and how
    exploitability is computed.
-   [API](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/api.md): where each public name is defined. Its docstring
    is the reference.
-   [Contributing](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/CONTRIBUTING.md): set up, check, and add a game.

## License

Apache 2.0. See [LICENSE](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/LICENSE).
