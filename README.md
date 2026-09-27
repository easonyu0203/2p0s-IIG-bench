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
-   **Exact exploitability** of that policy, from milliseconds for poker to
    under 10 seconds on a GPU for games with 10^10 histories.

| Game | Information sets | Terminal histories | Exploitability (GPU) |
| --- | --- | --- | --- |
| [`kuhn_poker`](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/kuhn_poker.md) | 12 | 30 | 1.3 ms |
| [`leduc_poker`](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/leduc_poker.md) | 936 | 5,520 | 1.2 ms |
| [`goofspiel`](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/goofspiel.md) | 23,050,572 | 373,248,000 | 83 ms |
| [`phantom_ttt`](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/phantom_ttt.md) | 5,990,669 | 9,829,101,024 | 5.4 s |
| [`phantom_ttt_abrupt`](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/phantom_ttt.md) | 23,310,269 | 13,578,403,440 | 8.1 s |
| [`dark_hex3`](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/dark_hex.md) | 6,072,917 | 9,469,697,760 | 5.3 s |
| [`dark_hex3_abrupt`](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/dark_hex.md) | 27,325,277 | 14,663,760,672 | 9.1 s |

Information sets are counted for both seats. Times are per call on one
NVIDIA RTX A6000, the GPU for every measurement in these docs, after the
first call for a game; see
[Evaluation](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/evaluation.md#how-its-computed).

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
-   [Games](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/games/README.md): the rules, actions, and observation
    layout of each game.
-   [Evaluation](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/evaluation.md): the policy interface and how
    exploitability is computed.
-   [API](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/docs/api.md): where each public name is defined. Its docstring
    is the reference.
-   [Contributing](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/CONTRIBUTING.md): set up, check, and add a game.

## License

Apache 2.0. See [LICENSE](https://github.com/easonyu0203/2p0s-IIG-bench/blob/main/LICENSE).
