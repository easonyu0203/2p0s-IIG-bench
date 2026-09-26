# nashbench

nashbench is a benchmark for algorithms that learn Nash equilibria in
two-player zero-sum imperfect-information games. Its games are chosen to be
as large as possible while their **exact exploitability** stays feasible to
compute, so you can measure how far a policy is from equilibrium instead of
estimating it. It has three parts:

-   [Games](games/index.md) that run under `jax.jit` and `jax.vmap`, and
    whose rules, actions, and observations match OpenSpiel.
-   A [policy interface](evaluation.md#policies): one function plays both
    seats.
-   [Exact exploitability](evaluation.md#exploitability) of a policy.

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
params = jax.random.normal(jax.random.key(0), (*game.observation_shape, game.num_actions))

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

## Next steps

Read [Conventions](conventions.md) before you write a training loop. It
explains how players, seats, observations, and rewards work, and how to play
many games in parallel.

```{toctree}
:hidden:

conventions
games/index
evaluation
api
contributing
```
