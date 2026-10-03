# Core concepts

This page explains nashbench's environment model and the assumptions that its
API follows.

## Players and seats

nashbench distinguishes two roles:

-   A *seat* is a position in the game, such as the player who moves first in
    poker or plays x in Phantom Tic-Tac-Toe. Seat `i` is OpenSpiel's player
    `i`.
-   A *player* is an index into the arrays that `reset` and `step` take and
    return. For example, player 0 can be the policy you train and player 1 its
    opponent.

At every reset, a fair coin assigns the players to seats. `state.seat[p]` is
the seat of player `p`. Over many episodes, each player plays each seat
equally often, so the game is symmetric between players.

## One policy plays both seats

nashbench assumes that one policy, with one set of parameters, plays both
seats. Observations start with the seat, so the policy can play each seat
differently. This means that you train, store, and evaluate one network per
game.

When player 0 and player 1 use different parameters, for example a learner
and a past version of itself, the seat shuffle still trains the learner on
both seats.

To evaluate a separate strategy for each seat, or a population of policies,
see [Evaluation](evaluation.md#mixtures-and-separate-seats).

## Step API

The API follows the functional style of JAX environments. Players take turns:
`step` takes the action of the player to act, so a policy runs once per step.

```python
state, timestep = game.reset(key)
state, timestep = game.step(state, action)  # action: [] int32.
```

| Field | Shape | Meaning |
| --- | --- | --- |
| `observation` | `[*observation_shape]` | Observation of the player to act, while not `done` |
| `legal_action_mask` | `[num_actions]` | Legal actions of the player to act, while not `done` |
| `reward` | `[2]` | Reward of each player for the last step |
| `done` | `[]` | Whether the episode has ended |
| `current_player` | `[]` | Player to act: 0 or 1 |

The outcome of an illegal action is unspecified.

### Simultaneous moves

In Goofspiel, Oshi-Zumo, and Blotto, both seats move at once. nashbench
takes these moves in turn: seat 0 moves first, then seat 1 moves without
seeing seat 0's move, as in OpenSpiel's `turn_based_simultaneous_game`. Both
versions have the same information states, so they have the same equilibria.

## Observations

A seat's observation is a one-hot encoding of the seat followed by
OpenSpiel's information-state tensor for that seat. In Kuhn poker, Leduc
poker, and Liar's Dice, OpenSpiel's tensor already starts with the seat, so
the observation equals OpenSpiel's tensor. OpenSpiel's Oshi-Zumo has no such
tensor, so nashbench [defines one](games/oshi-zumo.md#observation), and in
limit poker, nashbench [adds the order of public
cards](games/universal-poker.md) that OpenSpiel's tensor lacks.

As a result, observations have these properties:

-   **One-to-one with information states.** Two histories give a seat the same
    observation if and only if the seat can't tell them apart. This holds
    across both seats.
-   **Absolute.** Encodings don't depend on the observer. For example, a stone
    of seat 0 has the same encoding in both seats' observations.
-   **Perfect recall.** An observation encodes everything the seat has seen,
    so a policy doesn't need memory.

Each [game's page](games/index.md) lists what every dimension means.

## Actions

Action IDs match OpenSpiel's. `legal_action_mask` marks the legal actions.

## Rewards

`reward[p]` is player `p`'s reward for the last step. Rewards sum to zero
across players and match OpenSpiel's returns. All current games give rewards
only at the end.

A player can receive a reward on a step where the other player acted. For
example, in Kuhn poker, if you bet and your opponent folds, you win on your
opponent's step. Credit assignment is up to you: when you compute player
`p`'s return, sum `reward[p]` over all steps, not only over the steps where
`p` acts.

## Chance events

All chance events, such as card deals, happen at the start of an episode.
`Game.initial_states` lists every possible outcome with its probability, and
`reset` samples one. Players observe an outcome only when OpenSpiel reveals
it, such as the public card in round 2 of Leduc poker. This process is
equivalent to OpenSpiel's chance nodes.

## End of an episode

After `done` becomes true, `step` leaves the state unchanged and returns zero
rewards. You can therefore step a batch of games together for a fixed number
of steps, even if some finish early.

To keep playing, wrap `step` with `nashbench.auto_reset`. On the step that
ends an episode, the wrapped step returns that step's `reward` and
`done=True`, together with the observation, legal action mask, and
`current_player` of the next episode's first step.

## Play many games at once

To train, step a batch of games with `jax.vmap`, compile the step with
`jax.jit`, and wrap it with `auto_reset` so that games restart as they end:

```python
import jax
import jax.numpy as jnp
import nashbench

game = nashbench.make("leduc_poker")
step = jax.jit(jax.vmap(nashbench.auto_reset(game)))
policy = jax.jit(jax.vmap(nashbench.uniform_random))

key = jax.random.key(0)
state, timestep = jax.vmap(game.reset)(jax.random.split(key, 128))
for _ in range(100):
    key, subkey = jax.random.split(key)
    probs = policy(timestep.observation, timestep.legal_action_mask)
    action = jax.random.categorical(subkey, jnp.log(probs))  # [128]
    state, timestep = step(state, action)
```

This code plays 128 games in parallel for 100 steps. In each game,
`timestep.current_player` tells which player the policy acts for.
