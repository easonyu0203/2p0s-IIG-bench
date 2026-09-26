# Conventions

Multi-agent reinforcement learning has no standard interface for
two-player zero-sum games. This page states the conventions and assumptions
that nashbench follows.

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

## Step API

The API follows the functional style of JAX environments:

```python
state, timestep = game.reset(key)
state, timestep = game.step(state, action)  # action: [2] int32, one per player.
```

The API is *parallel*: `step` takes an action from each player, and each
field of `TimeStep` with a leading axis of size 2 holds a value for each
player.

| Field | Shape | Defined for |
| --- | --- | --- |
| `observation` | `[2, *observation_shape]` | The player to act, while not `done` |
| `legal_action_mask` | `[2, num_actions]` | The player to act, while not `done` |
| `reward` | `[2]` | Both players, always |
| `done` | `[]` | Always |
| `current_player` | `[]` | Always |

`current_player` is the player to act: 0, 1, or `nashbench.BOTH` if both
act at once. Entries that the table doesn't define are unspecified, so don't
use them. `step` ignores the action of a player who doesn't act, and the
outcome of an illegal action is unspecified.

Most games are turn-based. In Goofspiel, both seats bid at once, and
`current_player` is `nashbench.BOTH` at every step.

```{note}
In a turn-based game, if you run the policy for both players at every step,
as in the examples, half of that computation is for the player who isn't
acting. nashbench accepts this cost in exchange for one API for all games.
```

## Observations

A seat's observation is a one-hot encoding of the seat followed by
OpenSpiel's information-state tensor for that seat. In Kuhn and Leduc poker,
OpenSpiel's tensor already starts with the seat, so the observation equals
OpenSpiel's tensor.

As a result, observations have these properties:

-   **One-to-one with information states.** Two histories give a seat the same
    observation if and only if the seat can't tell them apart. This holds
    across both seats.
-   **Absolute.** Encodings don't depend on the observer. For example, a stone
    of seat 0 has the same encoding in both seats' observations.
-   **Perfect recall.** An observation encodes everything the seat has seen,
    so a policy doesn't need memory.

Each game's page lists what every dimension means.

## Actions

Action IDs match OpenSpiel's. `legal_action_mask` marks the legal actions.

## Rewards

`reward[p]` is player `p`'s reward for the last step. Rewards sum to zero
across players and match OpenSpiel's returns. All current games give rewards
only at the end.

In a turn-based game, a player can receive a reward on a step where the
other player acted. For example, in Kuhn poker, if you bet and your opponent
folds, you win on your opponent's step. Credit assignment is up to you: when
you compute player `p`'s return, sum `reward[p]` over all steps, not only
over the steps where `p` acts.

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
policy = jax.jit(jax.vmap(jax.vmap(nashbench.uniform_random)))

key = jax.random.key(0)
state, timestep = jax.vmap(game.reset)(jax.random.split(key, 128))
for _ in range(100):
    key, subkey = jax.random.split(key)
    probs = policy(timestep.observation, timestep.legal_action_mask)
    action = jax.random.categorical(subkey, jnp.log(probs))  # [128, 2]
    state, timestep = step(state, action)
```

This code plays 128 games in parallel for 100 steps. The inner `jax.vmap`
runs the policy for both players of each game.
