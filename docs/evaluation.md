# Evaluation

nashbench evaluates a policy by its exploitability: how much a best
response gains against it.

## Policies

A policy is a function from one player's observation and legal action mask to
action probabilities:

```python
def policy(observation, legal_action_mask):
    """Returns [num_actions] probabilities, zero for illegal actions."""
```

The function takes a single, unbatched observation. It must be
JAX-transformable, because evaluation calls it under `jax.jit` and
`jax.vmap`. One policy plays both seats; see
[Core concepts](core-concepts.md#one-policy-plays-both-seats).

To turn a network into a policy, close over its parameters and mask its
logits. For example, with Flax:

```python
def policy(observation, legal_action_mask):
    logits = network.apply(params, observation)
    return jax.nn.softmax(jnp.where(legal_action_mask, logits, -jnp.inf))
```

`nashbench.uniform_random` is a policy that plays uniformly at random. Its
exploitability is a useful baseline.

## Mixtures and separate seats

Evaluation can also take a strategy for each seat, as a tuple
`(strategy_0, strategy_1)`. The tuple is indexed by seat, not by player. A
strategy is a policy or a `nashbench.Mixture`.

A mixture is a population of policies with weights, such as a PSRO
meta-strategy. At the start of each episode, a seat samples one of the
policies, with probability proportional to its weight, and plays it for the
whole episode. Seats sample independently. At an information state $I$, a
mixture therefore plays

$$
\pi(a \mid I) = \frac{\sum_k w_k \, r_k(I) \, \pi_k(a \mid I)}{\sum_k w_k \, r_k(I)},
$$

where $r_k(I)$ is the probability that policy $k$'s own actions lead to $I$.
This differs from averaging the policies' action probabilities.

```python
import jax
import jax.numpy as jnp
import nashbench

game = nashbench.make("leduc_poker")


def make_policy(key):
    params = jax.random.normal(key, (*game.observation_shape, game.num_actions))

    def policy(observation, legal_action_mask):
        logits = observation @ params
        return jax.nn.softmax(jnp.where(legal_action_mask, logits, -jnp.inf))

    return policy


a, b, c = map(make_policy, jax.random.split(jax.random.key(0), 3))
seat_0 = nashbench.Mixture([a, b, c], weights=[1, 1, 2])  # 1/4, 1/4, 1/2.
seat_1 = nashbench.Mixture([a, b])  # Equal weights.
print(nashbench.exploitability(game, (seat_0, seat_1)))
print(nashbench.exploitability(game, seat_0))  # seat_0 plays both seats.
```

## Exploitability

`nashbench.exploitability(game, policy)` returns the exploitability of the
strategy profile $\sigma = (\sigma_0, \sigma_1)$ that `policy` defines:

$$
\frac{1}{2} \left( \max_{\pi} u_0(\pi, \sigma_1) + \max_{\pi} u_1(\sigma_0, \pi) \right),
$$

where $u_i(\sigma_0, \sigma_1)$ is the expected return of seat $i$ when seat 0
plays $\sigma_0$ and seat 1 plays $\sigma_1$. Exploitability is zero if and
only if the profile is a Nash equilibrium. It equals NashConv / 2, the value
of OpenSpiel's `exploitability.exploitability`.

`nashbench.evaluate(game, policy)` returns each seat's terms:
`best_response_values`, `profile_values`, which sum to 0, and `gains`, their
differences. A best-response value isn't a gain: Kuhn poker favors seat 1,
so at an equilibrium, both gains are 0, but the best-response values are
$-1/18$ and $1/18$.

### How it's computed

The computation is exact. It uses the game's *sequence form*
(`Game.sequence_form`), which lists each seat's information sets and the
payoff of every terminal history, independent of the policy. Each call then:

1.  Evaluates each policy on the information sets of the seats it plays, in
    batches.
1.  Computes each seat's *realization plan*: the probability that the seat's
    own actions reach each of its sequences. A mixture's plan is the
    weighted sum of its policies' plans.
1.  Computes each seat's expected payoff per sequence against the other
    seat's plan.
1.  Finds each seat's best response, from the deepest information sets up.

The first call for a game builds the sequence form, so it takes longer; later
calls with the same game object reuse it.

A mixture repeats steps 1 and 2 for each policy, one at a time, so its time
grows with the number of policies but its memory doesn't. Policies with
weight 0 are skipped.

For per-game first-call and later-call times, see
[Benchmark results](benchmarks.md#exact-exploitability-evaluation).
