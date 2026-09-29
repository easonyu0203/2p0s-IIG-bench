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

## Exploitability

`nashbench.exploitability(game, policy)` returns the exploitability of
`policy` when it plays both seats of `game`:

$$
\frac{1}{2} \left( \max_{\pi} u_0(\pi, \sigma) + \max_{\pi} u_1(\sigma, \pi) \right),
$$

where $\sigma$ is the policy and $u_i(\sigma_0, \sigma_1)$ is the expected
return of seat $i$ when seat 0 plays $\sigma_0$ and seat 1 plays
$\sigma_1$. Exploitability is zero if and only if the policy is a Nash
equilibrium. It equals NashConv / 2, the value of OpenSpiel's
`exploitability.exploitability`.

### How it's computed

The computation is exact. It uses the game's *sequence form*
(`Game.sequence_form`), which lists each seat's information sets and the
payoff of every terminal history, independent of the policy. Each call then:

1.  Evaluates the policy on every information set, in batches.
1.  Computes each seat's *realization plan*: the probability that the seat's
    own actions reach each of its sequences.
1.  Computes each seat's expected payoff per sequence against the other
    seat's plan.
1.  Finds each seat's best response, from the deepest information sets up.

The first call for a game builds the sequence form; later calls with the same
game object reuse it.

-   **Poker games** have small trees. The build enumerates the tree once,
    and later calls take milliseconds.
-   **Phantom games** have about 10^10 terminal histories, too many to store.
    Each call traverses the whole tree on the accelerator.
-   **Goofspiel**'s tree is regular: a terminal history is an order of the
    point cards and each seat's order of bids. Each call sums over all of
    them in one vectorized pass, without a traversal.
-   **Liar's Dice**'s tree is regular too: an information set is a seat's
    dice and the set of bids so far. Each call sums over all terminal
    histories with one small matrix product per bid.
-   **Oshi-Zumo**'s bids are public, so each history is an information set
    of both seats. The build enumerates the histories once, and each call
    sums over the stored terminal histories.

For per-game first-call and later-call times, see
[Benchmark results](benchmarks.md#exact-exploitability-evaluation).
