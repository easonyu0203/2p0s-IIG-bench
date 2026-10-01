"""Policies and helpers that validate.py and measure.py share."""

import copy
import dataclasses
import itertools
from pathlib import Path
import platform
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import nashbench

# Information sets per policy call, as in nashbench.
CHUNK_SIZE = 2**20


def network(game, hidden=(256, 256), seed=0):
    """Returns a dense ReLU network policy, initialized from `seed`.

    Weights are float32, normal with variance 1 / fan-in, without biases.
    Matmuls use the highest precision, so probabilities don't depend on the
    batch.
    """
    rng = np.random.default_rng(seed)
    sizes = [game.observation_shape[0], *hidden, game.num_actions]
    *hidden_weights, last = [
        jnp.asarray(rng.normal(size=(n, m)) / np.sqrt(n), jnp.float32)
        for n, m in itertools.pairwise(sizes)
    ]

    def policy(observation, legal_action_mask):
        x = observation
        for w in hidden_weights:
            x = jax.nn.relu(jnp.dot(x, w, precision="highest"))
        logits = jnp.dot(x, last, precision="highest")
        return jax.nn.softmax(jnp.where(legal_action_mask, logits, -jnp.inf))

    return policy


def policy(name, game, seed=0):
    """Returns a test policy.

    `name` is "uniform"; "random", which is stochastic; "deterministic",
    which plays the most likely action of "random"; "sparse", which plays its
    two most likely actions; or "network", from `network`.
    """
    if name == "uniform":
        return nashbench.uniform_random
    if name == "network":
        return network(game, seed=seed)
    rng = np.random.default_rng(seed)
    w = jnp.asarray(
        rng.normal(size=(game.observation_shape[0], game.num_actions)),
        jnp.float32,
    )

    def sine_policy(observation, legal_action_mask):
        logits = 3 * jnp.sin(jnp.dot(observation, w, precision="highest"))
        probs = jax.nn.softmax(jnp.where(legal_action_mask, logits, -jnp.inf))
        if name == "deterministic":
            return jax.nn.one_hot(jnp.argmax(probs), probs.size)
        if name == "sparse":
            probs = jnp.where(probs >= jnp.sort(probs)[-2], probs, 0.0)
        return probs / probs.sum()

    return sine_policy


PROFILES = (
    "uniform",
    "random",
    "deterministic",
    "sparse",
    "network",
    "population",
)


def profile(name, game):
    """Returns the members of a test profile and their weight in each seat.

    "population" mixes different policies in each seat; any other name is
    one policy that plays both seats.
    """
    if name != "population":
        return [(policy(name, game), (1.0, 1.0))]
    return [
        (policy("random", game, 1), (0.75, 0.0)),
        (policy("deterministic", game, 1), (0.25, 0.0)),
        (policy("sparse", game, 2), (0.0, 0.5)),
        (policy("network", game, 1), (0.0, 0.5)),
    ]


def table(game, policy):
    """Returns a policy's probabilities at every information set.

    Probabilities are rounded to multiples of 2**-24 whose rows sum to
    exactly 1, so evaluators that renormalize play the same probabilities.
    """
    form = game.sequence_form
    n = form.parent.size
    chunks = []
    for start in range(0, n, CHUNK_SIZE):
        ids = jnp.arange(start, min(start + CHUNK_SIZE, n))
        probs = jax.vmap(policy)(form.observe(ids), form.legal_action_mask[ids])
        chunks.append(np.asarray(probs, np.float64))
    units = np.round(np.concatenate(chunks) * 2**24)
    units[np.arange(n), units.argmax(1)] += 2**24 - units.sum(1)
    return (units / 2**24).astype(np.float32)


def evaluate_tables(game, members):
    """Evaluates tables with nashbench.

    Args:
        game: The game.
        members: `[(table, (seat 0 weight, seat 1 weight))]`.
    """
    # Policies observe information set ids, and look up their rows.
    lookup = copy.copy(game)
    lookup.sequence_form = dataclasses.replace(
        game.sequence_form, observe=lambda ids: ids
    )
    policies = [_lookup(jnp.asarray(t)) for t, _ in members]
    weights = np.array([w for _, w in members])
    return nashbench.evaluate(
        lookup,
        (
            nashbench.Mixture(policies, weights[:, 0]),
            nashbench.Mixture(policies, weights[:, 1]),
        ),
    )


def _lookup(rows):
    def policy(observation, legal_action_mask):
        del legal_action_mask  # Unused.
        return rows[observation]

    return policy


def environment():
    """Returns the code revision, versions and hardware, for records."""
    root = Path(__file__).parent

    def git(*args):
        return subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "jax": jax.__version__,
        "device": jax.devices()[0].device_kind,
        "host": platform.node(),
    }
