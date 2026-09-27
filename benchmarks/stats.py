"""Measures the statistics that the docs list for each game.

For each game, prints its numbers for the tables in docs/benchmarks.md: its
information sets, the time of the first and of a later exploitability call,
and how many million steps per second it runs.
Times and speeds are the mean ± standard deviation of 10 runs. The docs
report numbers from one NVIDIA RTX A6000:

    uv run --with benchmarks/stats.py [GAME ...]

Without arguments, it measures every game.
"""

import statistics
import sys
import time

import jax
import jax.numpy as jnp

import nashbench

BATCH_SIZES = (1, 64, 1024, 16384)
NUM_RUNS = 10


def steps_per_second(game, batch_size):
    """Returns how many steps per second a batch of games runs, per run."""
    num_steps = max(100, min(2000, 2**22 // batch_size))
    step = jax.vmap(nashbench.auto_reset(game))

    @jax.jit
    def run(key):
        keys = jax.random.split(key, batch_size)
        state, timestep = jax.vmap(game.reset)(keys)

        def body(carry, key):
            state, timestep = carry
            mask = timestep.legal_action_mask
            action = jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))
            return step(state, action), None

        keys = jax.random.split(key, num_steps)
        (_, timestep), _ = jax.lax.scan(body, (state, timestep), keys)
        # Using every field keeps XLA from skipping any, such as observations.
        return sum(x.sum() for x in jax.tree.leaves(timestep))

    run(jax.random.key(0)).block_until_ready()  # Compiles.
    return [
        batch_size * num_steps / _seconds(run, jax.random.key(seed))
        for seed in range(1, NUM_RUNS + 1)
    ]


def first_call_seconds(name):
    """Returns the time of the first exploitability call for a game."""
    # A new game and empty caches make the call build the sequence form and
    # compile, as in a new process.
    jax.clear_caches()
    game = nashbench.make(name)
    return _seconds(nashbench.exploitability, game, nashbench.uniform_random)


def later_call_seconds(game):
    """Returns the times of exploitability calls after the first."""
    nashbench.exploitability(game, nashbench.uniform_random)  # Builds.
    return [
        _seconds(nashbench.exploitability, game, nashbench.uniform_random)
        for _ in range(NUM_RUNS)
    ]


def _seconds(fn, *args):
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - start


def _round(x):
    """Formats `x` with two significant digits."""
    return f"{float(f'{x:.2g}'):g}"


def _mean_std(xs):
    return f"{_round(statistics.mean(xs))} ± {_round(statistics.stdev(xs))}"


def _duration(seconds):
    if statistics.mean(seconds) < 0.1:
        return f"{_mean_std([s * 1e3 for s in seconds])} ms"
    return f"{_mean_std(seconds)} s"


def main(names):
    """Prints the statistics of the games named `names`."""
    print(f"Device: {jax.devices()[0].device_kind}\n")
    for name in names:
        game = nashbench.make(name)
        rates = [steps_per_second(game, b) for b in BATCH_SIZES]
        first = [first_call_seconds(name) for _ in range(NUM_RUNS)]
        later = later_call_seconds(game)
        speed = " | ".join(
            _mean_std([r / 1e6 for r in rate]) for rate in rates[1:]
        )
        per_step = _mean_std([1e6 / r for r in rates[0]])
        print(name)
        print(f"  Information sets: {game.sequence_form.parent.size:,}")
        print(f"  Exploitability, first call: {_duration(first)}")
        print(f"  Exploitability, later call: {_duration(later)}")
        print(f"  One game: {per_step} microseconds per step")
        print(f"  Speed table row: | `{name}` | {speed} |\n")


if __name__ == "__main__":
    main(sys.argv[1:] or list(nashbench.REGISTRY))
