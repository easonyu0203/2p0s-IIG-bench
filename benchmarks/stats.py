"""Measures the statistics that the docs list for each game.

For each game, prints its numbers for the tables in README.md and
docs/games/README.md: its information sets, the time of an exploitability
call, and how many million steps per second it runs. The docs report
numbers from one NVIDIA RTX A6000; on CPU, exploitability of phantom games
takes more than 40 minutes. For example, with CUDA 13:

    uv run --with "jax[cuda13]" python benchmarks/stats.py [GAME ...]

Without arguments, it measures every game.
"""

import sys
import time

import jax
import jax.numpy as jnp

import nashbench

BATCH_SIZES = (1, 1024, 16384, 262144)


def random_actions(key, legal_action_mask):
    """Returns a uniformly random legal action for each row of the mask."""
    # Unrolled, Gumbel-max sampling fuses into one kernel; with two actions,
    # jax.random.categorical is slower than a step of Kuhn poker.
    noise = jax.random.gumbel(key, legal_action_mask.shape)
    score = jnp.where(legal_action_mask, noise, -jnp.inf)
    best, action = score[..., 0], jnp.zeros(score.shape[:-1], jnp.int32)
    for a in range(1, score.shape[-1]):
        action = jnp.where(score[..., a] > best, a, action)
        best = jnp.maximum(best, score[..., a])
    return action


def steps_per_second(game, batch_size):
    """Returns how many steps per second a batch of games runs."""
    num_steps = max(100, min(2000, 2**22 // batch_size))
    step = jax.vmap(nashbench.auto_reset(game))

    @jax.jit
    def run(key):
        keys = jax.random.split(key, batch_size)
        state, timestep = jax.vmap(game.reset)(keys)

        def body(carry, key):
            state, timestep = carry
            action = random_actions(key, timestep.legal_action_mask)
            return step(state, action), None

        keys = jax.random.split(key, num_steps)
        (_, timestep), _ = jax.lax.scan(body, (state, timestep), keys)
        # Using every field keeps XLA from skipping any, such as observations.
        return sum(x.sum() for x in jax.tree.leaves(timestep))

    run(jax.random.key(0)).block_until_ready()  # Compiles.
    seconds = min(_seconds(run, jax.random.key(i)) for i in range(1, 6))
    return batch_size * num_steps / seconds


def exploitability_seconds(game):
    """Returns the time of the first exploitability call and of a later one."""
    policy = nashbench.uniform_random
    times = [_seconds(nashbench.exploitability, game, policy) for _ in range(3)]
    # Calls of milliseconds vary with host dispatch, so repeat them for 1 s.
    while sum(times[1:]) < 1:
        times.append(_seconds(nashbench.exploitability, game, policy))
    return times[0], min(times[1:])


def _seconds(fn, *args):
    start = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - start


def _round(x):
    """Formats `x` with two significant digits."""
    return f"{float(f'{x:.2g}'):g}"


def _duration(seconds):
    if seconds < 0.1:
        return f"{_round(seconds * 1e3)} ms"
    return f"{_round(seconds)} s"


def main(names):
    """Prints the statistics of the games named `names`."""
    print(f"Device: {jax.devices()[0].device_kind}\n")
    for name in names:
        game = nashbench.make(name)
        rates = [steps_per_second(game, b) for b in BATCH_SIZES]
        first, later = exploitability_seconds(game)
        speed = " | ".join(_round(r / 1e6) for r in rates[1:])
        print(name)
        print(f"  Information sets: {game.sequence_form.parent.size:,}")
        print(
            f"  Exploitability: {_duration(later)} per call, "
            f"{_duration(first)} for the first"
        )
        print(f"  One game: {_round(1e6 / rates[0])} microseconds per step")
        print(f"  Speed table row: | `{name}` | {speed} |\n")


if __name__ == "__main__":
    main(sys.argv[1:] or list(nashbench.REGISTRY))
