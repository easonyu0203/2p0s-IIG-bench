"""Measures exact evaluation and simulation on one GPU.

    python benchmarks/measure.py run [--only REGEX] [--out FILE] [--repeats N]
    python benchmarks/measure.py report [--out FILE]

`run` runs each measurement that matches REGEX in a new process and appends
a JSON line to FILE, which later runs skip. `report` prints the tables of
docs/benchmarks.md. See README.md.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import common
import jax
import jax.numpy as jnp
import numpy as np
import references

import nashbench

GAMES = list(nashbench.REGISTRY)
NETWORKS = {"2x256": (256, 256), "3x512": (512, 512, 512)}
BATCH_SIZES = (64, 1024, 16384)
#: Parallel episodes of the scaling measurement, and its games (the core suite).
SCALING = tuple(4**k for k in range(9))
SCALING_GAMES = (
    "leduc_poker",
    "universal_poker",
    "liars_dice",
    "goofspiel",
    "oshi_zumo",
    "blotto",
    "dark_hex3_abrupt",
)
MEASUREMENTS = [
    *(
        {"workload": "network", "game": game, "network": network}
        for game in GAMES
        for network in NETWORKS
    ),
    *({"workload": "table", "game": game} for game in GAMES),
    *(
        {"workload": "exp_a_spiel", "game": game}
        for game in references.TRAVERSERS
    ),
    *(
        {"workload": "population", "game": game, "k": k}
        for game in ("leduc_poker", "dark_hex3_abrupt")
        for k in (1, 4, 16, 64)
    ),
    *(
        {"workload": "simulation", "game": game, "network": network}
        for game in GAMES
        for network in ("uniform", "2x256")
    ),
    *(
        {"workload": "scaling", "game": game, "network": network}
        for game in SCALING_GAMES
        for network in ("uniform", "2x256")
    ),
    *({"workload": "openspiel", "game": game} for game in SCALING_GAMES),
]


def run(args):
    """Runs the measurements that the output file doesn't have."""
    done = {r["id"] for r in _read(args.out) if "error" not in r}
    for spec in MEASUREMENTS:
        name = "/".join(map(str, spec.values()))
        if name in done or not re.search(args.only, name):
            continue
        print(f"{time.strftime('%H:%M:%S')} {name}", flush=True)
        busy = _gpu_processes()
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "worker",
                json.dumps(spec | {"repeats": args.repeats}),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        )
        record = last_object(result.stdout) if result.returncode == 0 else None
        record = record or {"error": (result.stderr or result.stdout)[-2000:]}
        record = {
            "id": name,
            **spec,
            **record,
            # Other processes on the GPU, before and after.
            "gpu_processes": sorted(set(busy) | set(_gpu_processes())),
        }
        with args.out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if "error" in record:
            print(record["error"])


def last_object(text):
    """The worker's record: the last top-level JSON object in its stdout.

    exp-a-spiel's C++ code also logs to stdout, without flushing, so its
    lines can precede or follow the record, even on the same line.
    """
    decoder, found = json.JSONDecoder(), None
    start = text.find('{"')
    while start >= 0:
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find('{"', start + 1)
            continue
        if isinstance(value, dict):
            found = value
        start = text.find('{"', end)
    return found


def _gpu_processes():
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return []


def worker(spec):
    """Runs one measurement and prints its record as a JSON line."""
    game = nashbench.make(spec["game"], **spec.get("game_params", {}))
    record = {}
    if spec["workload"] not in ("simulation", "scaling", "openspiel"):
        _, record["build"] = _timed(lambda: game.sequence_form)
        record["infosets"] = game.sequence_form.parent.size
    workloads = {
        "network": _network,
        "table": _table,
        "exp_a_spiel": _exp_a_spiel,
        "population": _population,
        "simulation": _simulation,
        "scaling": lambda spec, game: _simulation(spec, game, SCALING),
        "openspiel": _openspiel,
    }
    record |= workloads[spec["workload"]](spec, game)
    stats = jax.devices()[0].memory_stats() or {}
    record["peak_gib"] = stats.get("peak_bytes_in_use", 0) / 2**30
    print(json.dumps(record | common.environment()))


def _timed(fn):
    """Returns `fn()`, once done on the device, and its time in seconds."""
    start = time.perf_counter()
    result = jax.block_until_ready(fn())
    return result, time.perf_counter() - start


def _network(spec, game):
    """Evaluates networks of one architecture, each with new parameters."""
    policies = [
        common.network(
            game, NETWORKS[spec["network"]], spec.get("seed", 0) * 100 + seed
        )
        for seed in range(spec["repeats"] + 1)
    ]
    times = [
        _timed(lambda p=p: nashbench.evaluate(game, p))[1] for p in policies
    ]
    return {"first": times[0], "warm": times[1:]}


def _table(spec, game):
    """Evaluates a table of a network's probabilities, already on the GPU."""
    rows = jnp.asarray(
        common.table(game, common.network(game, seed=spec.get("seed", 0)))
    )
    times = [
        _timed(lambda: common.evaluate_tables(game, [(rows, (1, 1))]))[1]
        for _ in range(spec["repeats"] + 1)
    ]
    return {"first": times[0], "warm": times[1:]}


def _exp_a_spiel(spec, game):
    """Evaluates the same table with exp-a-spiel, then its complete path.

    The complete path computes exp-a-spiel's observations, runs the network
    on the GPU, copies the probabilities to the host, and evaluates them.
    """
    policy = common.network(game)
    table = common.table(game, policy)
    reference, build = _timed(lambda: references.ExpASpiel(spec["game"], game))
    traverser = reference.traverser
    rows = references.match(game, reference)
    tables = [table[r].astype(np.float64) for r in rows]
    # Each evaluation takes minutes.
    warm = [
        _timed(lambda: traverser.ev_and_exploitability(*tables))[1]
        for _ in range(min(3, spec["repeats"]))
    ]

    def infer(seat):
        x = traverser.compute_openspiel_infostates(seat)
        probs = []
        for i in range(0, len(x), common.CHUNK_SIZE):
            chunk = x[i : i + common.CHUNK_SIZE]
            observations = np.zeros((len(chunk), 2 + chunk.shape[1]))
            observations[:, seat] = 1
            observations[:, 2:] = chunk
            legal = reference.legal[seat][i : i + common.CHUNK_SIZE]
            probs.append(np.asarray(jax.vmap(policy)(observations, legal)))
        return np.concatenate(probs).astype(np.float64)

    for seat in range(2):
        infer(seat)  # Warms up the network's operations.
    _, complete = _timed(
        lambda: traverser.ev_and_exploitability(infer(0), infer(1))
    )
    return {
        "reference_build": build,
        "warm": warm,
        "complete": complete,
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _population(spec, game):
    """Evaluates a different mixture of K networks in each seat."""
    k = spec["k"]
    profile = (
        nashbench.Mixture(
            [common.network(game, seed=s) for s in range(k)],
            np.arange(1, k + 1),
        ),
        nashbench.Mixture([common.network(game, seed=k + s) for s in range(k)]),
    )
    # Large populations take minutes per evaluation.
    times = [
        _timed(lambda: nashbench.evaluate(game, profile))[1]
        for _ in range(min(3, spec["repeats"]) + 1)
    ]
    return {"first": times[0], "warm": times[1:]}


def _simulation(spec, game, batch_sizes=BATCH_SIZES):
    """Steps batches of games, choosing actions by a policy.

    Rates are steps per second, including resets and observations. Each step
    is one player decision.
    """
    if spec["network"] == "uniform":
        policy = nashbench.uniform_random
    else:
        policy = common.network(
            game, NETWORKS[spec["network"]], seed=spec.get("seed", 0)
        )
    rates = {}
    for batch in batch_sizes:
        num_steps = max(100, min(2000, 2**22 // batch))
        rollout = _rollout(game, policy, batch, num_steps)
        # Finish compilation and warmup before timing a later dispatch.
        jax.block_until_ready(
            rollout(jax.random.key(spec.get("seed", 0) * 100))
        )
        rates[batch] = [
            batch
            * num_steps
            / _timed(
                lambda s=s, f=rollout: f(
                    jax.random.key(spec.get("seed", 0) * 100 + s)
                )
            )[1]
            for s in range(1, spec["repeats"] + 1)
        ]
    return {"steps_per_second": rates}


def _openspiel(spec, game, seconds=10.0):
    """Plays OpenSpiel's implementation of the game, for reference.

    It plays from Python on one CPU thread with uniform random actions. Like the simulation, every decision computes the acting seat's tensor (its
    information state where the game provides one); a simultaneous move counts
    as two decisions.
    """
    import random

    import pyspiel
    import traverse

    os_game = pyspiel.load_game(traverse.GAMES[spec["game"]][1])
    info = os_game.get_type().provides_information_state_tensor
    rng = random.Random(0)
    decisions, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        state = os_game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes(), strict=True)
                state.apply_action(rng.choices(outcomes, probs)[0])
                continue
            seats = (
                range(2)
                if state.is_simultaneous_node()
                else [state.current_player()]
            )
            for seat in seats:
                if info:
                    state.information_state_tensor(seat)
                else:
                    state.observation_tensor(seat)
            if state.is_simultaneous_node():
                state.apply_actions(
                    [rng.choice(state.legal_actions(p)) for p in seats]
                )
            else:
                state.apply_action(rng.choice(state.legal_actions()))
            decisions += len(seats)
    return {"steps_per_second": decisions / (time.perf_counter() - start)}


def _rollout(game, policy, batch, num_steps):
    step = jax.vmap(nashbench.auto_reset(game))

    @jax.jit
    def rollout(key):
        state, timestep = jax.vmap(game.reset)(jax.random.split(key, batch))

        def body(carry, key):
            state, timestep = carry
            probs = jax.vmap(policy)(
                timestep.observation, timestep.legal_action_mask
            )
            action = jax.random.categorical(key, jnp.log(probs))
            return step(state, action), None

        keys = jax.random.split(key, num_steps)
        (_, timestep), _ = jax.lax.scan(body, (state, timestep), keys)
        # Using every field keeps XLA from skipping any, such as observations.
        return jax.tree.map(jnp.sum, timestep)

    return rollout


def report(args):
    """Prints tables of the records, with medians of warm runs in seconds."""
    records = {r["id"]: r for r in _read(args.out) if "error" not in r}

    def cell(name, key="warm"):
        value = records.get(name, {}).get(key)
        if isinstance(value, list):
            value = statistics.median(value)
        if isinstance(value, int):
            return f"{value:,}"
        return "" if value is None else f"{value:.3g}"

    _print_table(
        ["Game", "Information sets", "Build", "First", *NETWORKS, "Table"]
        + ["Peak GiB"],
        [
            [
                f"`{g}`",
                cell(f"table/{g}", "infosets"),
                cell(f"table/{g}", "build"),
                cell(f"network/{g}/2x256", "first"),
                *(cell(f"network/{g}/{n}") for n in NETWORKS),
                cell(f"table/{g}"),
                cell(f"network/{g}/2x256", "peak_gib"),
            ]
            for g in GAMES
        ],
    )
    _print_table(
        ["Game", "Build: nashbench", "Build: exp-a-spiel"]
        + ["Table: nashbench", "Table: exp-a-spiel"]
        + ["Complete: nashbench", "Complete: exp-a-spiel"],
        [
            [
                f"`{g}`",
                cell(f"table/{g}", "build"),
                cell(f"exp_a_spiel/{g}", "reference_build"),
                cell(f"table/{g}"),
                cell(f"exp_a_spiel/{g}"),
                cell(f"network/{g}/2x256"),
                cell(f"exp_a_spiel/{g}", "complete"),
            ]
            for g in references.TRAVERSERS
        ],
    )
    _print_table(
        ["Game", "K", "First", "Warm", "Peak GiB"],
        [
            [f"`{r['game']}`", r["k"], cell(name, "first"), cell(name)]
            + [cell(name, "peak_gib")]
            for name, r in records.items()
            if r["workload"] == "population"
        ],
    )
    _print_table(
        ["Game", "Policy", *(f"M steps/s, {b} games" for b in BATCH_SIZES)],
        [
            [f"`{r['game']}`", r["network"]]
            + [
                f"{statistics.median(rates) / 1e6:.3g}"
                for rates in r["steps_per_second"].values()
            ]
            for r in records.values()
            if r["workload"] == "simulation"
        ],
    )


def _print_table(columns, rows):
    print("\n| " + " | ".join(columns) + " |")
    print("| --- |" + " ---: |" * (len(columns) - 1))
    for row in rows:
        print("| " + " | ".join(map(str, row)) + " |")


def _read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    """Parses the command line."""
    if sys.argv[1] == "worker":
        return worker(json.loads(sys.argv[2]))
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["run", "report"])
    parser.add_argument("--only", default="", help="measurements to run")
    parser.add_argument("--out", type=Path, default=Path("measurements.jsonl"))
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
