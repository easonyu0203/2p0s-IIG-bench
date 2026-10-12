"""Measures the peak memory of one exact evaluation.

    python benchmarks/memory.py [--only REGEX] [--out FILE]

Each measurement runs in a new process, which evaluates once and records its
peak resident host memory and, for nashbench, its peak GPU memory.
nashbench evaluates a network of two layers of 256 units, as in measure.py.
exp-a-spiel evaluates the uniform strategy of the phantom games, since its
memory doesn't depend on the probabilities. Records append to FILE, which
later runs skip. See README.md.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

import common
import jax
import measure
import references

import nashbench

MEASUREMENTS = [
    *({"evaluator": "nashbench", "game": g} for g in nashbench.REGISTRY),
    *({"evaluator": "exp_a_spiel", "game": g} for g in references.TRAVERSERS),
]


def run(args):
    """Runs the measurements that the output file doesn't have."""
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.open()}
    for spec in MEASUREMENTS:
        name = f"{spec['evaluator']}/{spec['game']}"
        if name in done or not re.search(args.only, name):
            continue
        print(f"{time.strftime('%H:%M:%S')} {name}", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, "worker", json.dumps(spec)],
            capture_output=True,
            text=True,
            check=False,
            env=os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        )
        found = measure.last_object(result.stdout)
        if result.returncode or found is None:
            print(result.stderr[-2000:])
            continue
        record = {"id": name, **spec, **found}
        with args.out.open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def worker(spec):
    """Evaluates once and prints the peak memory as a JSON line."""
    gpu = None
    if spec["evaluator"] == "nashbench":
        game = nashbench.make(spec["game"])
        nashbench.evaluate(game, common.network(game))
        stats = jax.devices()[0].memory_stats() or {}
        gpu = stats.get("peak_bytes_in_use", 0) / 2**30
    else:
        eas = importlib.import_module("eas")
        traverser = getattr(eas, references.TRAVERSERS[spec["game"]])()
        traverser.ev_and_exploitability(
            *traverser.construct_uniform_strategies()
        )
    # Linux reports the peak resident set in KiB.
    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    record = {"host_gib": host, "gpu_gib": gpu}
    print(json.dumps(record | common.environment()), flush=True)


def main():
    """Parses the command line."""
    if sys.argv[1:2] == ["worker"]:
        return worker(json.loads(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="measurements to run")
    parser.add_argument("--out", type=Path, default=Path("memory.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
