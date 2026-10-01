"""Compares nashbench's values with independent evaluators.

    python benchmarks/validate.py [--only REGEX] [--out FILE]

For each configuration that matches REGEX and each test profile, nashbench
and the reference evaluate the same table of probabilities. Each comparison
appends a JSON line to FILE, which later runs skip. The script then prints
the largest error of each configuration. See README.md.
"""

import argparse
import json
from pathlib import Path
import re

import common
import numpy as np
import references

import nashbench

# OpenSpiel can't fit the default Goofspiel, Liar's Dice and Oshi-Zumo in
# memory, so it evaluates reduced sizes.
CONFIGS = [
    ("openspiel", "kuhn_poker", {}),
    ("openspiel", "leduc_poker", {}),
    *(("openspiel", "goofspiel", {"num_cards": k}) for k in (3, 4, 5)),
    *(
        ("openspiel", "liars_dice", {"numdice": n, "dice_sides": s})
        for n, s in [(1, 6), (2, 2), (2, 3)]
    ),
    *(
        ("openspiel", "oshi_zumo", {"coins": c, "size": s})
        for c, s in [(6, 1), (8, 3), (10, 3)]
    ),
    *(("exp-a-spiel", name, {}) for name in references.TRAVERSERS),
]
REFERENCES = {
    "openspiel": references.OpenSpiel,
    "exp-a-spiel": references.ExpASpiel,
}
# A value passes if its error is at most TOLERANCE times the game's largest
# absolute return: MAX_RETURN, or 1.
TOLERANCE = 1e-9
MAX_RETURN = {"kuhn_poker": 2, "leduc_poker": 13}


def main():
    """Runs the comparisons that the output file doesn't have."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", default="", help="configurations to run")
    parser.add_argument("--out", type=Path, default=Path("validation.jsonl"))
    args = parser.parse_args()
    done = {(r["config"], r["profile"]) for r in _read(args.out)}
    environment = common.environment()
    for reference_name, name, params in CONFIGS:
        config = f"{reference_name}/{name}"
        if params:
            config += f"({','.join(f'{k}={v}' for k, v in params.items())})"
        todo = [p for p in common.PROFILES if (config, p) not in done]
        if not (todo and re.search(args.only, config)):
            continue
        game = nashbench.make(name, **params)
        reference = REFERENCES[reference_name](name, game)
        rows = references.match(game, reference)
        for profile in todo:
            members = [
                (common.table(game, policy), weights)
                for policy, weights in common.profile(profile, game)
            ]
            evaluation = common.evaluate_tables(game, members)
            ours = evaluation.best_response_values, evaluation.profile_values
            theirs = reference.evaluate(
                [([t[r] for r in rows], w) for t, w in members]
            )
            record = {
                "config": config,
                "profile": profile,
                "nashbench": ours,
                "reference": theirs,
                "error": float(np.abs(np.subtract(ours, theirs)).max()),
                "tolerance": TOLERANCE * MAX_RETURN.get(name, 1),
                **environment,
            }
            with args.out.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"{config} {profile}: error {record['error']:.1e}")
    _summary(_read(args.out))


def _read(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _summary(records):
    """Prints the largest error and passes of each configuration."""
    print("| Configuration | Largest error | Tolerance | Passed |")
    print("| --- | ---: | ---: | ---: |")
    for config in dict.fromkeys(r["config"] for r in records):
        rs = [r for r in records if r["config"] == config]
        passed = sum(r["error"] <= r["tolerance"] for r in rs)
        print(
            f"| `{config}` | {max(r['error'] for r in rs):.1e} "
            f"| {rs[0]['tolerance']:.0e} | {passed}/{len(rs)} |"
        )


if __name__ == "__main__":
    main()
