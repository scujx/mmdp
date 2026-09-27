"""Analyze Cash5 evaluation CSV/JSONL files with the paper's fixed estimands."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.cash5"

import argparse
import csv
import json
import numpy as np
from . import statistics as ppo


def records(paths):
    for path in paths:
        with path.open() as handle:
            if path.suffix == ".csv":
                yield from csv.DictReader(handle)
            elif path.suffix == ".jsonl":
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            else:
                raise ValueError(f"Expected CSV or JSONL: {path.name}")


def calculate(controller, paths):
    if controller == "ppo":
        data = np.full((10, 12, 200), np.nan)
        for row in records(paths):
            setting = row["setting_id"]
            i, j, k = (
                int(row["training_seed"]) - 42,
                ppo.SETTINGS.index(setting),
                int(row["environment_seed"]) - 980000,
            )
            if not (0 <= i < 10 and 0 <= k < 200):
                raise ValueError("Unexpected training or environment seed")
            if row["mask_type"] != ppo.MASK_TYPES[setting]:
                raise ValueError("Mask family mismatch")
            if not np.isnan(data[i, j, k]):
                raise ValueError("Duplicate PPO episode")
            value = float(row["full_return"])
            if not np.isfinite(value):
                raise ValueError("Nonfinite PPO return")
            data[i, j, k] = value
        if not np.isfinite(data).all():
            raise ValueError("Need 12 settings x 10 training seeds x 200 instances")
        return ppo.analyze_cube(data)
    data = np.full((7, 200, 10), np.nan)
    for row in records(paths):
        i, j, k = (
            ppo.RANDOM_SETTINGS.index(row["setting_id"]),
            int(row["environment_seed"]) - 980000,
            int(row["policy_rng_seed"]),
        )
        if not (0 <= j < 200 and 0 <= k < 10):
            raise ValueError("Unexpected environment or policy-RNG seed")
        if not np.isnan(data[i, j, k]):
            raise ValueError("Duplicate Random episode")
        value = float(row["full_return"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite Random return")
        data[i, j, k] = value
    if not np.isfinite(data).all():
        raise ValueError(
            "Need seven settings x 200 instances x ten action-RNG replicates"
        )
    return ppo.analyze_random(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["ppo", "random"], required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = calculate(args.controller, args.input)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print("Completed paired analysis; no saved reference results were used.")
