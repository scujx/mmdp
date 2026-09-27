"""Fresh four-cell forecast-accuracy rerun on the fixed demand/coarse banks.

The published 80% setting is the default experiment. This command reruns one
accuracy at a time; it never loads historical weights or changes source files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .runtime import CELLS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-accuracy", type=float, choices=(0.75, 0.80, 0.85), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the package")
    parser.add_argument("--smoke", action="store_true", help="Four short development runs; not paper results")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating outputs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output == root or root in output.parents or output.exists():
        parser.error("Use a new output directory outside the source package")
    seeds = (42,) if args.smoke else tuple(range(42, 52))
    commands = []
    for cell in CELLS:
        for seed in seeds:
            run = output / "runs" / f"{cell}_{seed}"
            test = output / "evaluations" / f"{cell}_{seed}.jsonl"
            commands.append([sys.executable, "-B", "-m", "experiments.replenishment.run", "train",
                             "--cell", cell, "--seed", str(seed), "--fine-accuracy",
                             str(args.fine_accuracy), "--output", str(run)] + (["--smoke"] if args.smoke else []))
            commands.append([sys.executable, "-B", "-m", "experiments.replenishment.run", "evaluate",
                             "--run", str(run), "--trust-checkpoint", "--output", str(test)]
                            + (["--smoke"] if args.smoke else []))
    if not args.smoke:
        commands.append([sys.executable, "-B", "-m", "experiments.replenishment.analyze",
                         "--runs", str(output / "runs"), "--evaluations", str(output / "evaluations"),
                         "--output", str(output / "analysis")])
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluations").mkdir()
    (output / "protocol.json").write_text(json.dumps({
        "fine_accuracy": args.fine_accuracy, "smoke": args.smoke, "cells": CELLS,
        "training_seeds": seeds, "coarse_accuracy": 0.57875,
        "description": "Fresh training and evaluation; outputs do not contain published weights."
    }, indent=2) + "\n", encoding="utf-8")
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    print(f"Completed {len(seeds) * len(CELLS)} fresh policies in {output}")


if __name__ == "__main__":
    main()
