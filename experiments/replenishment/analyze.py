"""Analyze a complete forty-policy rerun; dependencies: NumPy only.

The statistical unit is a training seed. Complete four-cell rows are resampled
together, conditional on the shared instance bank. No plotting is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "configs/experiment.json").read_text())
CELLS = CONFIG["cells"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def contrasts(x):
    return {
        "mmdp_minus_flat_updated": x[:, 3] - x[:, 1],
        "mmdp_minus_flat_retained": x[:, 2] - x[:, 0],
        "updated_minus_retained_flat": x[:, 1] - x[:, 0],
        "updated_minus_retained_mmdp": x[:, 3] - x[:, 2],
        "interaction_secondary": (x[:, 3] - x[:, 1]) - (x[:, 2] - x[:, 0]),
    }


def summarize(test, auc):
    expected = (len(CONFIG["seeds"]), len(CELLS))
    if test.shape != expected or auc.shape != expected:
        raise ValueError(f"Expected two complete arrays of shape {expected}.")
    if not np.isfinite(test).all() or not np.isfinite(auc).all():
        raise ValueError("Nonfinite metrics.")
    idx = np.random.Generator(np.random.PCG64(CONFIG["bootstrap_seed"])).integers(
        0, expected[0], (CONFIG["bootstrap_draws"], expected[0])
    )
    statistics = []
    for metric, values in [("test_return", test), ("validation_auc", auc)]:
        estimates = {c: values[:, ci] for ci, c in enumerate(CELLS)}
        estimates.update(contrasts(values))
        for name, v in estimates.items():
            lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
            statistics.append(
                dict(
                    metric=metric,
                    estimand=name,
                    mean=float(v.mean()),
                    seed_sd=float(v.std(ddof=1)),
                    ci_low=float(lo),
                    ci_high=float(hi),
                )
            )
    return statistics


def load_metrics(runs, evaluations):
    test = np.zeros((len(CONFIG["seeds"]), len(CELLS)))
    curves = np.zeros((*test.shape, len(CONFIG["validation_steps"])))
    fingerprints = {}
    source_hashes = None
    initial_hashes = {}
    expected_names = {f"{c}_{s}" for c in CELLS for s in CONFIG["seeds"]}
    actual_names = {p.stem for p in evaluations.glob("*.jsonl")}
    if actual_names != expected_names:
        raise ValueError(
            "Evaluation directory must contain exactly the forty formal policy files."
        )
    for ci, cell in enumerate(CELLS):
        for si, seed in enumerate(CONFIG["seeds"]):
            run = runs / f"{cell}_{seed}"
            meta = json.loads((run / "run.json").read_text())
            if (
                meta["smoke"]
                or meta["status"] != "complete"
                or meta["cell"] != cell
                or meta["seed"] != seed
                or meta["actual_steps"] != CONFIG["actual_steps"]
                or meta["parameter_count"] != CONFIG["parameter_count"]
                or meta["validation_ids"] != CONFIG["validation_ids"]
                or meta["validation_split"] != "validation"
            ):
                raise ValueError(f"Protocol mismatch: {cell}, {seed}")
            if source_hashes is None:
                source_hashes = meta["source_hashes"]
            if source_hashes != meta["source_hashes"]:
                raise ValueError("Mixed source/configuration versions.")
            if (
                seed in initial_hashes
                and initial_hashes[seed] != meta["initial_policy_hash"]
            ):
                raise ValueError(
                    "Initial weights differ within a paired training seed."
                )
            initial_hashes[seed] = meta["initial_policy_hash"]
            validation_path = run / "validation.jsonl"
            if digest(validation_path) != meta["validation_sha256"]:
                raise ValueError("Changed validation records.")
            checkpoint_hash = meta["selection"]["sha256"]
            if digest(run / "selected.zip") != checkpoint_hash:
                raise ValueError("Changed selected checkpoint.")
            validation = read_rows(validation_path)
            episodes = read_rows(evaluations / f"{cell}_{seed}.jsonl")
            validate_records(
                validation,
                "validation",
                CONFIG["validation_ids"],
                fingerprints,
                steps=CONFIG["validation_steps"],
            )
            validate_records(episodes, "test", CONFIG["test_ids"], fingerprints)
            for row in episodes:
                if (
                    row["cell"] != cell
                    or row["training_seed"] != seed
                    or row["checkpoint_sha256"] != checkpoint_hash
                    or row["smoke"]
                ):
                    raise ValueError(
                        "Test rows do not belong to the selected formal policy."
                    )
            for ti, step in enumerate(CONFIG["validation_steps"]):
                curves[si, ci, ti] = np.mean(
                    [r["return"] for r in validation if r["step"] == step]
                )
            best = int(np.argmax(curves[si, ci]))  # Earliest checkpoint breaks ties.
            if meta["selection"]["step"] != CONFIG["validation_steps"][
                best
            ] or not np.isclose(
                meta["selection"]["mean"], curves[si, ci, best], rtol=0, atol=1e-8
            ):
                raise ValueError(
                    "Selection does not maximize the recorded validation means."
                )
            test[si, ci] = np.mean([r["return"] for r in episodes])
    steps = np.asarray(CONFIG["validation_steps"])
    auc = np.trapezoid(curves, steps, axis=2) / (steps[-1] - steps[0])
    return test, auc


def validate_records(rows, split, ids, fingerprints, steps=None):
    expected = (
        {(s, i) for s in steps for i in ids} if steps else {(None, i) for i in ids}
    )
    keys = [(r.get("step"), r["instance_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(f"Missing, duplicate, or unexpected {split} records.")
    for row in rows:
        if row["split"] != split or row["ordinal"] != 0 or row["length"] != 26:
            raise ValueError("Wrong instance split, ordinal, or episode length.")
        value = float(row["return"])
        reconstructed = row["revenue"] - sum(
            row[k]
            for k in ["regular_cost", "flexible_cost", "holding_cost", "shortage_cost"]
        )
        if not np.isfinite(value) or not np.isclose(
            value, reconstructed, rtol=0, atol=1e-8
        ):
            raise ValueError("Invalid return accounting.")
        key = (split, row["instance_id"])
        if key in fingerprints and fingerprints[key] != row["fingerprint"]:
            raise ValueError(
                "Instance fingerprints differ across policies/checkpoints."
            )
        fingerprints[key] = row["fingerprint"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    package = ROOT.parents[1]
    if output.exists() or output == package or package in output.parents:
        parser.error("Use a new output directory outside the source package.")
    test, auc = load_metrics(args.runs, args.evaluations)
    statistics = summarize(test, auc)
    seed_metrics = [
        dict(
            cell=c,
            training_seed=s,
            test_return=float(test[si, ci]),
            validation_auc=float(auc[si, ci]),
        )
        for si, s in enumerate(CONFIG["seeds"])
        for ci, c in enumerate(CELLS)
    ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x") as handle:
        json.dump(
            {
                "seed_metrics": seed_metrics,
                "statistics": statistics,
                "inference": "Paired training seeds; marginal percentile 95% intervals conditional on fixed banks; no p-values or selection correction; AUC 5K--100K / 95K.",
            },
            handle,
            indent=2,
        )
    print(f"Complete matrix verified; summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
