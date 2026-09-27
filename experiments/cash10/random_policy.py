"""Evaluate the fixed ten-account Random controller on shared instances.

Random chooses uniformly among the semantic routes exposed by an interface and
STOP, then independently samples a transfer ratio uniformly on [0, 1]. It uses
no learned policy or unrevealed future. Full mode reproduces the appendix's
paired 200-instance, ten-action-replicate protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .runtime import make_env


def candidate_routes(base, interface):
    if base.stage == 0:
        if interface == "flat":
            return list(dict.fromkeys(list(base.stage1_edges) + list(base.stage2_edges)))
        persistent = set(base.stage2_edges)
        return [edge for edge in base.stage1_edges if edge not in persistent]
    return list(base.stage2_edges)


def evaluate_episode(interface, environment_seed, replicate_index, entropy):
    rng = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([entropy, environment_seed, replicate_index])
    ))
    env = make_env(interface, environment_seed)
    try:
        env.reset()  # Constructor reset is ordinal one; this is ordinal two.
        base = env.base_env
        total = 0.0
        for length in range(1, 35):
            routes = candidate_routes(base, interface)
            current = base.get_current_edges()
            candidate_index = int(rng.integers(0, len(routes) + 1))
            ratio = float(rng.uniform(0.0, 1.0))
            if candidate_index == len(routes):
                action = (len(current), ratio)
            else:
                route = routes[candidate_index]
                if route not in current:
                    raise RuntimeError("A semantic route is not executable")
                action = (current.index(route), ratio)
            _, reward, done, info = base.step(action)
            total += float(reward)
            if done:
                break
        else:
            raise RuntimeError("Random episode exceeded the documented horizon")
        return {
            "interface": interface,
            "environment_seed": environment_seed,
            "replicate_index": replicate_index,
            "action_rng_entropy": entropy,
            "reset_ordinal": 2,
            "episode_length": length,
            "episode_return": total,
        }
    finally:
        env.close()


def paired_interval(differences, *, resamples=100_000, seed=2026092402):
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(resamples, dtype=np.float64)
    for first in range(0, resamples, 1000):
        last = min(first + 1000, resamples)
        indices = rng.integers(0, len(differences), size=(last - first, len(differences)))
        samples[first:last] = differences[indices].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975], method="linear")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--smoke", action="store_true", help="One development instance and replicate, not a paper result")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    args.output.mkdir(parents=True, exist_ok=False)
    first, count, replicates = (880000, 1, 1) if args.smoke else (994000, 200, 10)
    entropy = 2026092401
    rows = []
    for environment_seed in range(first, first + count):
        for replicate_index in range(replicates):
            for interface in ("flat", "mmdp"):
                rows.append(evaluate_episode(interface, environment_seed, replicate_index, entropy))
    with (args.output / "episodes.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_key = {(r["environment_seed"], r["replicate_index"], r["interface"]): r["episode_return"] for r in rows}
    instance_means = []
    for environment_seed in range(first, first + count):
        flat = float(np.mean([by_key[environment_seed, replicate_index, "flat"] for replicate_index in range(replicates)]))
        mmdp = float(np.mean([by_key[environment_seed, replicate_index, "mmdp"] for replicate_index in range(replicates)]))
        instance_means.append((flat, mmdp))
    differences = np.asarray([mmdp - flat for flat, mmdp in instance_means], dtype=np.float64)
    summary = {
        "scope": "smoke_only" if args.smoke else "appendix_random_interface_reference",
        "instances": count,
        "replicates_per_instance_interface": replicates,
        "episodes": len(rows),
        "flat_mean": float(np.mean([flat for flat, _ in instance_means])),
        "mmdp_mean": float(np.mean([mmdp for _, mmdp in instance_means])),
        "mmdp_minus_flat": float(np.mean(differences)),
        "paired_bootstrap_95_ci": None if args.smoke else paired_interval(differences),
        "bootstrap_unit": "instance after averaging action replicates",
        "positive_zero_negative_instances": [int((differences > 1e-9).sum()), int((abs(differences) <= 1e-9).sum()), int((differences < -1e-9).sum())],
    }
    with (args.output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
