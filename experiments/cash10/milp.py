"""Evaluate the forecast-based rolling-horizon MILP on ten-account routes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.milp_planner import ForecastMeanMILPRollingHorizon
from .envs.cash_flow_env_v2 import CashFlowEnvConfigV2, CashFlowEnvV2


INTERFACES = ("Flat", "MMDP")
TEST_SEEDS = range(994000, 994200)


def mmdp_routes(env: CashFlowEnvV2) -> list[tuple[int, int]]:
    """Expose only routes expiring at the first-stage boundary."""
    if env.stage == 0:
        persistent = {tuple(edge) for edge in env.stage2_edges}
        return [tuple(edge) for edge in env.stage1_edges if tuple(edge) not in persistent]
    return [tuple(edge) for edge in env.stage2_edges]


class RestrictedRouteView:
    """Supply the original planner with current public state and legal routes."""

    __slots__ = ("_base", "_observation", "cfg", "generator")

    def __init__(self, base: CashFlowEnvV2, routes: list[tuple[int, int]]):
        self._base = base
        source = base.get_observation_dict()
        self._observation = {
            "stage": int(source["stage"]),
            "step_in_stage": int(source["step_in_stage"]),
            "balances": deepcopy(source["balances"]),
            "predicted_remaining_outflow": deepcopy(source["predicted_remaining_outflow"]),
            "current_edges": list(routes),
        }
        self.cfg = deepcopy(base.cfg)
        self.generator = SimpleNamespace(cfg=deepcopy(base.generator.cfg))

    def get_observation_dict(self) -> dict:
        return deepcopy(self._observation)

    def compute_heuristic_amount_for_edge(
        self, source: int, target: int, *, stage_aware: bool = True
    ) -> float:
        return self._base.compute_heuristic_amount_for_edge(
            source, target, stage_aware=stage_aware
        )


def evaluate_episode(interface: str, seed: int) -> dict:
    """Run the frozen controller on the canonical second-reset instance."""
    if interface not in INTERFACES:
        raise ValueError(interface)
    base = CashFlowEnvV2(CashFlowEnvConfigV2(seed=int(seed)))
    base.reset()  # Match the scored reset ordinal used by the policy tests.
    base.reset()
    planner = ForecastMeanMILPRollingHorizon()
    rewards: list[float] = []
    selected_routes: list[tuple[int, int]] = []

    while not base.done:
        full_routes = [tuple(edge) for edge in base.get_current_edges()]
        legal_routes = full_routes if interface == "Flat" else mmdp_routes(base)
        view = RestrictedRouteView(base, legal_routes)
        (local_index, ratio), trace = planner.select_action(view)
        decision = trace["decision"]
        if decision["kind"] == "STOP":
            base_index = len(full_routes)
        else:
            edge = tuple(decision["edge"])
            if edge not in legal_routes or edge not in full_routes:
                raise RuntimeError("MILP selected a route outside the active interface")
            if local_index != legal_routes.index(edge):
                raise RuntimeError("MILP route index does not match its selected edge")
            base_index = full_routes.index(edge)
            selected_routes.append(edge)
        _observation, reward, _done, info = base.step((base_index, ratio))
        planner.note_transition()
        rewards.append(float(reward))
        if base_index != len(full_routes):
            planned = float(decision["executable_amount"])
            if abs(float(info.get("transfer_amount", 0.0)) - planned) > 1e-8:
                raise RuntimeError("MILP action differs from the executed transfer amount")
        if len(rewards) > 34:
            raise RuntimeError("Ten-account episode exceeded the decision limit")

    return {
        "interface": interface,
        "environment_seed": int(seed),
        "return": float(sum(rewards)),
        "episode_length": len(rewards),
        "selected_routes": [list(edge) for edge in selected_routes],
        "milp_solves": planner.ledger.milp_solves,
    }


def summarize(rows: list[dict], *, smoke: bool) -> dict:
    by_interface = {
        interface: np.asarray([row["return"] for row in rows if row["interface"] == interface])
        for interface in INTERFACES
    }
    result = {"means": {key: float(value.mean()) for key, value in by_interface.items()}}
    if not smoke:
        difference = by_interface["MMDP"] - by_interface["Flat"]
        rng = np.random.Generator(np.random.PCG64(2026092301))
        indices = rng.integers(0, 200, size=(100000, 200), dtype=np.int64)
        result["paired_instance_contrast"] = {
            "mean": float(difference.mean()),
            "ci95": [float(x) for x in np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; required for full evaluation")
    parser.add_argument("--smoke", action="store_true", help="One instance per interface; not a paper result")
    args = parser.parse_args()
    if not args.smoke and args.output is None:
        parser.error("Full evaluation requires --output")
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    seeds = range(994000, 994001) if args.smoke else TEST_SEEDS
    rows = [evaluate_episode(interface, seed) for seed in seeds for interface in INTERFACES]
    summary = summarize(rows, smoke=args.smoke)
    if args.output is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        with (args.output / "episodes.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
