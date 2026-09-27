"""Evaluate the forecast-based rolling-horizon MILP on five-account routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.milp_planner import ForecastMeanMILPRollingHorizon
from .envs.cash_flow_env import CashFlowEnv, CashFlowEnvConfig


SETTINGS = ("Flat", "True", "C1", "C2", "C3", "C4", "C5")
TEST_SEEDS = range(980000, 980200)
MASK_FILE = Path(__file__).resolve().parent / "configs" / "mask_inventory.json"


def route_lists() -> dict[str, list[tuple[int, int]]]:
    """Read the fixed route sets used by the PPO mask comparison."""
    inventory = json.loads(MASK_FILE.read_text(encoding="utf-8"))
    selected = {
        row["setting_id"]: [tuple(edge) for edge in row["stage0_routes"]]
        for row in inventory["settings"]
        if row["setting_id"] in SETTINGS
    }
    if set(selected) != set(SETTINGS):
        raise ValueError("The fixed five-account MILP route sets are incomplete")
    cfg = CashFlowEnvConfig()
    if [tuple(edge) for edge in inventory["persistent_routes"]] != list(cfg.stage2_edges):
        raise ValueError("The mask inventory disagrees with the persistent route set")
    return selected


class VisibleRouteView:
    """Expose only current observations and public coefficients to the planner."""

    __slots__ = ("cfg", "generator", "balances", "observation")

    def __init__(self, env: CashFlowEnv, routes: list[tuple[int, int]]):
        cfg_fields = (
            "num_accounts", "investment_idx", "stage1_max_steps", "stage2_max_steps",
            "transfer_fee_rate", "step_penalty", "yield_weight", "gap_weight",
            "account_yield_rate", "account_gap_penalty",
        )
        generator_fields = (
            "shock_mean", "shock_std", "shock_target_probs", "operational_indices",
            "shock_prob",
        )
        source = env.get_observation_dict()
        self.cfg = SimpleNamespace(**{key: getattr(env.cfg, key) for key in cfg_fields})
        public_generator = SimpleNamespace(
            **{key: getattr(env.generator.cfg, key) for key in generator_fields}
        )
        self.generator = SimpleNamespace(cfg=public_generator)
        self.observation = {
            "stage": int(source["stage"]),
            "step_in_stage": int(source["step_in_stage"]),
            "balances": np.asarray(source["balances"], dtype=float).copy(),
            "predicted_remaining_outflow": np.asarray(
                source["predicted_remaining_outflow"], dtype=float
            ).copy(),
            "current_edges": list(routes),
        }
        self.balances = self.observation["balances"]

    def get_observation_dict(self) -> dict:
        return self.observation

    def _current_predicted_remaining_outflow(self) -> np.ndarray:
        return self.observation["predicted_remaining_outflow"]

    def compute_heuristic_amount_for_edge(
        self, source: int, target: int, *, stage_aware: bool = True
    ) -> float:
        # Use the unchanged five-account executor on this detached public view.
        return CashFlowEnv._heuristic_transfer_amount(self, source, target, 1.0)


def evaluate_episode(setting: str, seed: int, routes: dict | None = None) -> dict:
    """Run one deterministic controller on the canonical second reset."""
    if setting not in SETTINGS:
        raise ValueError(setting)
    routes = route_lists() if routes is None else routes
    env = CashFlowEnv(CashFlowEnvConfig(seed=int(seed)))
    env.reset()  # Match the historical wrapper's discarded constructor reset.
    env.reset()
    planner = ForecastMeanMILPRollingHorizon()
    rewards: list[float] = []
    selected_routes: list[tuple[int, int]] = []

    while not env.done:
        visible_routes = routes[setting] if env.stage == 0 else list(env.cfg.stage2_edges)
        view = VisibleRouteView(env, visible_routes)
        (local_index, ratio), decision = planner.select_action(view)
        base_routes = list(env.get_current_edges())
        if local_index == len(visible_routes):
            base_index = len(base_routes)
        else:
            edge = visible_routes[local_index]
            if edge not in base_routes or tuple(decision["decision"]["edge"]) != edge:
                raise RuntimeError("MILP selected a route outside the active interface")
            base_index = base_routes.index(edge)
            selected_routes.append(edge)
        _observation, reward, _done, info = env.step((base_index, ratio))
        planner.note_transition()
        rewards.append(float(reward))
        if base_index != len(base_routes):
            planned = float(decision["decision"]["executable_amount"])
            if abs(float(info.get("transfer_amount", 0.0)) - planned) > 1e-8:
                raise RuntimeError("MILP action differs from the executed transfer amount")
        if len(rewards) > 22:
            raise RuntimeError("Five-account episode exceeded the decision limit")

    return {
        "setting": setting,
        "environment_seed": int(seed),
        "return": float(sum(rewards)),
        "episode_length": len(rewards),
        "selected_routes": [list(edge) for edge in selected_routes],
        "milp_solves": planner.ledger.milp_solves,
    }


def summarize(rows: list[dict], *, smoke: bool) -> dict:
    by_setting = {
        setting: np.asarray([row["return"] for row in rows if row["setting"] == setting])
        for setting in SETTINGS
    }
    result = {"means": {key: float(value.mean()) for key, value in by_setting.items()}}
    if not smoke:
        contrasts = {
            "MMDP_minus_Flat": by_setting["True"] - by_setting["Flat"],
            "MMDP_minus_Mask": by_setting["True"] - np.mean(
                [by_setting[f"C{i}"] for i in range(1, 6)], axis=0
            ),
        }
        # Identical routes can leave sub-picounit averaging roundoff.
        contrasts = {
            name: np.where(np.abs(values) < 1e-10, 0.0, values)
            for name, values in contrasts.items()
        }
        rng = np.random.Generator(np.random.PCG64(2026092302))
        indices = rng.integers(0, 200, size=(100000, 200), dtype=np.int64)
        result["paired_instance_contrasts"] = {
            name: {
                "mean": float(values.mean()),
                "ci95": [float(x) for x in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])],
            }
            for name, values in contrasts.items()
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
    routes = route_lists()
    seeds = range(980000, 980001) if args.smoke else TEST_SEEDS
    rows = [evaluate_episode(setting, seed, routes) for seed in seeds for setting in SETTINGS]
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
