"""Forecast-mean MILP rolling-horizon controller for synthetic cash management."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass
class ComputeLedger:
    analytic_objective_evaluations: int = 0
    milp_solves: int = 0
    solver_nodes: int = 0
    solver_iterations_reported: int = 0
    action_domain_size_accumulated: int = 0
    selected_actions: int = 0
    simulator_transitions: int = 0
    forecast_updates: int = 0
    executor_calls: int = 0
    wall_time_seconds: float = 0.0


class ForecastMeanMILPRollingHorizon:
    """Non-clairvoyant deterministic-equivalent rolling horizon.

    The information boundary is deliberately narrow: ``select_action`` reads
    only get_observation_dict(), public config coefficients, and current edges.
    It never reads a snapshot or latent generator fields.
    """

    name = "FM-DE-MILP-RH"

    def __init__(self, *, amount_tolerance: float = 1e-7):
        self.amount_tolerance = float(amount_tolerance)
        self.ledger = ComputeLedger()
        self.trace: list[dict[str, Any]] = []

    @staticmethod
    def _positive_normal_mean(mu: float, sigma: float) -> float:
        a = mu / sigma
        phi = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
        Phi = 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))
        return sigma * phi + mu * Phi

    def point_demand(self, env, obs: dict[str, Any]) -> np.ndarray:
        visible = np.asarray(obs["predicted_remaining_outflow"], dtype=float).copy()
        if int(obs["stage"]) == 1:
            return visible

        # Public ex-ante shock model; no realized target/magnitude is read.
        gen = env.generator.cfg
        mean_positive_shock = self._positive_normal_mean(gen.shock_mean, gen.shock_std)
        probs = np.asarray(gen.shock_target_probs, dtype=float)
        probs /= probs.sum()
        for p, idx in zip(probs, gen.operational_indices):
            visible[int(idx)] += float(gen.shock_prob) * float(p) * mean_positive_shock
        return visible

    def _solve(self, env, obs: dict[str, Any], demand: np.ndarray) -> dict[str, Any]:
        edges = [tuple(map(int, e)) for e in obs["current_edges"]]
        n_e, n_a = len(edges), int(env.cfg.num_accounts)
        balances = np.asarray(obs["balances"], dtype=float)
        pred = np.asarray(demand, dtype=float)
        investment = int(env.cfg.investment_idx)
        remaining_actions = (
            int(env.cfg.stage1_max_steps) if int(obs["stage"]) == 0
            else int(env.cfg.stage2_max_steps)
        ) - int(obs["step_in_stage"])

        source_surplus = np.maximum(balances - pred, 0.0)
        target_gap = np.maximum(pred - balances, 0.0)
        caps = np.zeros(n_e, dtype=float)
        for k, (u, v) in enumerate(edges):
            caps[k] = source_surplus[u] if v == investment else min(source_surplus[u], target_gap[v])

        # Variables: amount[e], used[e], positive_terminal[i], negative_terminal[i].
        n_var = 2 * n_e + 2 * n_a
        x0, z0, p0, q0 = 0, n_e, 2 * n_e, 2 * n_e + n_a
        c = np.zeros(n_var, dtype=float)
        c[x0:x0+n_e] = float(env.cfg.transfer_fee_rate)
        c[z0:z0+n_e] = float(env.cfg.step_penalty)
        c[p0:p0+n_a] = -float(env.cfg.yield_weight) * np.asarray(env.cfg.account_yield_rate)
        c[q0:q0+n_a] = float(env.cfg.gap_weight) * np.asarray(env.cfg.account_gap_penalty)

        lower = np.zeros(n_var, dtype=float)
        upper = np.full(n_var, np.inf, dtype=float)
        upper[x0:x0+n_e] = caps
        upper[z0:z0+n_e] = 1.0
        integrality = np.zeros(n_var, dtype=int)
        integrality[z0:z0+n_e] = 1

        # Terminal flow conservation.
        a_eq = np.zeros((n_a, n_var), dtype=float)
        for k, (u, v) in enumerate(edges):
            a_eq[u, x0+k] += 1.0
            a_eq[v, x0+k] -= 1.0
        for i in range(n_a):
            a_eq[i, p0+i] = 1.0
            a_eq[i, q0+i] = -1.0
        rhs = balances - pred

        rows, ub = [], []
        # Link route amount to binary route use.
        for k in range(n_e):
            row = np.zeros(n_var); row[x0+k] = 1.0; row[z0+k] = -caps[k]
            rows.append(row); ub.append(0.0)
        # Shared source-surplus and target-gap budgets reproduce executor caps.
        for i in range(n_a):
            row = np.zeros(n_var)
            for k, (u, _v) in enumerate(edges):
                if u == i: row[x0+k] = 1.0
            rows.append(row); ub.append(float(source_surplus[i]))
        for j in range(n_a):
            if j == investment: continue
            row = np.zeros(n_var)
            for k, (_u, v) in enumerate(edges):
                if v == j: row[x0+k] = 1.0
            rows.append(row); ub.append(float(target_gap[j]))
        row = np.zeros(n_var); row[z0:z0+n_e] = 1.0
        rows.append(row); ub.append(float(max(remaining_actions, 0)))

        constraints = [LinearConstraint(a_eq, rhs, rhs)]
        if rows:
            constraints.append(LinearConstraint(np.asarray(rows), -np.inf, np.asarray(ub)))

        started = time.perf_counter()
        result = milp(c, integrality=integrality, bounds=Bounds(lower, upper), constraints=constraints,
                      options={"presolve": True, "mip_rel_gap": 0.0})
        elapsed = time.perf_counter() - started
        self.ledger.milp_solves += 1
        self.ledger.analytic_objective_evaluations += 1
        self.ledger.action_domain_size_accumulated += n_e + 1
        self.ledger.wall_time_seconds += elapsed
        self.ledger.solver_nodes += int(getattr(result, "mip_node_count", 0) or 0)
        self.ledger.solver_iterations_reported += int(getattr(result, "nit", 0) or 0)
        if not result.success:
            raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")
        amounts = np.asarray(result.x[x0:x0+n_e], dtype=float)
        return {"edges": edges, "amounts": amounts, "objective": float(result.fun),
                "solver_message": str(result.message), "solve_seconds": elapsed,
                "remaining_action_budget": remaining_actions}

    def select_action(self, env) -> tuple[tuple[int, float], dict[str, Any]]:
        started = time.perf_counter()
        obs = env.get_observation_dict()
        demand = self.point_demand(env, obs)
        self.ledger.forecast_updates += 1
        plan = self._solve(env, obs, demand)
        amounts = plan["amounts"]

        positive = np.flatnonzero(amounts > self.amount_tolerance)
        if positive.size == 0:
            action = (len(obs["current_edges"]), 1.0)
            decision = {"kind": "STOP", "planned_amount": 0.0, "executable_amount": 0.0,
                        "ratio": 1.0}
        else:
            # Largest amount, then lexicographic edge: deterministic sequencing.
            ranked = sorted((int(k) for k in positive), key=lambda k: (-amounts[k], plan["edges"][k]))
            k = ranked[0]
            edge = plan["edges"][k]
            full_edges = [tuple(e) for e in obs["current_edges"]]
            edge_index = full_edges.index(edge)
            self.ledger.executor_calls += 1
            cap = float(env.compute_heuristic_amount_for_edge(*edge, stage_aware=True))
            executable = min(float(amounts[k]), cap)
            if executable <= self.amount_tolerance or cap <= self.amount_tolerance:
                action = (len(full_edges), 1.0)
                decision = {"kind": "STOP", "planned_edge": edge,
                            "planned_amount": float(amounts[k]), "executable_amount": 0.0, "ratio": 1.0}
            else:
                ratio = float(np.clip(executable / cap, 0.0, 1.0))
                action = (edge_index, ratio)
                decision = {"kind": "TRANSFER", "edge": edge, "edge_index": edge_index,
                            "planned_amount": float(amounts[k]), "executor_cap": cap,
                            "executable_amount": executable, "ratio": ratio}

        self.ledger.selected_actions += 1
        self.ledger.wall_time_seconds += time.perf_counter() - started - plan["solve_seconds"]
        record = {"stage": int(obs["stage"]), "step_in_stage": int(obs["step_in_stage"]),
                  "visible_forecast": np.asarray(obs["predicted_remaining_outflow"]).tolist(),
                  "planning_demand": demand.tolist(), "route_count": len(obs["current_edges"]),
                  "milp_objective": plan["objective"], "decision": decision,
                  "information_fields_read": ["stage", "step_in_stage", "balances",
                                              "predicted_remaining_outflow", "current_edges"]}
        self.trace.append(record)
        return action, record

    def note_transition(self) -> None:
        self.ledger.simulator_transitions += 1

    def accounting(self) -> dict[str, Any]:
        return asdict(self.ledger)
