"""Visible-information DFM-Q9 Search for CashFlowEnvV2.

The public controller API accepts a frozen policy, a plain visible state, and a
wrapper factory.  It never accepts the real environment and has no snapshot
API.  Complete latent fields exist only inside newly constructed sanitized
planning worlds.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from .envs.cash_flow_env_v2 import CashFlowEnvConfigV2, CashFlowEnvV2


PLANNING_SEED_BASE = 990_000


@dataclass(frozen=True)
class MethodSpec:
    method: str
    wrapper_name: str
    discrete: bool
    ratio_grid: tuple[float, ...]
    include_policy_ratio: bool
    max_candidates: int | None


METHOD_SPECS: dict[str, MethodSpec] = {
    "flat_ppo": MethodSpec(
        "flat_ppo", "FlatCashGymEnvV2", False, (0.25, 0.5, 0.75, 1.0), True, 32
    ),
    "mmdp_ppo": MethodSpec(
        "mmdp_ppo", "MMDPCashGymEnvV2", False,
        (0.25, 0.5, 0.75, 1.0), True, 32,
    ),
    "flat_aa_ppo": MethodSpec(
        "flat_aa_ppo", "FlatCashGymEdgeOnlyEnvV2", True,
        (0.25, 0.5, 0.75, 1.0), True, 16,
    ),
    "mmdp_aa_ppo": MethodSpec(
        "mmdp_aa_ppo", "MMDPCashGymEdgeOnlyEnvV2", True,
        (0.25, 0.5, 0.75, 1.0), True, 16,
    ),
}


@dataclass(frozen=True)
class PublicCashModel:
    shock_probability: Decimal = Decimal("0.90")
    shock_target_probabilities: tuple[Decimal, ...] = (
        Decimal("0.16"), Decimal("0.15"), Decimal("0.14"), Decimal("0.13"),
        Decimal("0.12"), Decimal("0.11"), Decimal("0.10"), Decimal("0.09"),
    )
    shock_mean: float = 28.0
    shock_std: float = 9.0
    operational_indices: tuple[int, ...] = tuple(range(2, 10))

    def validate(self) -> None:
        if sum(self.shock_target_probabilities, Decimal("0")) != Decimal("1"):
            raise ValueError("public shock target probabilities do not sum to one")
        if len(self.shock_target_probabilities) != len(self.operational_indices):
            raise ValueError("shock target probability length mismatch")


@dataclass(frozen=True)
class VisibleState:
    stage: int
    step_in_stage: int
    balances: tuple[float, ...]
    visible_forecast: tuple[float, ...]
    current_routes: tuple[tuple[int, int], ...]
    shock_target_account: int
    shock_size: float

    @classmethod
    def from_observation_dict(cls, payload: dict[str, Any]) -> "VisibleState":
        allowed = {
            "stage", "step_in_stage", "balances", "predicted_remaining_outflow",
            "current_edges", "edge_count", "shock_target_account", "shock_size",
        }
        missing = {
            "stage", "step_in_stage", "balances", "predicted_remaining_outflow",
            "current_edges", "shock_target_account", "shock_size",
        } - set(payload)
        if missing:
            raise KeyError(f"visible observation missing keys: {sorted(missing)}")
        # Deliberately iterate only the allow-list; additional poisoned/latent
        # keys in a mapping are never inspected.
        view = {key: payload[key] for key in allowed if key in payload}
        return cls(
            stage=int(view["stage"]),
            step_in_stage=int(view["step_in_stage"]),
            balances=tuple(float(x) for x in view["balances"]),
            visible_forecast=tuple(float(x) for x in view["predicted_remaining_outflow"]),
            current_routes=tuple(tuple(int(y) for y in x) for x in view["current_edges"]),
            shock_target_account=int(view["shock_target_account"]),
            shock_size=float(view["shock_size"]),
        )

    def validate(self) -> None:
        if self.stage not in (0, 1):
            raise ValueError(f"invalid stage {self.stage}")
        if len(self.balances) != 10 or len(self.visible_forecast) != 10:
            raise ValueError("Cash-10 visible vectors must have length ten")
        if self.step_in_stage < 0 or self.step_in_stage >= 16:
            raise ValueError("step_in_stage outside active decision range")
        if self.stage == 0 and (self.shock_target_account != -1 or self.shock_size != 0.0):
            raise ValueError("stage-0 visible state must not expose shock realization")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    weight_decimal: str
    shock_target_account: int
    shock_size: float
    terminal_outflow: tuple[float, ...]

    @property
    def weight(self) -> float:
        return float(Decimal(self.weight_decimal))

    def canonical(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "weight_decimal": self.weight_decimal,
            "shock_target_account": self.shock_target_account,
            "shock_size": self.shock_size,
            "terminal_outflow": list(self.terminal_outflow),
        }


@dataclass(frozen=True)
class Candidate:
    kind: str
    local_edge_idx: int | None = None
    ratio: float | None = None
    source: int | None = None
    target: int | None = None

    def semantic_key(self) -> tuple[Any, ...]:
        # STOP wins exact score ties; other ties use a deterministic semantic order.
        if self.kind == "stop":
            return (0, -1, -1, -1.0)
        ratio = 1.0 if self.ratio is None else round(float(self.ratio), 6)
        return (1, int(self.source), int(self.target), ratio)

    def canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "local_edge_idx": self.local_edge_idx,
            "ratio": self.ratio,
            "source": self.source,
            "target": self.target,
        }


@dataclass
class ComputeLedger:
    candidates_generated: int = 0
    candidates_scheduled: int = 0
    candidates_scored: int = 0
    candidates_skipped_cap: int = 0
    scenario_worlds_scored: int = 0
    planning_transitions: int = 0
    boundary_crossing_transitions: int = 0
    terminal_transitions: int = 0
    policy_predict_calls: int = 0
    policy_distribution_calls: int = 0
    executor_calls: int = 0
    cap: int = 0
    conservative_bundle_bound: int = 0
    cap_exhausted: bool = False
    wall_seconds: float = 0.0

    @property
    def policy_forwards(self) -> int:
        return self.policy_predict_calls + self.policy_distribution_calls

    def public_record(self) -> dict[str, Any]:
        # Scores, selected actions, and returns are intentionally absent.
        return {
            "candidates_generated": self.candidates_generated,
            "candidates_scheduled": self.candidates_scheduled,
            "candidates_scored": self.candidates_scored,
            "candidates_skipped_cap": self.candidates_skipped_cap,
            "scenario_worlds_scored": self.scenario_worlds_scored,
            "planning_transitions": self.planning_transitions,
            "boundary_crossing_transitions": self.boundary_crossing_transitions,
            "terminal_transitions": self.terminal_transitions,
            "policy_predict_calls": self.policy_predict_calls,
            "policy_distribution_calls": self.policy_distribution_calls,
            "policy_forwards": self.policy_forwards,
            "executor_calls": self.executor_calls,
            "transition_cap": self.cap,
            "conservative_bundle_bound": self.conservative_bundle_bound,
            "cap_exhausted": self.cap_exhausted,
            "wall_seconds": self.wall_seconds,
        }


@dataclass
class SearchResult:
    action: Any
    candidate: Candidate
    ledger: ComputeLedger
    scenario_bank_hash: str
    score_digest: str = field(repr=False)


def positive_part_normal_mean(mu: float, sigma: float) -> float:
    """E[max(0, X)] for X ~ Normal(mu, sigma)."""
    if sigma <= 0:
        return max(float(mu), 0.0)
    z = float(mu) / float(sigma)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(sigma * phi + mu * Phi)


def build_scenario_bank(
    state: VisibleState,
    public_model: PublicCashModel = PublicCashModel(),
) -> tuple[tuple[Scenario, ...], str]:
    state.validate()
    public_model.validate()
    forecast = np.asarray(state.visible_forecast, dtype=np.float64)
    if state.stage == 1:
        scenario = Scenario(
            scenario_id="stage1_visible_exact",
            weight_decimal="1",
            shock_target_account=state.shock_target_account,
            shock_size=state.shock_size,
            terminal_outflow=tuple(float(x) for x in forecast),
        )
        bank = (scenario,)
    else:
        no_shock_weight = Decimal("1") - public_model.shock_probability
        scenarios: list[Scenario] = [
            Scenario(
                scenario_id="no_shock",
                weight_decimal=str(no_shock_weight),
                shock_target_account=-1,
                shock_size=0.0,
                terminal_outflow=tuple(float(x) for x in forecast),
            )
        ]
        magnitude = positive_part_normal_mean(
            public_model.shock_mean, public_model.shock_std
        )
        for account, target_probability in zip(
            public_model.operational_indices,
            public_model.shock_target_probabilities,
        ):
            outflow = forecast.copy()
            outflow[int(account)] += magnitude
            weight = public_model.shock_probability * target_probability
            scenarios.append(
                Scenario(
                    scenario_id=f"shock_target_{account}",
                    weight_decimal=str(weight),
                    shock_target_account=int(account),
                    shock_size=magnitude,
                    terminal_outflow=tuple(float(x) for x in outflow),
                )
            )
        bank = tuple(scenarios)
    exact_sum = sum((Decimal(s.weight_decimal) for s in bank), Decimal("0"))
    if exact_sum != Decimal("1"):
        raise AssertionError(f"scenario weights sum to {exact_sum}, not one")
    payload = [scenario.canonical() for scenario in bank]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return bank, digest


def _shock_vector(scenario: Scenario) -> np.ndarray:
    vector = np.zeros(10, dtype=np.float64)
    if scenario.shock_target_account >= 0:
        vector[scenario.shock_target_account] = float(scenario.shock_size)
    return vector


def build_sanitized_world(
    state: VisibleState,
    scenario: Scenario,
    wrapper_factory: Callable[[CashFlowEnvV2], Any],
    scenario_ordinal: int,
) -> tuple[Any, np.ndarray]:
    """Construct a complete planning world without a real environment object."""
    state.validate()
    planning_base = CashFlowEnvV2(
        CashFlowEnvConfigV2(seed=PLANNING_SEED_BASE + int(scenario_ordinal))
    )
    planning_env = wrapper_factory(planning_base)

    forecast = np.asarray(state.visible_forecast, dtype=np.float64)
    terminal = np.asarray(scenario.terminal_outflow, dtype=np.float64)
    shock = _shock_vector(scenario)
    base_point = np.maximum(terminal - shock, 0.0)

    # Synthetic writes only. No field on a real evaluation environment is read.
    planning_base.stage = int(state.stage)
    planning_base.step_in_stage = int(state.step_in_stage)
    planning_base.done = False
    planning_base.balances = np.asarray(state.balances, dtype=np.float64).copy()
    planning_base.base_total_demand = base_point.copy()
    planning_base.true_total_outflow = terminal.copy()
    planning_base.seasonal_base_forecast = (
        forecast.copy() if state.stage == 0 else base_point.copy()
    )
    planning_base.updated_total_forecast = terminal.copy()
    planning_base.shock_target_account = int(scenario.shock_target_account)
    planning_base.shock_size = float(scenario.shock_size)
    planning_base.shock_vector = shock.copy()
    planning_base.total_transfer_fee = 0.0
    planning_base.total_transfer_amount = 0.0
    planning_base.terminal_yield = 0.0
    planning_base.terminal_gap_penalty = 0.0
    planning_base.stop_count_stage0 = 0
    planning_base.stop_count_stage1 = 0
    planning_base.total_stop_count = 0
    planning_base.zero_transfer_count = 0
    planning_base.stage0_steps_used = 0
    planning_base.stage1_steps_used = 0
    planning_base.transition_history = []
    planning_base.last_info = {}

    synthetic_routes = tuple(planning_env._get_mmdp_edges_for_current_stage())
    if synthetic_routes != state.current_routes:
        raise AssertionError(
            f"sanitized route mismatch: expected {state.current_routes}, got {synthetic_routes}"
        )
    obs = np.asarray(planning_env._build_obs(), dtype=np.float32)
    return planning_env, obs


def _normalize_action(action: Any) -> Any:
    if np.isscalar(action):
        return int(action)
    array = np.asarray(action)
    if array.size == 1:
        return int(array.reshape(-1)[0])
    return array.astype(np.float32)


def _predict(model: Any, obs: np.ndarray, ledger: ComputeLedger) -> Any:
    action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
    ledger.policy_predict_calls += 1
    return _normalize_action(action)


def _discrete_scores(model: Any, obs: np.ndarray, ledger: ComputeLedger) -> np.ndarray | None:
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(np.asarray(obs, dtype=np.float32)[None, :])
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_tensor)
        ledger.policy_distribution_calls += 1
        base = distribution.distribution
        values = base.probs if getattr(base, "probs", None) is not None else base.logits
        output = values.detach().cpu().numpy()
        return np.asarray(output[0] if output.ndim == 2 else output, dtype=np.float64)
    except Exception:
        return None


def _decode_continuous(action: Any, num_edges: int) -> tuple[int, float]:
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError("continuous Cash action must have two coordinates")
    selector = (float(np.clip(arr[0], -1.0, 1.0)) + 1.0) / 2.0
    idx = min(int(math.floor(selector * (num_edges + 1))), num_edges)
    ratio = (float(np.clip(arr[1], -1.0, 1.0)) + 1.0) / 2.0
    return idx, float(np.clip(ratio, 0.0, 1.0))


def _candidate_action(candidate: Candidate, num_edges: int, discrete: bool) -> Any:
    if discrete:
        return int(num_edges if candidate.kind == "stop" else candidate.local_edge_idx)
    if candidate.kind == "stop":
        local_idx, ratio = num_edges, 1.0
    else:
        local_idx, ratio = int(candidate.local_edge_idx), float(candidate.ratio)
    selector = (float(local_idx) + 0.5) / float(num_edges + 1)
    return np.asarray(
        [float(np.clip(2.0 * selector - 1.0, -1.0, 1.0)), 2.0 * ratio - 1.0],
        dtype=np.float32,
    )


def _candidate_for(
    kind: str,
    routes: Sequence[tuple[int, int]],
    local_edge_idx: int | None = None,
    ratio: float | None = None,
) -> Candidate:
    if kind == "stop":
        return Candidate("stop")
    source, target = routes[int(local_edge_idx)]
    return Candidate("transfer", int(local_edge_idx), ratio, int(source), int(target))


def build_candidates(
    model: Any,
    obs: np.ndarray,
    state: VisibleState,
    method_spec: MethodSpec,
    ledger: ComputeLedger,
) -> tuple[list[Candidate], dict[str, Any]]:
    routes = state.current_routes
    n = len(routes)
    policy_action = _predict(model, obs, ledger)
    policy_meta: dict[str, Any] = {
        "action": policy_action,
        "scores": None,
        "num_edges": n,
    }

    if method_spec.discrete:
        policy_idx = int(policy_action)
        if policy_idx < 0 or policy_idx >= n:
            policy_idx = n
        candidates = [_candidate_for("stop", routes)] + [
            _candidate_for("transfer", routes, idx, None) for idx in range(n)
        ]
        scores = None
        if method_spec.max_candidates is not None and len(candidates) > method_spec.max_candidates:
            keep = max(0, method_spec.max_candidates - 1)
            scores = _discrete_scores(model, obs, ledger)
            if scores is not None and len(scores) >= n:
                preferred = [int(x) for x in np.argsort(-scores[:n])[:keep]]
                if policy_idx < n and policy_idx not in preferred:
                    preferred = [policy_idx] + preferred[: max(0, keep - 1)]
            else:
                center = policy_idx if policy_idx < n else 0
                left = max(0, center - keep // 2)
                right = min(n, left + keep)
                left = max(0, right - keep)
                preferred = list(range(left, right))
            candidates = [_candidate_for("stop", routes)] + [
                _candidate_for("transfer", routes, idx, None) for idx in preferred[:keep]
            ]
        policy_meta.update(policy_idx=policy_idx, policy_ratio=None, scores=scores)
    else:
        policy_idx, policy_ratio = _decode_continuous(policy_action, n)
        ratios = list(method_spec.ratio_grid)
        if method_spec.include_policy_ratio:
            ratios.append(policy_ratio)
        ratios = sorted({float(np.clip(round(x, 6), 0.0, 1.0)) for x in ratios})
        candidates = [_candidate_for("stop", routes)]
        for idx in range(n):
            for ratio in ratios:
                candidates.append(_candidate_for("transfer", routes, idx, ratio))
        if method_spec.max_candidates is not None and len(candidates) > method_spec.max_candidates:
            if policy_idx < n:
                preferred = [policy_idx]
                if policy_idx - 1 >= 0:
                    preferred.append(policy_idx - 1)
                if policy_idx + 1 < n:
                    preferred.append(policy_idx + 1)
            else:
                preferred = list(range(min(n, 3)))
            preferred = list(dict.fromkeys(preferred))
            pruned = [_candidate_for("stop", routes)]
            for idx in preferred:
                for ratio in ratios:
                    pruned.append(_candidate_for("transfer", routes, idx, ratio))
            candidates = pruned[: method_spec.max_candidates]
        policy_meta.update(policy_idx=policy_idx, policy_ratio=policy_ratio, scores=None)
    ledger.candidates_generated = len(candidates)
    return candidates, policy_meta


def _schedule_candidates(
    candidates: Iterable[Candidate],
    policy_meta: dict[str, Any],
    method_spec: MethodSpec,
) -> list[Candidate]:
    policy_idx = int(policy_meta["policy_idx"])
    num_edges = int(policy_meta["num_edges"])
    policy_ratio = policy_meta.get("policy_ratio")
    discrete_scores = policy_meta.get("scores")

    def priority(candidate: Candidate) -> tuple[Any, ...]:
        if candidate.kind == "stop":
            is_policy = policy_idx == num_edges
            return (1 if not is_policy else 0, 0, 0.0, candidate.semantic_key())
        idx = int(candidate.local_edge_idx)
        if method_spec.discrete:
            if idx == policy_idx:
                return (0, 0, 0.0, candidate.semantic_key())
            score_rank = -float(discrete_scores[idx]) if discrete_scores is not None else float(abs(idx - policy_idx))
            return (2, 0, score_rank, candidate.semantic_key())
        ratio = float(candidate.ratio)
        if idx == policy_idx and round(ratio, 6) == round(float(policy_ratio), 6):
            return (0, 0, 0.0, candidate.semantic_key())
        return (
            2,
            abs(idx - policy_idx),
            abs(ratio - float(policy_ratio)),
            candidate.semantic_key(),
        )

    # Input order is discarded by this total deterministic key.
    return sorted(list(candidates), key=priority)


def _max_remaining_transitions(state: VisibleState) -> int:
    if state.stage == 0:
        return (16 - state.step_in_stage) + 16
    return 16 - state.step_in_stage


def _executor_called(env: Any, action: Any, method_spec: MethodSpec) -> bool:
    n = len(env._get_mmdp_edges_for_current_stage())
    if method_spec.discrete:
        return int(env._decode_action(int(action), n)) < n
    idx, _ = env._decode_action(np.asarray(action, dtype=np.float32), n)
    return int(idx) < n


def _score_candidate(
    *,
    model: Any,
    state: VisibleState,
    bank: Sequence[Scenario],
    candidate: Candidate,
    method_spec: MethodSpec,
    wrapper_factory: Callable[[CashFlowEnvV2], Any],
    ledger: ComputeLedger,
) -> float:
    weighted_score = 0.0
    for ordinal, scenario in enumerate(bank):
        planning_env, _ = build_sanitized_world(state, scenario, wrapper_factory, ordinal)
        action = _candidate_action(candidate, len(state.current_routes), method_spec.discrete)
        while True:
            before_stage = int(planning_env.base_env.stage)
            if _executor_called(planning_env, action, method_spec):
                ledger.executor_calls += 1
            next_obs, reward, terminated, truncated, _ = planning_env.step(action)
            ledger.planning_transitions += 1
            after_stage = int(planning_env.base_env.stage)
            if before_stage == 0 and after_stage == 1:
                ledger.boundary_crossing_transitions += 1
            done = bool(terminated or truncated)
            if done:
                ledger.terminal_transitions += 1
                weighted_score += scenario.weight * float(reward)
                break
            weighted_score += scenario.weight * float(reward)
            action = _predict(model, np.asarray(next_obs, dtype=np.float32), ledger)
        planning_env.close()
        ledger.scenario_worlds_scored += 1
    return float(weighted_score)


def _score_digest(scores: Sequence[tuple[Candidate, float]]) -> str:
    payload = [
        {"candidate": candidate.canonical(), "score_hex": float(score).hex()}
        for candidate, score in sorted(scores, key=lambda x: x[0].semantic_key())
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def select_dfm_q9_action(
    *,
    model: Any,
    current_obs: np.ndarray,
    state: VisibleState,
    method_spec: MethodSpec,
    wrapper_factory: Callable[[CashFlowEnvV2], Any],
    transition_cap: int,
    candidate_input_permutation: Sequence[int] | None = None,
) -> SearchResult:
    """Select one action with complete-bundle, transition-capped DFM-Q9."""
    started = time.perf_counter()
    ledger = ComputeLedger(cap=int(transition_cap))
    bank, bank_hash = build_scenario_bank(state)
    candidates, policy_meta = build_candidates(
        model, np.asarray(current_obs, dtype=np.float32), state, method_spec, ledger
    )
    if candidate_input_permutation is not None:
        permutation = list(candidate_input_permutation)
        if sorted(permutation) != list(range(len(candidates))):
            raise ValueError("candidate permutation is not a bijection")
        candidates = [candidates[idx] for idx in permutation]
    scheduled = _schedule_candidates(candidates, policy_meta, method_spec)
    ledger.candidates_scheduled = len(scheduled)
    bundle_bound = len(bank) * _max_remaining_transitions(state)
    ledger.conservative_bundle_bound = bundle_bound

    scores: list[tuple[Candidate, float]] = []
    for candidate in scheduled:
        if ledger.planning_transitions + bundle_bound > ledger.cap:
            ledger.candidates_skipped_cap += 1
            ledger.cap_exhausted = True
            continue
        score = _score_candidate(
            model=model,
            state=state,
            bank=bank,
            candidate=candidate,
            method_spec=method_spec,
            wrapper_factory=wrapper_factory,
            ledger=ledger,
        )
        scores.append((candidate, score))
        ledger.candidates_scored += 1
    if not scores:
        raise RuntimeError("transition cap cannot fund one complete scenario bundle")
    best_score = max(score for _, score in scores)
    tied = [candidate for candidate, score in scores if score == best_score]
    selected = min(tied, key=lambda candidate: candidate.semantic_key())
    action = _candidate_action(selected, len(state.current_routes), method_spec.discrete)
    ledger.wall_seconds = float(time.perf_counter() - started)
    return SearchResult(
        action=action,
        candidate=selected,
        ledger=ledger,
        scenario_bank_hash=bank_hash,
        score_digest=_score_digest(scores),
    )


def reconstruct_terminal_reward(base: CashFlowEnvV2) -> float:
    positive = np.maximum(np.asarray(base.balances, dtype=np.float64), 0.0)
    negative = np.maximum(-np.asarray(base.balances, dtype=np.float64), 0.0)
    yields = np.asarray(base.cfg.account_yield_rate, dtype=np.float64)
    gaps = np.asarray(base.cfg.account_gap_penalty, dtype=np.float64)
    return float(base.cfg.yield_weight * np.sum(positive * yields) - base.cfg.gap_weight * np.sum(negative * gaps))
