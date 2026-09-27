"""Public-observation Random controller with a fixed sampling rule."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math
import json
import hashlib
from .wrappers import route_masks as me
from .envs.cash_flow_env import CashFlowEnv
import numpy as np

RANDOM_ENTROPY = 2027091301


@dataclass(frozen=True)
class PublicSnapshot:
    stage: int
    step_in_stage: int
    balances: tuple[float, ...]
    predicted_remaining_outflow: tuple[float, ...]
    forecast_gap: tuple[float, ...]
    shock_target_account: int
    shock_size: float
    master_idx: int
    investment_idx: int
    operational_indices: tuple[int, ...]


@dataclass(frozen=True)
class SemanticAction:
    route: tuple[int, int] | None
    ratio: float


def public_snapshot(base: CashFlowEnv) -> PublicSnapshot:
    # Only get_observation_dict is consulted. The latent snapshot/export APIs are
    # deliberately not reachable from controller code.
    obs = base.get_observation_dict()
    balances = np.asarray(obs["balances"], dtype=np.float64)
    pred = np.asarray(obs["predicted_remaining_outflow"], dtype=np.float64)
    cfg = base.cfg
    return PublicSnapshot(
        stage=int(obs["stage"]),
        step_in_stage=int(obs["step_in_stage"]),
        balances=tuple(float(x) for x in balances),
        predicted_remaining_outflow=tuple(float(x) for x in pred),
        forecast_gap=tuple(float(x) for x in np.maximum(pred - balances, 0.0)),
        shock_target_account=int(obs.get("shock_target_account", -1)),
        shock_size=float(obs.get("shock_size", 0.0)),
        master_idx=int(cfg.master_idx),
        investment_idx=int(cfg.investment_idx),
        operational_indices=tuple(int(x) for x in cfg.operational_indices),
    )


def random_action(
    snapshot: PublicSnapshot,
    routes: Sequence[tuple[int, int]],
    environment_seed: int,
    replicate_id: int,
) -> SemanticAction:
    ss = np.random.SeedSequence(
        [
            RANDOM_ENTROPY,
            int(environment_seed),
            int(replicate_id),
            int(snapshot.stage),
            int(snapshot.step_in_stage),
        ]
    )
    rng = np.random.Generator(np.random.PCG64(ss))
    u_choice, u_ratio = rng.random(2)
    choice = min(int(math.floor(float(u_choice) * (len(routes) + 1))), len(routes))
    route = None if choice == len(routes) else tuple(routes[choice])
    return SemanticAction(route=route, ratio=float(u_ratio))


def execute_semantic(base: CashFlowEnv, action: SemanticAction):
    current = [tuple(map(int, e)) for e in base.get_current_edges()]
    if action.route is None:
        index = len(current)
    else:
        if tuple(action.route) not in current:
            raise RuntimeError(
                f"Illegal semantic route {action.route}; base routes={current}"
            )
        index = current.index(tuple(action.route))
    return base.step((index, float(action.ratio)))


def require(ok, msg):
    if not ok:
        raise AssertionError(msg)


def cj(v):
    return json.dumps(
        me.to_jsonable(v), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def ch(v):
    return hashlib.sha256(cj(v).encode()).hexdigest()


def episode(setting, seed, rep):
    env = me.make_env(setting, seed)
    try:
        require(env.reset_ordinal == 1, "constructor reset")
        obs, _ = env.reset()
        require(env.reset_ordinal == 2, "canonical reset")
        fingerprint = ch(me.exogenous_payload(env))
        base = env.base_env
        trace = []
        total = 0.0
        done = False
        while not done:
            snap = public_snapshot(base)
            routes = tuple(tuple(r) for r in me.current_routes(env))
            before = np.asarray(snap.balances)
            pred = np.asarray(snap.predicted_remaining_outflow)
            # Pass only public observations, not latent or evaluation-only state.
            action = random_action(snap, routes, seed, rep)
            _, reward, done, info = execute_semantic(base, action)
            stop = action.route is None
            require(stop or tuple(action.route) in routes, "illegal route")
            amount = float(info.get("transfer_amount", 0))
            fee = float(info.get("transfer_fee", 0))
            shaping = (
                0.0
                if stop or action.route[1] != base.cfg.investment_idx
                else base.cfg.investment_shaping_weight * amount
            )
            penalty = 0.0 if stop else float(base.cfg.step_penalty)
            ty = float(info.get("terminal_yield", 0))
            tg = float(info.get("terminal_gap_penalty", 0))
            recon = (
                -fee
                + shaping
                - penalty
                + base.cfg.yield_weight * ty
                - base.cfg.gap_weight * tg
            )
            require(abs(reward - recon) < 1e-8, "action reward reconstruction")
            if not stop:
                u, v = action.route
                surplus = max(float(before[u] - pred[u]), 0.0)
                cap = (
                    surplus
                    if v == base.cfg.investment_idx
                    else min(surplus, max(float(pred[v] - before[v]), 0.0))
                )
                expected = max(0.0, min(action.ratio * cap, max(float(before[u]), 0.0)))
                require(abs(expected - amount) < 1e-8, "executor mismatch")
            row = dict(
                setting_id=setting,
                environment_seed=seed,
                policy_rng_seed=rep,
                action_index=len(trace),
                stage=int(snap.stage),
                stage_step=int(snap.step_in_stage),
                legal_routes=routes,
                route=None if stop else action.route,
                ratio=float(action.ratio),
                amount=amount,
                reward=float(reward),
                fee=fee,
                shaping=shaping,
                penalty=penalty,
                terminal_yield=ty,
                terminal_gap=tg,
                yield_weight=float(base.cfg.yield_weight),
                gap_weight=float(base.cfg.gap_weight),
                fee_rate=float(base.cfg.transfer_fee_rate),
                step_penalty_rate=float(base.cfg.step_penalty),
                shaping_rate=float(base.cfg.investment_shaping_weight),
                investment_idx=int(base.cfg.investment_idx),
                balances_before=before.tolist(),
                predicted_before=pred.tolist(),
                done=bool(done),
                reward_reconstruction=recon,
                latent_fingerprint=fingerprint,
            )
            trace.append(row)
            total += float(reward)
            require(
                len(trace) <= base.cfg.stage1_max_steps + base.cfg.stage2_max_steps,
                "episode too long",
            )
        summ = base.episode_summary()
        recon = (
            -float(summ["total_transfer_fee"])
            + sum(r["shaping"] - r["penalty"] for r in trace)
            + base.cfg.yield_weight * float(summ["terminal_yield"])
            - base.cfg.gap_weight * float(summ["terminal_gap_penalty"])
        )
        require(abs(total - recon) < 1e-8, "episode reward reconstruction")
        return (
            dict(
                setting_id=setting,
                environment_seed=seed,
                policy_rng_seed=rep,
                reset_ordinal=2,
                full_return=total,
                episode_length=len(trace),
                latent_fingerprint=fingerprint,
                action_trace_hash=ch(trace),
                reward_reconstruction=recon,
                reconstruction_error=total - recon,
                total_fee=float(summ["total_transfer_fee"]),
                total_step_penalty=sum(r["penalty"] for r in trace),
                terminal_yield=float(summ["terminal_yield"]),
                terminal_gap=float(summ["terminal_gap_penalty"]),
            ),
            trace,
        )
    finally:
        env.close()
