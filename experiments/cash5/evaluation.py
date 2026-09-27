"""Evaluate trusted final PPO policies on shared instances.

Action and reward accounting checks run alongside each episode.
Policy inputs exclude evaluation-only latent diagnostics.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys

sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mmdp-cash5-matplotlib")
)
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn

from .wrappers.route_masks import (
    CFG,
    GLOBAL_ROUTES,
    STAGE1_ROUTES,
    current_routes,
    exogenous_payload,
    load_inventory,
    make_env,
    setting_routes,
    to_jsonable,
)

ROOT = Path(__file__).resolve().parent / "configs"
SETTINGS = ("Flat", "True", "U1", "U2", "U3", "U4", "U5", "C1", "C2", "C3", "C4", "C5")
TRAINING_SEEDS = tuple(range(42, 52))
MASK_TYPES = {
    "Flat": "flat",
    "True": "true",
    **{f"U{i}": "uniform" for i in range(1, 6)},
    **{f"C{i}": "conditional" for i in range(1, 6)},
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(to_jsonable(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    require(bool(rows), "Cannot write an empty endpoint table")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def key(row):
    return row["setting_id"], int(row["training_seed"]), int(row["environment_seed"])


def model_key(row):
    return row["setting_id"], int(row["training_seed"])


def inventory_rows(inventory):
    values = inventory["settings"]
    indexed = (
        {row["setting_id"]: row for row in values}
        if isinstance(values, list)
        else values
    )
    require(
        len(indexed) == 12 and set(indexed) == set(SETTINGS),
        "Inventory must contain exactly 12 settings",
    )
    for setting in SETTINGS:
        require(
            indexed[setting]["mask_type"] == MASK_TYPES[setting],
            f"Incorrect mask family {setting}",
        )
        setting_routes(setting, inventory)
    return indexed


def initialize_worker():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def load_frozen_model(checkpoint_meta, *, smoke=False):
    """Load only trusted project checkpoints; schedule compatibility changes no weights."""
    path = Path(checkpoint_meta["checkpoint_path"])
    require(
        sha256_file(path) == checkpoint_meta["checkpoint_sha256"],
        f"Changed checkpoint: {path}",
    )
    model = PPO.load(
        str(path),
        device="cpu",
        custom_objects={
            "clip_range": 0.2,
            "lr_schedule": get_schedule_fn(0.0003),
            "ep_info_buffer": None,
            "ep_success_buffer": None,
        },
    )
    required_steps = 256 if smoke else 500224
    require(
        int(model.num_timesteps)
        == int(checkpoint_meta["actual_steps"])
        == required_steps,
        f"Wrong {'smoke' if smoke else 'frozen final'} timestep: {path}",
    )
    require(
        int(model.seed) == int(checkpoint_meta["training_seed"]),
        f"Wrong checkpoint training seed: {path}",
    )
    require(
        model.observation_space.shape == (28,) and model.action_space.shape == (2,),
        "Expected a 28-dimensional observation and raw Box(2) PPO action space",
    )
    require(
        np.array_equal(model.action_space.low, np.array([-1.0, -1.0], dtype=np.float32))
        and np.array_equal(
            model.action_space.high, np.array([1.0, 1.0], dtype=np.float32)
        ),
        "Changed raw action bounds",
    )
    require(
        sum(p.numel() for p in model.policy.parameters()) == 40837,
        "Changed raw PPO parameter count",
    )
    model.policy.set_training_mode(False)
    return model


def evaluate_episode(
    model,
    setting_id,
    training_seed,
    environment_seed,
    checkpoint_meta,
    inventory=None,
    *,
    smoke=False,
):
    """Return (episode_row, chronological_action_trace) from constructor + reset #2.

    The only argument to model.predict is the 28-dimensional float32
    observation. Extra latent fields are read solely for logged integrity checks.
    smoke=True permits only 256-step smoke models and development instances
    981000--981199, separate from the 980000--980199 test bank.
    """
    inventory = load_inventory(inventory)
    setting = inventory_rows(inventory)[setting_id]
    require(
        checkpoint_meta["setting_id"] == setting_id
        and int(checkpoint_meta["training_seed"]) == int(training_seed),
        "Checkpoint/episode label mismatch",
    )
    if smoke:
        require(
            981000 <= int(environment_seed) < 981200,
            "Smoke evaluation requires a separate development instance",
        )
        require(
            int(checkpoint_meta["actual_steps"]) == 256,
            "Smoke evaluation requires a 256-step model",
        )
    else:
        require(
            int(checkpoint_meta["actual_steps"]) == 500224,
            "Only the final 500,224-step endpoint is allowed",
        )
    identifiers = dict(
        setting_id=setting_id,
        mask_type=setting["mask_type"],
        design_rng_seed=setting["design_rng_seed"],
        training_seed=int(training_seed),
        environment_seed=int(environment_seed),
        checkpoint_sha256=checkpoint_meta["checkpoint_sha256"],
        actual_steps=int(checkpoint_meta["actual_steps"]),
        evaluation_role="smoke" if smoke else "formal_final",
    )
    env = make_env(setting_id, int(environment_seed), inventory)
    try:
        require(env.reset_ordinal == 1, "Constructor reset count changed")
        observation, reset_info = env.reset()
        require(
            env.reset_ordinal == 2 and reset_info["canonical_reset_ordinal"] == 2,
            "Expected canonical reset #2",
        )
        base = env.base_env
        latent = to_jsonable(exogenous_payload(env))
        fingerprint = canonical_hash(latent)
        initial_observation_hash = canonical_hash(observation)
        model.policy.set_training_mode(False)
        trace = []
        total = 0.0
        done = False
        while not done:
            require(
                np.asarray(observation).shape == (28,)
                and np.asarray(observation).dtype == np.float32
                and np.all(np.isfinite(observation)),
                "Policy input must be a finite 28-dimensional float32 observation",
            )
            stage, stage_step = int(base.stage), int(base.step_in_stage)
            legal_routes = tuple(tuple(route) for route in current_routes(env))
            balances_before = base.balances.copy()
            predicted_before = base._current_predicted_remaining_outflow().copy()
            # No labels, masks, exogenous audit fields, or future values enter this call.
            action, _ = model.predict(observation, deterministic=True)
            raw = np.asarray(action, dtype=np.float32)
            require(
                raw.shape == (2,)
                and np.all(np.isfinite(raw))
                and np.all(raw >= -1.0)
                and np.all(raw <= 1.0),
                "PPO.predict must return a finite clipped raw Box(2) action",
            )
            next_observation, reward, terminated, truncated, info = env.step(raw)
            require(not truncated, "A full endpoint episode cannot be truncated")
            done = bool(terminated)
            route = info["decoded_edge"]
            stop = route == "STOP" if isinstance(route, str) else False
            require(
                stop or tuple(route) in legal_routes,
                "Decoded action outside the fixed interface/stage route set",
            )
            require(
                stop
                == bool(info["wrapper_stop_selected"])
                == bool(info["stop_action"]),
                "STOP decoding disagreement",
            )
            ratio = float(info["decoded_ratio"])
            amount = float(info.get("transfer_amount", 0.0))
            fee = float(info.get("transfer_fee", 0.0))
            shaping = (
                0.0
                if stop or int(route[1]) != int(base.cfg.investment_idx)
                else float(base.cfg.investment_shaping_weight) * amount
            )
            step_penalty = 0.0 if stop else float(base.cfg.step_penalty)
            terminal_yield = float(info.get("terminal_yield", 0.0)) if done else 0.0
            terminal_gap = float(info.get("terminal_gap_penalty", 0.0)) if done else 0.0
            yield_component = float(base.cfg.yield_weight) * terminal_yield
            gap_component = float(base.cfg.gap_weight) * terminal_gap
            reconstructed = (
                -fee + shaping - step_penalty + yield_component - gap_component
            )
            require(
                math.isfinite(float(reward))
                and abs(float(reward) - reconstructed) < 1e-8,
                "Per-action complete reward decomposition mismatch",
            )
            expected_index = min(
                int(np.floor((float(raw[0]) + 1.0) / 2.0 * (len(legal_routes) + 1))),
                len(legal_routes),
            )
            require(
                expected_index == int(info["decoded_edge_index_local"]),
                "Decoded route index disagrees with the action",
            )
            require(
                ratio == (float(raw[1]) + 1.0) / 2.0,
                "Decoded amount ratio disagrees with the action",
            )
            if stop:
                require(amount == fee == 0.0, "STOP cannot incur a transfer")
            else:
                source, target = (int(v) for v in route)
                surplus = max(
                    float(balances_before[source] - predicted_before[source]), 0.0
                )
                capacity = (
                    surplus
                    if target == base.cfg.investment_idx
                    else min(
                        surplus,
                        max(
                            float(predicted_before[target] - balances_before[target]),
                            0.0,
                        ),
                    )
                )
                expected_amount = max(
                    0.0, min(ratio * capacity, max(float(balances_before[source]), 0.0))
                )
                require(
                    abs(amount - expected_amount) < 1e-8
                    and abs(fee - amount * base.cfg.transfer_fee_rate) < 1e-8,
                    "Transfer or fee accounting disagrees with the executed action",
                )
            action_row = dict(
                **identifiers,
                action_index=len(trace),
                stage=stage,
                stage_step=stage_step,
                observation=observation.tolist(),
                raw_action=raw.tolist(),
                route=to_jsonable(route),
                ratio=ratio,
                decoded_index_local=int(info["decoded_edge_index_local"]),
                decoded_index_full=int(info["decoded_edge_index_full"]),
                legal_route_count=len(legal_routes),
                amount=amount,
                reward=float(reward),
                stop=stop,
                zero=bool(not stop and amount <= 1e-8),
                transfer_fee=fee,
                investment_shaping=shaping,
                step_penalty=step_penalty,
                terminal_yield=terminal_yield,
                terminal_gap_penalty=terminal_gap,
                weighted_terminal_yield=yield_component,
                weighted_terminal_gap_penalty=gap_component,
                reward_reconstruction=reconstructed,
                reward_reconstruction_error=float(reward) - reconstructed,
                balances_before=balances_before.tolist(),
                predicted_remaining_outflow_before=predicted_before.tolist(),
                early_persistent=bool(
                    stage == 0 and not stop and tuple(route) in STAGE1_ROUTES
                ),
                terminated=done,
                truncated=False,
                transition_reason=info.get("transition_reason"),
            )
            if not trace:
                action_row["initial_exogenous_payload"] = latent
            if done:
                action_row["final_balances"] = base.balances.tolist()
            trace.append(action_row)
            total += float(reward)
            observation = next_observation
            require(
                len(trace)
                <= int(base.cfg.stage1_max_steps + base.cfg.stage2_max_steps),
                "Episode exceeded the decision budget",
            )
        summary = base.episode_summary()
        require(summary["done"] is True, "Missing complete terminal episode")
        transfer_fee = float(summary["total_transfer_fee"])
        transfer_amount = float(summary["total_transfer_amount"])
        shaping = math.fsum(a["investment_shaping"] for a in trace)
        penalty = math.fsum(a["step_penalty"] for a in trace)
        terminal_yield = float(summary["terminal_yield"])
        terminal_gap = float(summary["terminal_gap_penalty"])
        weighted_yield = float(base.cfg.yield_weight) * terminal_yield
        weighted_gap = float(base.cfg.gap_weight) * terminal_gap
        reconstructed = (
            -transfer_fee + shaping - penalty + weighted_yield - weighted_gap
        )
        require(
            abs(total - reconstructed) < 1e-8,
            "Full episodic return reconstruction mismatch",
        )
        require(
            abs(math.fsum(a["amount"] for a in trace) - transfer_amount) < 1e-8
            and abs(math.fsum(a["transfer_fee"] for a in trace) - transfer_fee) < 1e-8,
            "Episode transfer sums mismatch",
        )
        row = dict(
            **identifiers,
            reset_ordinal=2,
            full_return=float(total),
            episode_length=len(trace),
            total_transfer_fee=transfer_fee,
            total_transfer_amount=transfer_amount,
            total_investment_shaping=shaping,
            total_step_penalty=penalty,
            investment_shaping_weight=float(base.cfg.investment_shaping_weight),
            transfer_fee_rate=float(base.cfg.transfer_fee_rate),
            step_penalty_per_nonstop=float(base.cfg.step_penalty),
            yield_weight=float(base.cfg.yield_weight),
            gap_weight=float(base.cfg.gap_weight),
            terminal_yield=terminal_yield,
            terminal_gap_penalty=terminal_gap,
            weighted_terminal_yield=weighted_yield,
            weighted_terminal_gap_penalty=weighted_gap,
            terminal_reward=float(summary["terminal_reward"]),
            reward_reconstruction=reconstructed,
            reward_reconstruction_error=float(total - reconstructed),
            total_stop_count=int(summary["total_stop_count"]),
            zero_transfer_count=int(summary["zero_transfer_count"]),
            latent_fingerprint=fingerprint,
            initial_observation_sha256=initial_observation_hash,
            observation_dimension=28,
            complete=True,
            truncated=False,
        )
        for stage in (0, 1):
            part = [a for a in trace if a["stage"] == stage]
            require(bool(part), "Both stages must be represented in a full episode")
            stops = sum(a["stop"] for a in part)
            steps = sum(not a["stop"] for a in part)
            require(
                stops == int(summary[f"stop_count_stage{stage}"])
                and steps == int(summary[f"stage{stage}_steps_used"]),
                "Stage counters disagree with the trace",
            )
            row.update(
                {
                    f"stage{stage}_stop_count": stops,
                    f"stage{stage}_steps_used": steps,
                    f"stage{stage}_decisions": len(part),
                    f"stage{stage}_zero_count": sum(a["zero"] for a in part),
                    f"stage{stage}_meaningful_count": sum(
                        a["amount"] > 1e-8 for a in part
                    ),
                    f"stage{stage}_immediate_stop": bool(part[0]["stop"]),
                }
            )
        early = [a for a in trace if a["early_persistent"]]
        row.update(
            early_persistent_action_count=len(early),
            early_persistent_meaningful_count=sum(a["amount"] > 1e-8 for a in early),
            early_persistent_amount=math.fsum(a["amount"] for a in early),
            early_persistent_used=bool(early),
        )
        return row, trace
    finally:
        env.close()
