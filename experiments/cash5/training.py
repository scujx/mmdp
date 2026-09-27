"""PPO training for Flat, MMDP, and fixed route-mask controls."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .runtime import (
    CHECKPOINT_FREQ,
    INTERFACES,
    NOMINAL_STEPS,
    PPO_KWARGS,
    make_env,
    model_metadata,
    now_iso,
    set_determinism,
    sha256_file,
    system_inventory,
    write_json,
)


import hashlib
import random
import resource
import traceback
import tempfile
import numpy as np
import torch
from . import runtime as core
from .wrappers.route_masks import make_env as make_mask_env, load_inventory

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mmdp-cash5-matplotlib")
)
SETTINGS = core.PPO_KWARGS


class PeriodicCheckpointCallback(BaseCallback):
    def __init__(self, save_path: Path, save_freq: int):
        super().__init__(0)
        self.save_path = save_path
        self.save_freq = int(save_freq)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(str(self.save_path / f"ckpt_{self.num_timesteps}.zip"))
        return True


class EpisodeAuditCallback(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.episodes = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                self.episodes.append(
                    {
                        "timesteps": int(self.num_timesteps),
                        "reward": float(info.get("episode", {}).get("r", float("nan"))),
                        "length": int(info.get("episode", {}).get("l", 0)),
                        "stage0_steps_used": int(
                            info.get(
                                "ep_stage0_steps_used", info.get("stage0_steps_used", 0)
                            )
                        ),
                        "stage1_steps_used": int(
                            info.get(
                                "ep_stage1_steps_used", info.get("stage1_steps_used", 0)
                            )
                        ),
                        "stop_count": int(
                            info.get(
                                "ep_total_stop_count", info.get("total_stop_count", 0)
                            )
                        ),
                        "zero_transfer_count": int(
                            info.get(
                                "ep_zero_transfer_count",
                                info.get("zero_transfer_count", 0),
                            )
                        ),
                    }
                )
        return True


def vec_env(interface: str, seed: int):
    return VecMonitor(DummyVecEnv([lambda: make_env(interface, seed)]))


def train_policy(interface: str, seed: int, timesteps: int, output_root: Path) -> dict:
    if interface not in INTERFACES:
        raise ValueError(interface)
    attempt = output_root
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "checkpoints").mkdir()
    (attempt / "best_model_not_used").mkdir()
    (attempt / "eval_logs_not_used_for_selection").mkdir()
    (attempt / "tb").mkdir()
    started = now_iso()
    write_json(
        attempt / "RUN_CONFIG.json",
        {
            "interface": interface,
            "training_seed": seed,
            "nominal_timesteps": timesteps,
            "policy": "MlpPolicy",
            "ppo_kwargs": PPO_KWARGS,
            "checkpoint_freq": CHECKPOINT_FREQ,
            "eval_freq": CHECKPOINT_FREQ,
            "eval_episodes": 10,
            "eval_seed_constructor": seed + 1000,
            "checkpoint_selection": "none; final_model after completed rollout only",
            "started_at": started,
            "system": system_inventory(),
        },
    )
    set_determinism(seed)
    train_env = vec_env(interface, seed)
    eval_env = vec_env(interface, seed + 1000)
    model = PPO(
        "MlpPolicy",
        env=train_env,
        tensorboard_log=str(attempt / "tb"),
        seed=seed,
        device="cpu",
        verbose=0,
        **PPO_KWARGS,
    )
    ep_cb = EpisodeAuditCallback()
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(attempt / "best_model_not_used"),
        log_path=str(attempt / "eval_logs_not_used_for_selection"),
        eval_freq=CHECKPOINT_FREQ,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0,
    )
    ckpt_cb = PeriodicCheckpointCallback(attempt / "checkpoints", CHECKPOINT_FREQ)
    t0 = time.perf_counter()
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[ep_cb, eval_cb, ckpt_cb],
            progress_bar=False,
        )
        final_path = attempt / "final_model.zip"
        model.save(str(final_path))
        elapsed = time.perf_counter() - t0
        meta = model_metadata(model)
        write_json(attempt / "training_episode_history.json", ep_cb.episodes)
        checkpoint_rows = []
        for p in sorted(
            (attempt / "checkpoints").glob("ckpt_*.zip"),
            key=lambda x: int(x.stem.split("_")[-1]),
        ):
            checkpoint_rows.append(
                {
                    "timesteps": int(p.stem.split("_")[-1]),
                    "path": str(p),
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                }
            )
        write_json(attempt / "checkpoint_inventory.json", checkpoint_rows)
        marker = {
            "status": "complete",
            "interface": interface,
            "training_seed": seed,
            "started_at": started,
            "completed_at": now_iso(),
            "wall_seconds": elapsed,
            "transitions_per_second": meta["num_timesteps"] / elapsed,
            "nominal_timesteps": timesteps,
            "actual_timesteps": meta["num_timesteps"],
            "model_metadata": meta,
            "final_model": str(final_path),
            "final_model_sha256": sha256_file(final_path),
            "checkpoint_count": len(checkpoint_rows),
            "checkpoint_500000_present": any(
                x["timesteps"] == 500000 for x in checkpoint_rows
            ),
            "final_checkpoint_policy": "final_model at complete rollout; intermediate best model forbidden for endpoint",
        }
        write_json(attempt / "TRAINING_COMPLETE.json", marker)
        return marker
    except Exception as e:
        write_json(
            attempt / "TRAINING_FAILED.json",
            {
                "status": "failed",
                "interface": interface,
                "training_seed": seed,
                "started_at": started,
                "failed_at": now_iso(),
                "exception_type": type(e).__name__,
                "exception": str(e),
                "attempt_preserved": True,
            },
        )
        raise
    finally:
        train_env.close()
        eval_env.close()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    core.write_json(Path(p), obj)


def sha(p):
    return core.sha256_file(Path(p))


def policy_hash(model):
    h = hashlib.sha256()
    for k, v in model.policy.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rng_hash():
    import pickle

    return hashlib.sha256(
        pickle.dumps(
            (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy())
        )
    ).hexdigest()


def mask_vec_env(setting, seed):
    return VecMonitor(DummyVecEnv([lambda: make_mask_env(setting, seed)]))


def build_mask_model(setting, seed, out):
    core.set_determinism(seed)
    env = mask_vec_env(setting, seed)
    evaluation = mask_vec_env(setting, seed + 1000)
    model = PPO(
        "MlpPolicy",
        env=env,
        tensorboard_log=str(Path(out) / "tb"),
        seed=seed,
        device="cpu",
        verbose=0,
        **SETTINGS,
    )
    return model, env, evaluation


class Progress(BaseCallback):
    def __init__(self, out):
        super().__init__(0)
        self.out = Path(out)
        self.started = time.perf_counter()

    def _on_step(self):
        if self.num_timesteps % 5000 == 0:
            write(
                self.out / "PROGRESS.json",
                dict(
                    actual_timesteps=self.num_timesteps,
                    wall_seconds=time.perf_counter() - self.started,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                ),
            )
        return True


def mask_callbacks(model, evaluation, out, extra=True):
    ep = EpisodeAuditCallback()
    ev = EvalCallback(
        eval_env=evaluation,
        best_model_save_path=str(out / "best_model_not_used"),
        log_path=str(out / "eval_logs_not_used_for_selection"),
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0,
    )
    ck = PeriodicCheckpointCallback(out / "checkpoints", 5000)
    cb = [ep, ev, ck]
    if extra:
        cb.append(Progress(out))
    return cb, ep


def train_mask(setting, seed, steps, out, extra=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for n in [
        "checkpoints",
        "best_model_not_used",
        "eval_logs_not_used_for_selection",
        "tb",
    ]:
        (out / n).mkdir()
    record = next(r for r in load_inventory()["settings"] if r["setting_id"] == setting)
    start = time.perf_counter()
    model = None
    env = None
    evaluation = None
    try:
        model, env, evaluation = build_mask_model(setting, seed, out)
        cfg = dict(
            setting_id=setting,
            mask_type=record["mask_type"],
            design_rng_seed=record["design_rng_seed"],
            stage0_indices=record["stage0_indices"],
            training_seed=seed,
            nominal_timesteps=steps,
            policy="MlpPolicy",
            ppo_kwargs=SETTINGS,
            model_metadata=core.model_metadata(model),
            initial_policy_sha256=policy_hash(model),
            initial_rng_sha256=rng_hash(),
            system=core.system_inventory(),
            eval_seed_constructor=seed + 1000,
            eval_freq=5000,
            eval_episodes=10,
            checkpoint_freq=5000,
            extra_progress_logging=extra,
            callback_order=[
                "EpisodeAuditCallback",
                "EvalCallback",
                "PeriodicCheckpointCallback",
            ]
            + (["Progress"] if extra else []),
            final_selection="complete rollout final_model only; never best_model",
            implicit_policy=dict(
                ortho_init=model.policy.ortho_init,
                normalize_images=model.policy.normalize_images,
                log_std_init=model.policy.log_std_init,
                use_sde=model.use_sde,
                sde_sample_freq=model.sde_sample_freq,
                normalize_advantage=model.normalize_advantage,
                target_kl=model.target_kl,
                clip_range_vf=model.clip_range_vf,
                squash_output=model.policy.squash_output,
                optimizer_defaults=model.policy.optimizer.defaults,
            ),
        )
        write(out / "RUN_CONFIG.json", cfg)
        cb, ep = mask_callbacks(model, evaluation, out, extra)
        model.learn(total_timesteps=steps, callback=cb, progress_bar=False)
        path = out / "final_model.zip"
        model.save(path)
        metadata = core.model_metadata(model)
        assert metadata["num_timesteps"] == int(np.ceil(steps / 256)) * 256
        assert (
            metadata["parameter_count"] == 40837
            and metadata["observation_shape"] == [28]
            and metadata["action_shape"] == [2]
        )
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        write(out / "training_episode_history.json", ep.episodes)
        checkpoints = [
            dict(
                timesteps=int(p.stem.split("_")[-1]),
                path=str(p),
                sha256=sha(p),
                bytes=p.stat().st_size,
            )
            for p in sorted(
                (out / "checkpoints").glob("ckpt_*.zip"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
        ]
        write(out / "checkpoint_inventory.json", checkpoints)
        marker = dict(
            status="complete",
            setting_id=setting,
            mask_type=record["mask_type"],
            design_rng_seed=record["design_rng_seed"],
            training_seed=seed,
            nominal_timesteps=steps,
            actual_timesteps=metadata["num_timesteps"],
            model_metadata=metadata,
            final_model=str(path),
            final_model_sha256=sha(path),
            final_policy_tensor_sha256=policy_hash(model),
            final_rng_sha256=rng_hash(),
            checkpoint_count=len(checkpoints),
            wall_seconds=time.perf_counter() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            output_bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
            training_episode_count=len(ep.episodes),
            completed_at=core.now_iso(),
        )
        write(out / "TRAINING_COMPLETE.json", marker)
        return marker
    except BaseException as e:
        write(
            out / "TRAINING_FAILED.json",
            dict(
                status="failed",
                setting_id=setting,
                training_seed=seed,
                error=repr(e),
                traceback=traceback.format_exc(),
                actual_timesteps=getattr(model, "num_timesteps", 0),
                attempt_preserved=True,
                wall_seconds=time.perf_counter() - start,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()
        if evaluation is not None:
            evaluation.close()
