"""Shared PPO configuration, environment factories, and run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .envs.cash_flow_env import CashFlowEnv, CashFlowEnvConfig  # noqa: E402
from .wrappers.flat_cash_gym_env import FlatCashGymEnv  # noqa: E402
from .wrappers.mmdp_cash_gym_env import MMDPCashGymEnv  # noqa: E402

INTERFACES = ("flat", "mmdp")
TRAIN_SEEDS = tuple(range(42, 52))
NOMINAL_STEPS = 500_000
ACTUAL_STEPS = 500_224
CHECKPOINT_FREQ = 5_000

PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def make_env(interface: str, seed: int):
    base = CashFlowEnv(CashFlowEnvConfig(seed=int(seed)))
    if interface == "flat":
        return FlatCashGymEnv(base)
    if interface == "mmdp":
        return MMDPCashGymEnv(base)
    raise ValueError(interface)


def set_determinism(seed: int) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def system_inventory() -> dict[str, Any]:
    import cloudpickle
    import gymnasium
    import stable_baselines3

    return {
        "python_executable": "python",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "cloudpickle": cloudpickle.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def policy_parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.policy.parameters()))


def model_metadata(model: Any) -> dict[str, Any]:
    return {
        "seed": int(model.seed),
        "num_timesteps": int(model.num_timesteps),
        "_total_timesteps": int(model._total_timesteps),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "clip_range_at_1": float(model.clip_range(1.0)),
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "learning_rate": float(model.learning_rate),
        "n_envs": int(model.n_envs),
        "observation_shape": list(model.observation_space.shape),
        "action_shape": list(model.action_space.shape),
        "parameter_count": policy_parameter_count(model),
        "optimizer": type(model.policy.optimizer).__name__,
        "activation": model.policy.activation_fn.__name__,
        "net_arch": to_jsonable(model.policy.net_arch),
    }


def now_iso() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()
