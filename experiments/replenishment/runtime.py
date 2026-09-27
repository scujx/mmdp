"""Single-thread runtime, policy factory and independent batched evaluation."""

import os

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "1"
from pathlib import Path
import json, hashlib, random, contextlib
import numpy as np
import torch
from stable_baselines3 import PPO
from .wrappers.training import Replenishment, FixedRewardScale


ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
CELLS = ["flat_retained", "flat_updated", "mmdp_retained", "mmdp_updated"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, x):
    with Path(p).open("x") as f:
        json.dump(x, f, indent=2)


def policy_hash(model):
    h = hashlib.sha256()
    for k, v in model.policy.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def make_model(env, seed):
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
            "activation_fn": torch.nn.Tanh,
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )


@contextlib.contextmanager
def preserve_rng():
    a = random.getstate()
    b = np.random.get_state()
    c = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(a)
        np.random.set_state(b)
        torch.set_rng_state(c)


def evaluate(model, cell, split, ids, trace=False, fine_accuracy=0.80):
    interface, information = cell.split("_")
    with preserve_rng():
        envs = [Replenishment(interface, information, split, i, fine_accuracy=fine_accuracy) for i in ids]
        obs = np.stack([e.reset()[0] for e in envs])
        traces = [[] for e in envs]
        for step in range(26):
            acts, _ = model.predict(obs, deterministic=True)
            out = []
            for j, (e, a) in enumerate(zip(envs, acts)):
                o, r, done, _, inf = e.step(a)
                out.append(o)
                assert done == (step == 25)
                if trace:
                    traces[j].append(dict(action=a.tolist(), reward=r, **inf))
            obs = np.stack(out)
        rows = []
        for j, e in enumerate(envs):
            reconstructed = e.totals["revenue"] - sum(
                v for k, v in e.totals.items() if k != "revenue"
            )
            assert abs(reconstructed - e.return_sum) < 1e-8 and e.length == 26
            row = {
                "instance_id": ids[j],
                "split": split,
                "ordinal": 0,
                "fingerprint": e.fingerprint,
                "return": e.return_sum,
                "length": 26,
                **e.totals,
                "terminal_inventory": e.inventory,
            }
            if trace:
                row["trace"] = traces[j]
            rows.append(row)
        return rows
