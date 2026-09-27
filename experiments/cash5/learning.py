"""Fresh offline fixed-bank learning curves from trusted scheduled checkpoints.

Training and its original continuing ten-episode EvalCallback are unchanged.
This evaluates saved Raw policies; it neither selects nor updates a checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from . import runtime
from .wrappers.route_masks import exogenous_payload, make_env, to_jsonable

SETTINGS = ("Flat", "True", "U1", "U2", "U3", "U4", "U5", "C1", "C2", "C3", "C4", "C5")
SCHEDULE = tuple(range(5_000, 500_001, 5_000))
SOURCE_ROOT = Path(__file__).resolve().parents[2]
CONVENTIONS = ("native", "vec-float32")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError("Require a complete hexadecimal SHA-256")
    return value.lower()


@contextmanager
def preserve_rng():
    """PPO.load may reseed globals; offline scoring restores its caller's state."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def inspect_run(run, setting, training_seed, inventory_sha256, *, trusted=False, smoke=False, steps=None):
    """Check the complete schedule and hashes before any deserialization.

    Inventory paths are resolved to this run's checkpoints directory, including
    inventories relocated from another machine. No external path is followed.
    """
    if not trusted:
        raise ValueError("Offline learning evaluation requires --trust-checkpoint")
    if setting not in SETTINGS or type(training_seed) is not int:
        raise ValueError("Require a valid setting and integer training seed")
    if not smoke and training_seed not in range(42, 52):
        raise ValueError("Formal curves use training seeds 42-51")
    run = Path(run).resolve(strict=True)
    inventory = run / "checkpoint_inventory.json"
    expected_digest = require_digest(inventory_sha256)
    if sha256(inventory) != expected_digest:
        raise ValueError("Checkpoint inventory SHA-256 mismatch")
    config_path, complete_path = run / "RUN_CONFIG.json", run / "TRAINING_COMPLETE.json"
    config, complete = (json.loads(path.read_text()) for path in (config_path, complete_path))
    expected_interface = {"Flat": "flat", "True": "mmdp"}.get(setting)
    for name, record in (("RUN_CONFIG", config), ("TRAINING_COMPLETE", complete)):
        identity = record.get("setting_id")
        if identity is None and expected_interface is not None and record.get("interface") == expected_interface:
            identity = setting
        if identity != setting or record.get("training_seed") != training_seed:
            raise ValueError(f"{name} setting/training seed mismatch")
    if complete.get("status") != "complete":
        raise ValueError("Training completion marker is not complete")
    endpoint, nominal = complete.get("actual_timesteps"), config.get("nominal_timesteps")
    if type(endpoint) is not int or type(nominal) is not int or endpoint < 1 or nominal < 1:
        raise ValueError("Missing actual/nominal training timestep metadata")
    if complete.get("nominal_timesteps") != nominal:
        raise ValueError("Training markers disagree on nominal timesteps")
    if not smoke and (nominal, endpoint) != (500_000, 500_224):
        raise ValueError("Formal curves require nominal 500000 and final 500224; use --smoke for fixtures")
    if steps is not None and not smoke:
        raise ValueError("Custom checkpoint schedules are smoke-only")
    rows = json.loads(inventory.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Checkpoint inventory must be a nonempty list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("timesteps")) is not int or row["timesteps"] < 1:
            raise ValueError("Invalid inventory checkpoint timestep")
        step = row["timesteps"]
        if step in indexed:
            raise ValueError("Duplicate inventory checkpoint timestep")
        path = row.get("path")
        if not isinstance(path, str) or Path(path).name != f"ckpt_{step}.zip":
            raise ValueError("Inventory checkpoint filename/timestep mismatch")
        checkpoint = (run / "checkpoints" / Path(path).name).resolve(strict=True)
        if not checkpoint.is_relative_to(run) or not checkpoint.is_file():
            raise ValueError("Checkpoint must be a file within the supplied run")
        digest = require_digest(row.get("sha256"))
        if sha256(checkpoint) != digest:
            raise ValueError(f"Checkpoint SHA-256 mismatch at {step}")
        if type(row.get("bytes")) is not int or checkpoint.stat().st_size != row["bytes"]:
            raise ValueError("Checkpoint byte count mismatch")
        indexed[step] = dict(timesteps=step, path=str(checkpoint), sha256=digest)
    selected = tuple(sorted(indexed)) if smoke and steps is None else (tuple(steps) if steps is not None else SCHEDULE)
    if not selected or any(type(step) is not int or step < 1 for step in selected) or tuple(sorted(set(selected))) != selected:
        raise ValueError("Checkpoint steps must be positive, unique, and increasing")
    if any(step > endpoint or step not in indexed for step in selected):
        raise ValueError("Scheduled checkpoint missing or beyond the actual training endpoint")
    if not smoke and set(indexed) != set(SCHEDULE):
        raise ValueError("Formal inventory must contain exactly the 100 prescribed 5K checkpoints")
    if complete.get("checkpoint_count") != len(indexed):
        raise ValueError("Completion marker checkpoint count disagrees with inventory")
    return dict(setting_id=setting, training_seed=training_seed, run=str(run),
                checkpoint_inventory_sha256=expected_digest, nominal_training_steps=nominal,
                training_endpoint_actual_steps=endpoint, scheduled_steps=list(selected),
                checkpoints=[indexed[step] for step in selected],
                protected_inputs={str(path): sha256(path) for path in (inventory, config_path, complete_path)})


def load_checkpoint(row, training_seed):
    """Load an inventory-verified scheduled model, never a selected/final substitute."""
    from stable_baselines3 import PPO
    if sha256(row["path"]) != row["sha256"]:
        raise ValueError("Checkpoint changed after inventory verification")
    model = PPO.load(row["path"], device="cpu")
    if model.num_timesteps != row["timesteps"] or model.seed != training_seed:
        raise ValueError("Loaded checkpoint actual timestep/training seed mismatch")
    if model.observation_space.shape != (28,) or model.action_space.shape != (2,) or sum(p.numel() for p in model.policy.parameters()) != 40837:
        raise ValueError("Expected the shared raw five-account PPO model")
    if (not np.array_equal(model.action_space.low, np.array([-1, -1], dtype=np.float32))
            or not np.array_equal(model.action_space.high, np.array([1, 1], dtype=np.float32))):
        raise ValueError("Changed raw action bounds")
    model.policy.set_training_mode(False)
    return model


def evaluate_episode(model, setting, environment_seed, reward_convention):
    if reward_convention not in CONVENTIONS:
        raise ValueError("Unknown reward convention")
    env = make_env(setting, environment_seed)
    try:
        if env.reset_ordinal != 1:
            raise RuntimeError("Wrapper constructor must perform reset one")
        observation, info = env.reset()
        if env.reset_ordinal != 2 or info["canonical_reset_ordinal"] != 2:
            raise RuntimeError("Learning curves require explicit reset two")
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError("Checkpoint spaces do not match the requested interface")
        initial_hash = canonical_hash(observation)
        latent_hash = canonical_hash(exogenous_payload(env))
        native_return, transported_return = 0.0, 0.0
        actions = []
        for length in range(1, 23):
            action, _ = model.predict(observation, deterministic=True)
            if not env.action_space.contains(action):
                raise RuntimeError("Raw policy action is outside the declared space")
            actions.append(to_jsonable(action))
            observation, reward, done, truncated, _ = env.step(action)
            if not math.isfinite(float(reward)):
                raise RuntimeError("Non-finite curve reward")
            native_return += float(reward)
            # Emulate DummyVecEnv's per-step float32 reward transport;
            # the accumulator itself remains a Python float.
            transported_return += float(np.float32(reward))
            if done or truncated:
                if truncated or not done:
                    raise RuntimeError("Learning curves require a complete terminated episode")
                chosen = native_return if reward_convention == "native" else transported_return
                return dict(environment_seed=environment_seed, reset_ordinal=2,
                            episode_return=chosen, native_return=native_return,
                            vec_float32_return=transported_return, episode_length=length,
                            initial_observation_sha256=initial_hash, latent_fingerprint=latent_hash,
                            actions_sha256=canonical_hash(actions))
        raise RuntimeError("Learning episode exceeded its documented decision limit")
    finally:
        env.close()


def validate_output(output):
    output = Path(output).resolve()
    if output.exists() or output == SOURCE_ROOT or output.is_relative_to(SOURCE_ROOT):
        raise ValueError("Use a new output directory outside the source package")
    return output


def evaluate_run(run, setting, training_seed, inventory_sha256, output, *, reward_convention,
                 trusted=False, smoke=False, steps=None):
    if reward_convention not in CONVENTIONS:
        raise ValueError("Choose --reward-convention native or vec-float32 explicitly")
    output = validate_output(output)
    plan = inspect_run(run, setting, training_seed, inventory_sha256,
                       trusted=trusted, smoke=smoke, steps=steps)
    if output == Path(plan["run"]) or output.is_relative_to(Path(plan["run"])):
        raise ValueError("Learning outputs must be separate from the immutable training run")
    environment_seeds = [training_seed + 1000 + j for j in range(20)]
    output.mkdir(parents=True, exist_ok=False)
    point_fields = ("setting_id", "training_seed", "timesteps", "loaded_actual_steps", "training_endpoint_actual_steps",
                    "checkpoint_sha256", "reward_convention", "mean_return", "return_sd", "native_mean_return",
                    "vec_float32_mean_return", "mean_episode_length", "episodes", "smoke_only")
    episode_fields = ("setting_id", "training_seed", "timesteps", "loaded_actual_steps", "checkpoint_sha256",
                      "reward_convention", "episode_index", "environment_seed", "reset_ordinal", "episode_return",
                      "native_return", "vec_float32_return", "episode_length", "initial_observation_sha256",
                      "latent_fingerprint", "actions_sha256", "smoke_only")
    points = []
    try:
        with preserve_rng(), (output / "learning_curve_points.csv").open("x", newline="") as point_file, (output / "learning_curve_episodes.csv").open("x", newline="") as episode_file:
            point_writer = csv.DictWriter(point_file, fieldnames=point_fields)
            episode_writer = csv.DictWriter(episode_file, fieldnames=episode_fields)
            point_writer.writeheader()
            episode_writer.writeheader()
            for row in plan["checkpoints"]:
                model = load_checkpoint(row, training_seed)
                episodes = [evaluate_episode(model, setting, seed, reward_convention) for seed in environment_seeds]
                identity = dict(setting_id=setting, training_seed=training_seed, timesteps=row["timesteps"],
                                loaded_actual_steps=int(model.num_timesteps), checkpoint_sha256=row["sha256"],
                                reward_convention=reward_convention, smoke_only=smoke)
                for index, episode in enumerate(episodes):
                    episode_writer.writerow(dict(**identity, episode_index=index, **episode))
                returns = np.asarray([episode["episode_return"] for episode in episodes], dtype=np.float64)
                point = dict(**identity, training_endpoint_actual_steps=plan["training_endpoint_actual_steps"],
                             mean_return=float(returns.mean()), return_sd=float(returns.std(ddof=0)),
                             native_mean_return=float(np.mean([episode["native_return"] for episode in episodes])),
                             vec_float32_mean_return=float(np.mean([episode["vec_float32_return"] for episode in episodes])),
                             mean_episode_length=float(np.mean([episode["episode_length"] for episode in episodes])), episodes=20)
                point_writer.writerow(point)
                point_file.flush()
                episode_file.flush()
                points.append(point)
                if sha256(row["path"]) != row["sha256"]:
                    raise RuntimeError("Scheduled checkpoint changed during evaluation")
                del model
        if any(sha256(path) != digest for path, digest in plan["protected_inputs"].items()):
            raise RuntimeError("Training metadata changed during offline evaluation")
        metadata = dict(provenance="fresh_offline_fixed20_evaluation", historical_results_recovered=False,
                        historical_value_parity="not_checked", checkpoint_selection=False, policy_updates=False,
                        deterministic=True, execution="Raw", search=False, reset_ordinal=2, smoke_only=smoke,
                        reward_convention=reward_convention, reward_accumulation="Python float step sum; float64 bank mean",
                        selected_reward_transport="native" if reward_convention == "native" else "float32 per step",
                        native_and_vec_float32_returns_recorded=True, within_bank_sd_ddof=0,
                        environment_seeds=environment_seeds, expected_points=len(plan["scheduled_steps"]),
                        points_recorded=len(points), episode_rows=len(points) * 20,
                        last_scheduled_step=plan["scheduled_steps"][-1], checkpoint_inventory_sha256=plan["checkpoint_inventory_sha256"],
                        nominal_training_steps=plan["nominal_training_steps"], training_endpoint_actual_steps=plan["training_endpoint_actual_steps"],
                        setting_id=setting, training_seed=training_seed, scheduled_steps=plan["scheduled_steps"],
                        scheduled_checkpoints=plan["checkpoints"],
                        reward_conventions={"native": "Sum native step rewards as Python floats",
                                            "vec-float32": "Convert each step reward to float32, then sum as Python floats; not a float32 accumulator"},
                        code_sha256=sha256(__file__),
                        files={name: sha256(output / name) for name in ("learning_curve_points.csv", "learning_curve_episodes.csv")},
                        note="Fresh checkpoint scoring; retains the original ten-episode training callback. Scheduled curve points are distinct from the final updated endpoint.")
        import stable_baselines3
        metadata["versions"] = dict(numpy=np.__version__, torch=torch.__version__, stable_baselines3=stable_baselines3.__version__)
        with (output / "learning_curve_metadata.json").open("x") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return metadata
    except Exception as exc:
        with (output / "FAILED.json").open("x") as handle:
            json.dump(dict(status="failed", error_type=type(exc).__name__, error=str(exc),
                           points_recorded=len(points), partial_files_retained=True), handle, indent=2)
            handle.write("\n")
        raise
