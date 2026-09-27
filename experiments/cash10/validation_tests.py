"""Bounded tests of validation isolation using only fresh temporary models."""
import copy
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from stable_baselines3.common.callbacks import BaseCallback
from . import runtime
from . import validation


def assert_nested_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def training_state(model):
    base = model.get_env().unwrapped.envs[0].base_env
    return copy.deepcopy(dict(
        policy=model.policy.state_dict(), optimizer=model.policy.optimizer.state_dict(),
        last_observation=model._last_obs, episode_starts=model._last_episode_starts,
        environment=base.get_snapshot(), env_rng=base.rng.bit_generator.state,
        generator_rng=base.generator.rng.bit_generator.state,
        modules=[module.training for module in model.policy.modules()],
        rng=rng_state()))


class CaptureTrajectory(BaseCallback):
    def __init__(self):
        super().__init__()
        self.steps = []

    def _on_step(self):
        self.steps.append(copy.deepcopy(tuple(self.locals[key]
                                             for key in ("actions", "new_obs", "rewards", "dones"))))
        return True


class ValidationTests(unittest.TestCase):
    def test_absolute_grid_across_segments(self):
        with tempfile.TemporaryDirectory(prefix="cash10-validation-grid-") as directory:
            callback = validation.RawValidationCallback("flat", 42, directory)
            model = type("Model", (), {})()
            model.policy = torch.nn.Linear(1, 1)
            model.num_timesteps = 0
            callback.model = model
            with patch.object(validation, "evaluate_raw_episode", return_value={
                    "curve_return": 1., "native_return": 1., "episode_length": 2}) as score:
                for step in range(1, 1_500_161):
                    model.num_timesteps = step
                    callback._on_step()
                self.assertEqual(callback.points_recorded, 300)
                # Same callback resumes from the prefix's completed rollout.
                for step in range(1_500_161, 2_000_385):
                    model.num_timesteps = step
                    callback._on_step()
            self.assertEqual(callback.points_recorded, 400)
            self.assertEqual(score.call_count, 8000)
            with callback.csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["timesteps"]) for row in rows], list(range(5000, 2_000_001, 5000)))
            self.assertEqual(callback.environment_seeds, tuple(range(1042, 1062)))
            self.assertFalse(callback.metadata()["checkpoint_selection"])
            self.assertFalse(callback.metadata()["historical_callback_recovered"])
            with self.assertRaises(ValueError):
                validation.RawValidationCallback("flat", 42, directory, interval=256)

    def test_raw_score_and_precision(self):
        for policy in runtime.WRAPPERS:
            model = runtime.make_model(policy, 42)
            try:
                with validation.preserve_training_state(model):
                    actual = validation.evaluate_raw_episode(model, policy, 1042)
                    reference = runtime.evaluate_episode(model, policy, 1042)
                self.assertEqual(actual["native_return"], reference["episode_return"])
                self.assertEqual(actual["episode_length"], reference["episode_length"])
            finally:
                model.get_env().close()
        class Env:
            def __init__(self):
                self.resets = 1
                self.rewards = iter((100000000., 1., -100000000.))
                self.closed = False
                self.action_space = type("Space", (), {"contains": lambda _, a: True})()
            def reset(self):
                self.resets += 1
                return np.zeros(1), {}
            def step(self, action):
                reward = next(self.rewards)
                return np.zeros(1), reward, reward < 0, False, {}
            def close(self):
                self.closed = True
        env = Env()
        model = type("Model", (), {"predict": lambda _, o, deterministic: (0, None)})()
        with patch.object(runtime, "make_env", return_value=env):
            result = validation.evaluate_raw_episode(model, "flat", 1042)
        self.assertEqual(result["native_return"], 1.)
        self.assertEqual(result["curve_return"], 0.)
        self.assertEqual(env.resets, 2)
        self.assertTrue(env.closed)

    def test_bank_preserves_rng_modes_weights_optimizer_and_training_environment(self):
        with tempfile.TemporaryDirectory(prefix="cash10-validation-state-") as directory:
            model = runtime.make_model("flat", 42)
            try:
                # Populate optimizer state and live observation before checking.
                model.learn(512)
                model.policy.set_training_mode(True)
                model.policy.mlp_extractor.policy_net.eval()
                before = training_state(model)
                callback = validation.RawValidationCallback(
                    "flat", 42, directory, smoke_only=True, interval=512, final_timestep=512)
                callback.model = model
                callback._on_step()
                assert_nested_equal(self, before, training_state(model))
            finally:
                model.get_env().close()

    def test_short_training_trajectory_identical_with_and_without_validation(self):
        for policy in runtime.WRAPPERS:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory(prefix="cash10-validation-train-") as directory:
                captures, states = [], []
                for enabled in (False, True):
                    model = runtime.make_model(policy, 42)
                    capture = CaptureTrajectory()
                    callbacks = [capture]
                    if enabled:
                        callbacks.append(validation.RawValidationCallback(
                            policy, 42, directory, smoke_only=True, interval=256, final_timestep=1024))
                    evaluate = validation.evaluate_raw_episode
                    def noisy_evaluate(*args):
                        # Instrumentation must not perturb PPO even if it consumes
                        # global RNG, rather than relying on today's evaluator purity.
                        random.random()
                        np.random.random()
                        torch.rand(3)
                        return evaluate(*args)
                    try:
                        # Exercise reuse after an actual PPO update and learn restart.
                        with patch.object(validation, "evaluate_raw_episode", side_effect=noisy_evaluate):
                            model.learn(512, callback=callbacks)
                            model.learn(512, callback=callbacks, reset_num_timesteps=False)
                        captures.append(capture.steps)
                        states.append(training_state(model))
                    finally:
                        model.get_env().close()
                assert_nested_equal(self, captures[0], captures[1])
                assert_nested_equal(self, states[0], states[1])

    def test_rng_restored_on_failure(self):
        model = runtime.make_model("flat", 42)
        try:
            before = training_state(model)
            with self.assertRaisesRegex(RuntimeError, "intentional"), validation.preserve_training_state(model):
                random.random()
                np.random.random()
                torch.rand(3)
                model.policy.set_training_mode(False)
                raise RuntimeError("intentional")
            assert_nested_equal(self, before, training_state(model))
        finally:
            model.get_env().close()

    def test_smoke_cli_records_only_smoke_grid(self):
        with tempfile.TemporaryDirectory(prefix="cash10-validation-cli-") as directory:
            output = Path(directory) / "fresh"
            subprocess.run([sys.executable, "-B", "-m", "experiments.cash10.run", "train",
                            "--policy", "flat", "--seed", "42", "--smoke", "--output", str(output)],
                           check=True, capture_output=True, text=True)
            protocol = json.loads((output / "validation.json").read_text())
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(protocol, run["validation"])
            self.assertEqual(run["actual_transitions"], 512)
            self.assertEqual(protocol["points_recorded"], 2)
            self.assertTrue(protocol["smoke_only"])
            self.assertFalse(run["selected_by_validation"])
            with (output / "validation.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["timesteps"]) for row in rows], [256, 512])
            self.assertTrue(all(row["smoke_only"] == "True" for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
