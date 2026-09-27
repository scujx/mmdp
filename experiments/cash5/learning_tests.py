"""Fixed20 offline curve checks using a newly trained 256-step fixture only.

No archived model is loaded and no formal experiment is run.
Run: python -B -m experiments.cash5.learning_tests
"""
import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from . import learning, runtime
from .envs.cash_flow_env import CashFlowEnv, CashFlowEnvConfig
from .wrappers.flat_cash_gym_env import FlatCashGymEnv
from .wrappers.mmdp_cash_gym_env import MMDPCashGymEnv
from .wrappers.route_masks import make_env


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class ScheduledFixture(BaseCallback):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory

    def _on_step(self):
        if self.num_timesteps in (125, 250):
            self.model.save(self.directory / f"ckpt_{self.num_timesteps}.zip")
        return True


class RewardEnvironment:
    def __init__(self, rewards):
        self.rewards = rewards
        self.reset_ordinal, self.index = 1, 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self):
        self.reset_ordinal += 1
        return np.zeros(28, dtype=np.float32), {"canonical_reset_ordinal": self.reset_ordinal}

    def step(self, action):
        value = self.rewards[self.index]
        self.index += 1
        return np.zeros(28, dtype=np.float32), value, self.index == len(self.rewards), False, {}

    def close(self):
        pass


class RewardPolicy:
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def predict(self, observation, deterministic=False):
        assert deterministic
        return np.zeros(2, dtype=np.float32), None


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix="cash5-learning-test-")
        cls.root = Path(cls.temp.name).resolve()
        cls.run_dir = cls.root / "new-short-training-fixture"
        checkpoints = cls.run_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        env = DummyVecEnv([lambda: make_env("Flat", 42)])
        model = PPO("MlpPolicy", env, seed=42, device="cpu", verbose=0, **runtime.PPO_KWARGS)
        try:
            model.learn(256, callback=ScheduledFixture(checkpoints))
            assert model.num_timesteps == 256
            model.save(cls.run_dir / "final_model.zip")
        finally:
            env.close()
        inventory = [dict(timesteps=step, path=str(checkpoints / f"ckpt_{step}.zip"),
                          sha256=learning.sha256(checkpoints / f"ckpt_{step}.zip"),
                          bytes=(checkpoints / f"ckpt_{step}.zip").stat().st_size) for step in (125, 250)]
        write_json(cls.run_dir / "checkpoint_inventory.json", inventory)
        write_json(cls.run_dir / "RUN_CONFIG.json", dict(interface="flat", training_seed=42, nominal_timesteps=256))
        write_json(cls.run_dir / "TRAINING_COMPLETE.json", dict(status="complete", interface="flat", training_seed=42,
                                                            nominal_timesteps=256, actual_timesteps=256, checkpoint_count=2))
        cls.digest = learning.sha256(cls.run_dir / "checkpoint_inventory.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def inspect(self, run=None, digest=None, **kwargs):
        return learning.inspect_run(run or self.run_dir, "Flat", 42, digest or self.digest,
                                    trusted=True, smoke=True, **kwargs)

    def clone_run(self, name):
        target = self.root / name
        shutil.copytree(self.run_dir, target)
        return target

    def test_fresh_offline_end_to_end_both_conventions_and_unchanged_inputs(self):
        before = {str(path.relative_to(self.run_dir)): learning.sha256(path) for path in self.run_dir.rglob("*") if path.is_file()}
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        for convention in learning.CONVENTIONS:
            output = self.root / f"curve-{convention}"
            metadata = learning.evaluate_run(self.run_dir, "Flat", 42, self.digest, output,
                                             reward_convention=convention, trusted=True, smoke=True)
            self.assertEqual(metadata["points_recorded"], 2)
            self.assertEqual(metadata["episode_rows"], 40)
            self.assertEqual(metadata["environment_seeds"], list(range(1042, 1062)))
            self.assertEqual(metadata["last_scheduled_step"], 250)
            self.assertEqual(metadata["training_endpoint_actual_steps"], 256)
            self.assertTrue(metadata["smoke_only"])
            self.assertFalse(metadata["historical_results_recovered"])
            self.assertFalse(metadata["checkpoint_selection"])
            self.assertTrue((output / "learning_curve_metadata.json").is_file())
            with (output / "learning_curve_points.csv").open() as handle:
                points = list(csv.DictReader(handle))
            with (output / "learning_curve_episodes.csv").open() as handle:
                episodes = list(csv.DictReader(handle))
            self.assertEqual([int(row["timesteps"]) for row in points], [125, 250])
            self.assertEqual([int(row["loaded_actual_steps"]) for row in points], [125, 250])
            self.assertTrue(all(row["reset_ordinal"] == "2" for row in episodes))
            for point in points:
                chosen = [float(row["episode_return"]) for row in episodes if row["timesteps"] == point["timesteps"]]
                self.assertEqual(len(chosen), 20)
                self.assertAlmostEqual(float(point["mean_return"]), float(np.mean(chosen)), places=12)
            self.assertEqual(random.getstate(), python_before)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
            self.assertEqual(np.random.get_state()[2:], numpy_before[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
        after = {str(path.relative_to(self.run_dir)): learning.sha256(path) for path in self.run_dir.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_float32_transport_is_not_float32_episode_accumulation(self):
        rewards = (100_000_000.0, 1.0, -100_000_000.0)
        with mock.patch.object(learning, "make_env", side_effect=lambda setting, seed: RewardEnvironment(rewards)), mock.patch.object(learning, "exogenous_payload", return_value={"seed": 1042}):
            row = learning.evaluate_episode(RewardPolicy(), "Flat", 1042, "vec-float32")
        self.assertEqual(row["episode_return"], 1.0)
        self.assertEqual(row["native_return"], 1.0)
        accumulator = np.float32(0)
        for reward in rewards:
            accumulator = np.float32(accumulator + np.float32(reward))
        self.assertEqual(float(accumulator), 0.0)

    def test_native_and_vec_transport_are_explicitly_distinct(self):
        rewards = (.1, .2, -.1)
        with mock.patch.object(learning, "make_env", side_effect=lambda setting, seed: RewardEnvironment(rewards)), mock.patch.object(learning, "exogenous_payload", return_value={}):
            row = learning.evaluate_episode(RewardPolicy(), "Flat", 1042, "native")
        self.assertEqual(row["episode_return"], sum(rewards))
        self.assertEqual(row["vec_float32_return"], sum(float(np.float32(value)) for value in rewards))
        self.assertNotEqual(row["native_return"], row["vec_float32_return"])

    def test_shared_wrapper_reset_convention(self):
        for wrapper in (FlatCashGymEnv, MMDPCashGymEnv):
            base = CashFlowEnv(CashFlowEnvConfig(seed=1042))
            original_reset = base.reset
            with mock.patch.object(base, "reset", wraps=original_reset) as resets:
                env = wrapper(base)
                try:
                    self.assertEqual(resets.call_count, 1)
                    env.reset()
                    self.assertEqual(resets.call_count, 2)
                finally:
                    env.close()

    def test_trust_and_inventory_hash_required_before_loading(self):
        with mock.patch.object(learning, "load_checkpoint") as load:
            for trusted, digest in ((False, self.digest), (True, "0" * 64)):
                with self.assertRaises(ValueError):
                    learning.evaluate_run(self.run_dir, "Flat", 42, digest, self.root / "rejected-preflight",
                                          reward_convention="native", trusted=trusted, smoke=True)
            load.assert_not_called()
        self.assertFalse((self.root / "rejected-preflight").exists())

    def test_checkpoint_tampering_rejected_before_loading(self):
        run = self.clone_run("tampered-checkpoint")
        with (run / "checkpoints/ckpt_125.zip").open("ab") as handle:
            handle.write(b"tampered-fixture")
        with mock.patch.object(learning, "load_checkpoint") as load, self.assertRaisesRegex(ValueError, "Checkpoint SHA-256"):
            self.inspect(run)
        load.assert_not_called()

    def test_duplicate_missing_or_unordered_schedule_rejected(self):
        run = self.clone_run("duplicate-inventory")
        inventory = json.loads((run / "checkpoint_inventory.json").read_text())
        inventory.append(inventory[0])
        write_json(run / "checkpoint_inventory.json", inventory)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.inspect(run, learning.sha256(run / "checkpoint_inventory.json"))
        for steps in ([126], [250, 125], [125, 125], [0]):
            with self.assertRaises(ValueError):
                self.inspect(steps=steps)

    def test_loaded_actual_timestep_and_seed_are_verified(self):
        row = self.inspect()["checkpoints"][0]
        with self.assertRaisesRegex(ValueError, "timestep/training seed"):
            learning.load_checkpoint(row, 43)
        changed = dict(row, timesteps=124)
        with self.assertRaisesRegex(ValueError, "timestep/training seed"):
            learning.load_checkpoint(changed, 42)

    def test_short_fixture_is_not_accepted_as_formal_curve(self):
        with self.assertRaisesRegex(ValueError, "Formal curves require"):
            learning.inspect_run(self.run_dir, "Flat", 42, self.digest, trusted=True)
        self.assertEqual(learning.SCHEDULE, tuple(range(5000, 500001, 5000)))

    def test_formal_preflight_requires_all_100_points_without_loading_models(self):
        # These are checksum fixtures, explicitly not model archives. Only the
        # inventory preflight is exercised; no formal result is produced.
        run = self.root / "formal-schedule-preflight-only"
        checkpoints = run / "checkpoints"
        checkpoints.mkdir(parents=True)
        rows = []
        for step in learning.SCHEDULE:
            path = checkpoints / f"ckpt_{step}.zip"
            path.write_bytes(b"PRECHECK_ONLY_NOT_A_MODEL")
            rows.append(dict(timesteps=step, path=str(path), sha256=learning.sha256(path), bytes=path.stat().st_size))
        write_json(run / "checkpoint_inventory.json", rows)
        write_json(run / "RUN_CONFIG.json", dict(interface="flat", training_seed=42, nominal_timesteps=500000))
        complete = dict(status="complete", interface="flat", training_seed=42, nominal_timesteps=500000,
                        actual_timesteps=500224, checkpoint_count=100)
        write_json(run / "TRAINING_COMPLETE.json", complete)
        with mock.patch.object(learning, "load_checkpoint") as load:
            plan = learning.inspect_run(run, "Flat", 42, learning.sha256(run / "checkpoint_inventory.json"), trusted=True)
            self.assertEqual(plan["scheduled_steps"], list(learning.SCHEDULE))
            self.assertEqual(len(plan["checkpoints"]), 100)
            self.assertEqual(plan["scheduled_steps"][-1], 500000)
            self.assertEqual(plan["training_endpoint_actual_steps"], 500224)
            write_json(run / "checkpoint_inventory.json", rows[:-1])
            complete["checkpoint_count"] = 99
            write_json(run / "TRAINING_COMPLETE.json", complete)
            with self.assertRaisesRegex(ValueError, "missing"):
                learning.inspect_run(run, "Flat", 42, learning.sha256(run / "checkpoint_inventory.json"), trusted=True)
            load.assert_not_called()
        self.assertFalse((run / "learning_curve_metadata.json").exists())

    def test_existing_source_or_training_run_output_rejected(self):
        for output in (self.run_dir, self.run_dir / "curve", learning.SOURCE_ROOT / "curve"):
            with self.assertRaises(ValueError):
                learning.evaluate_run(self.run_dir, "Flat", 42, self.digest, output,
                                      reward_convention="native", trusted=True, smoke=True)

    def test_failure_keeps_partial_files_and_restores_rng(self):
        output = self.root / "failure-output"
        random.seed(300)
        torch.manual_seed(300)
        before_python, before_torch = random.getstate(), torch.get_rng_state().clone()
        with mock.patch.object(learning, "evaluate_episode", side_effect=RuntimeError("fixture-only-failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture-only-failure"):
                learning.evaluate_run(self.run_dir, "Flat", 42, self.digest, output,
                                      reward_convention="native", trusted=True, smoke=True)
        self.assertTrue((output / "FAILED.json").exists())
        self.assertTrue((output / "learning_curve_points.csv").exists())
        self.assertEqual(random.getstate(), before_python)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))

    def test_cli_evaluates_only_new_short_fixture(self):
        output = self.root / "cli-smoke"
        result = subprocess.run([sys.executable, "-B", "-m", "experiments.cash5.run", "evaluate-learning",
                                 "--run", str(self.run_dir), "--setting", "Flat", "--seed", "42",
                                 "--inventory-sha256", self.digest, "--reward-convention", "native",
                                 "--trust-checkpoint", "--smoke", "--steps", "125", "250", "--output", str(output)],
                                cwd=learning.SOURCE_ROOT, text=True, capture_output=True, check=True)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["episode_rows"], 40)
        self.assertEqual(summary["last_scheduled_step"], 250)
        self.assertEqual(summary["training_endpoint_actual_steps"], 256)
        self.assertTrue(summary["smoke_only"])


if __name__ == "__main__":
    unittest.main()
