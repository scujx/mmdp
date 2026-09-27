"""Bounded installation tests using temporary models and development instances."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from . import runtime as r


class Cash10Tests(unittest.TestCase):
    def test_configuration(self):
        actual = json.loads(json.dumps(dataclasses.asdict(r.CashFlowEnvConfigV2(seed=42))))
        self.assertEqual(actual, r.CONFIG["environment"])

    def test_matched_initial_observations_and_spaces(self):
        observations = []
        for name in r.WRAPPERS:
            env = r.make_env(name, 880000)
            try:
                observation, _ = env.reset()
                observations.append(observation)
                self.assertEqual(observation.shape, (53,))
                if name.endswith("-aa"):
                    self.assertEqual(env.action_space.n, 75 if name == "flat-aa" else 59)
                else:
                    self.assertEqual(env.action_space.shape, (2,))
            finally:
                env.close()
        for observation in observations[1:]:
            np.testing.assert_array_equal(observation, observations[0])

    def test_zero_transfer_is_not_stop(self):
        for name in ("flat-aa", "mmdp-aa"):
            env = r.make_env(name, 880000)
            try:
                env.reset()
                env.base_env.balances[:] = 0.
                stage = env.base_env.stage
                _, reward, done, _, info = env.step(0)
                self.assertFalse(info["stop_action"])
                self.assertFalse(info["wrapper_stop_selected"])
                self.assertFalse(done)
                self.assertEqual(env.base_env.stage, stage)
                self.assertEqual(reward, -12.)
            finally:
                env.close()

    def test_raw_search_repeatability_and_compute(self):
        for name in r.WRAPPERS:
            model = r.make_model(name, 42)
            weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
            try:
                raw = r.evaluate_episode(model, name, 880000)
                for cap in (576, 1152, 2304):
                    episode = r.evaluate_episode(model, name, 880000, cap)
                    self.assertEqual(episode, r.evaluate_episode(model, name, 880000, cap))
                    self.assertEqual(raw["initial_observation_sha256"], episode["initial_observation_sha256"])
                    self.assertLessEqual(episode["max_planning_transitions_per_decision"], cap)
                    self.assertLess(episode["reward_reconstruction_error"], 1e-8)
                for key, value in model.policy.state_dict().items():
                    self.assertTrue(torch.equal(value, weights[key]))
            finally:
                model.get_env().close()

    def test_cli_training_loading_evaluation_and_guards(self):
        with tempfile.TemporaryDirectory(prefix="cash10-smoke-") as directory:
            for name in r.WRAPPERS:
                output = Path(directory) / name
                command = [sys.executable, "-B", "-m", "experiments.cash10.run", "train", "--policy", name,
                           "--seed", "42", "--smoke", "--output", str(output)]
                subprocess.run(command, check=True, capture_output=True, text=True)
                metadata = json.loads((output / "run.json").read_text())
                self.assertEqual(metadata["actual_transitions"], 512)
                self.assertTrue(metadata["smoke_only"])
                self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
                evaluate = [sys.executable, "-B", "-m", "experiments.cash10.run", "evaluate", "--policy", name,
                            "--checkpoint", str(output / "final.zip"), "--sha256", metadata["checkpoint_sha256"],
                            "--trust-checkpoint", "--caps", "0", "576", "--instances", "1",
                            "--seed-start", "880000", "--output", str(output / "evaluation")]
                subprocess.run(evaluate, check=True, capture_output=True, text=True)
                self.assertTrue((output / "evaluation/episodes.csv").is_file())
                bad = evaluate.copy()
                bad[bad.index("--sha256") + 1] = "0" * 64
                bad[-1] = str(output / "invalid")
                self.assertNotEqual(subprocess.run(bad, capture_output=True).returncode, 0)
                self.assertFalse((output / "invalid").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
