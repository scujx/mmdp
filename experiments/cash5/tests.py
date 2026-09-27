"""Small isolated runtime checks; no paper checkpoints or full training."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.cash5"

import csv
import json
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Cash5Tests(unittest.TestCase):
    def test_random_repeatability_and_shared_instances(self):
        from .random_policy import episode

        fingerprint = None
        for setting in ["Flat", "True", "C1", "C2", "C3", "C4", "C5"]:
            actual, _ = episode(setting, 981000, 0)
            repeated, _ = episode(setting, 981000, 0)
            self.assertEqual(actual, repeated)
            self.assertLess(abs(actual["reconstruction_error"]), 1e-8)
            if fingerprint is None:
                fingerprint = actual["latent_fingerprint"]
            self.assertEqual(actual["latent_fingerprint"], fingerprint)

    def test_fresh_training_and_masks(self):
        from .evaluation import (
            initialize_worker,
            load_frozen_model,
            evaluate_episode,
            sha256_file,
        )

        initialize_worker()
        with tempfile.TemporaryDirectory(prefix="cash5-smoke-") as directory:
            root = Path(directory)
            for setting in ["Flat", "True", "C1"]:
                subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        str(HERE / "run.py"),
                        "train",
                        "--setting",
                        setting,
                        "--seed",
                        "42",
                        "--output",
                        str(root / setting),
                        "--smoke",
                    ],
                    check=True,
                    timeout=120,
                    stdout=subprocess.DEVNULL,
                )
                self.assertTrue((root / setting / "final_model.zip").is_file())
                self.assertTrue(
                    json.loads((root / setting / "REPRODUCTION_MODE.json").read_text())[
                        "smoke"
                    ]
                )
                checkpoint = root / setting / "final_model.zip"
                meta = dict(
                    setting_id=setting,
                    training_seed=42,
                    actual_steps=256,
                    checkpoint_path=str(checkpoint),
                    checkpoint_sha256=sha256_file(checkpoint),
                )
                model = load_frozen_model(meta, smoke=True)
                row, trace = evaluate_episode(
                    model, setting, 42, 981000, meta, smoke=True
                )
                self.assertTrue(trace)
                self.assertEqual(row["evaluation_role"], "smoke")
                with self.assertRaises(AssertionError):
                    evaluate_episode(model, setting, 42, 980000, meta, smoke=True)
                with self.assertRaises(AssertionError):
                    load_frozen_model(meta)
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "run.py"),
                    "generate-masks",
                    "--output-directory",
                    str(root / "masks"),
                ],
                check=True,
                timeout=30,
                stdout=subprocess.DEVNULL,
            )
            actual = json.loads((root / "masks/mask_inventory.json").read_text())
            expected = json.loads((HERE / "configs/mask_inventory.json").read_text())
            self.assertEqual(actual["settings"], expected["settings"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
