"""Timing protocol checks with fake clocks/controllers; no policy deserialization.

Run: python -B -m experiments.controller_timing_tests
"""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from . import controller_timing as timing


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance(self, ns):
        self.now += ns


class Environment:
    def __init__(self, clock, length=2):
        self.clock, self.length = clock, length
        self.stage, self.step_in_stage = 0, 0

    def get_observation_dict(self):
        return dict(stage=self.stage, step_in_stage=self.step_in_stage, balances=[1, 2])

    def step(self, action):
        self.clock.advance(200)
        self.step_in_stage += 1
        return None, -1.0, self.step_in_stage == self.length, {}


def block(name="a", interface="Flat", controller="milp", benchmark="cash5", replicate=0):
    return timing.validate_block(dict(id=name, benchmark=benchmark, controller=controller,
                                      interface=interface, random_replicate=replicate))


def fake_episode(values=(1.0, 2.0), multiplier=1):
    decisions = [dict(decision_ordinal=i + 1, stage=0, step_in_stage=i,
                      controller_ms=value * multiplier, environment_ms=.1,
                      action_sha256=str(i), post_observation_sha256=f"state-{i}",
                      reward=-1.0, done=i == len(values) - 1)
                 for i, value in enumerate(values)]
    return dict(controller_ms=sum(values) * multiplier, environment_ms=.1 * len(values),
                return_=-float(len(values)), decisions=len(values), reset_ordinal=2,
                initial_observation_sha256="initial", actions_sha256="actions", decision_rows=decisions)


def episode(values=(1.0, 2.0), multiplier=1):
    result = fake_episode(values, multiplier)
    result["return"] = result.pop("return_")
    return result


class TimingTests(unittest.TestCase):
    def test_exclude_only_real_environment_and_restore_method(self):
        clock = Clock()
        env = Environment(clock)
        native = env.step

        def choose():
            clock.advance(500)  # Inference / public state / hypothetical Search.
            result = env.step((0, .5))
            clock.advance(100)  # Wrapper postprocessing.
            return result, "base"

        native_fingerprint = timing.fingerprint

        def slow_audit(value):
            clock.advance(10_000)
            return native_fingerprint(value)

        with mock.patch.object(timing, "fingerprint", side_effect=slow_audit):
            result = timing.timed_loop(env, choose, 3, clock)
        self.assertEqual(result["environment_ms"], .0004)
        self.assertAlmostEqual(result["controller_ms"], .0012)
        self.assertEqual([r["controller_ms"] for r in result["decision_rows"]], [.0006, .0006])
        self.assertEqual(env.step, native)
        self.assertNotIn("step", vars(env))
        self.assertEqual(result["reset_ordinal"], 2)

    def test_restore_existing_instance_override_after_failure(self):
        clock = Clock()
        env = Environment(clock)
        original = env.step
        env.step = original

        def fail():
            env.step((0, 1))
            raise RuntimeError("fake controller failure")

        with self.assertRaisesRegex(RuntimeError, "fake controller failure"):
            timing.timed_loop(env, fail, 3, clock)
        self.assertIs(env.step, original)

    def test_schedule_and_off_bank_warmup_are_reproducible_and_rng_isolated(self):
        blocks = [block("a"), block("b", "True")]
        calls = []

        def runner(spec, seed, model):
            calls.append((spec["id"], seed))
            return episode()

        random.seed(512)
        np.random.seed(1024)
        python_before, numpy_before = random.getstate(), np.random.get_state()
        models = {b["id"]: None for b in blocks}
        first = timing.run_suite(blocks, models, runner=runner)
        second = timing.run_suite(blocks, models, runner=runner)
        self.assertEqual(first["schedule"], second["schedule"])
        self.assertEqual(first["scheduler"]["seed"], 2026092401)
        self.assertEqual(random.getstate(), python_before)
        self.assertEqual(np.random.get_state()[0], numpy_before[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        self.assertEqual(np.random.get_state()[2:], numpy_before[2:])
        self.assertEqual(len(first["rows"]), 120)
        self.assertEqual(len(calls), 2 * (120 + 8))
        expected_seeds = {980000 + offset for offset in timing.OFFSETS}
        for round_index, schedule in enumerate(first["schedule"]):
            for item in schedule["blocks"]:
                self.assertEqual(set(item["environment_seeds"]), expected_seeds)
                self.assertEqual(item["warmup_environment_seeds"], [981180, 981181] if round_index == 0 else [981180])
        self.assertNotEqual(first["schedule"][0]["blocks"], first["schedule"][1]["blocks"])

    def test_repetition_medians_precede_instance_mean_and_decision_pooling(self):
        spec = block()
        rows = []
        for seed, values in ((980008, (1.0,)), (980041, (2.0, 4.0))):
            for repetition, factor in enumerate((1, 3, 2)):
                rows.append(dict(block_id="a", environment_seed=seed, repetition=repetition,
                                 **episode(values, factor)))
        summary = timing.aggregate_rows([spec], rows, 3)["controller_interfaces"][0]
        self.assertEqual(summary["mean_ms_per_episode"], 7.0)
        self.assertAlmostEqual(summary["pooled_ms_per_decision"], 14 / 3)

    def test_mask_episode_average_and_pooled_decision_time(self):
        blocks = [block(f"c{k}", f"C{k}") for k in range(1, 6)]
        rows = []
        for k, spec in enumerate(blocks, 1):
            for repetition in range(3):
                rows.append(dict(block_id=spec["id"], environment_seed=980008, repetition=repetition,
                                 **episode((float(k),) * k)))
        masks = timing.aggregate_rows(blocks, rows, 3)["masks"]
        self.assertEqual(masks[0]["mean_ms_per_episode"], 11.0)
        self.assertAlmostEqual(masks[0]["pooled_ms_per_decision"], 55 / 15)

    def test_episode_median_is_not_sum_of_decision_medians(self):
        rows = [dict(block_id="a", environment_seed=980008, repetition=index, **episode(values))
                for index, values in enumerate(((1.0, 100.0), (50.0, 50.0), (100.0, 1.0)))]
        summary = timing.aggregate_rows([block()], rows, 3)["controller_interfaces"][0]
        self.assertEqual(summary["mean_ms_per_episode"], 101.0)
        self.assertEqual(summary["pooled_ms_per_decision"], 50.0)

    def test_changed_actions_length_return_or_instance_rejected(self):
        first = episode()
        for field, value in (("return", -99), ("decisions", 3),
                             ("initial_observation_sha256", "changed"), ("actions_sha256", "changed"),
                             ("reset_ordinal", 1)):
            altered = copy.deepcopy(first)
            altered[field] = value
            with self.assertRaises(RuntimeError):
                timing.assert_same_episode(first, altered)
        altered = copy.deepcopy(first)
        altered["decision_rows"][0]["action_sha256"] = "different-action"
        with self.assertRaises(RuntimeError):
            timing.assert_same_episode(first, altered)
        altered = copy.deepcopy(first)
        altered["decision_rows"][0]["post_observation_sha256"] = "different-state"
        with self.assertRaises(RuntimeError):
            timing.assert_same_episode(first, altered)

    def test_manifest_duplicate_or_unhashed_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(json.dumps(dict(version=1, blocks=[block(), block()])))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                timing.read_manifest(manifest)
        learned = dict(id="p", benchmark="cash10", controller="raw", interface="flat", checkpoint="fresh.zip", training_seed=42)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            timing.validate_block(learned)
        with self.assertRaises(ValueError):
            timing.validate_output(timing.SOURCE_ROOT / "new-output.json")

    def test_incomplete_repetitions_rejected(self):
        rows = [dict(block_id="a", environment_seed=980008, repetition=0, **episode())]
        with self.assertRaisesRegex(RuntimeError, "Missing or duplicate"):
            timing.aggregate_rows([block()], rows, 3)

    def test_example_manifest_has_only_checkpoint_free_anonymous_ids(self):
        path = Path(timing.__file__).with_name("controller_timing_manifest.example.json")
        _, blocks = timing.read_manifest(path)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(b["controller"] == "milp" and not b.get("checkpoint") for b in blocks))

    def test_highs_single_thread_override_restored(self):
        from . import milp_planner
        solver = mock.Mock(return_value="fake-solve-result")
        original_options = {"presolve": True, "mip_rel_gap": 0.0}
        with mock.patch.object(milp_planner, "milp", solver):
            with timing.single_thread_highs():
                self.assertEqual(milp_planner.milp("objective", options=original_options), "fake-solve-result")
                self.assertEqual(solver.call_args.kwargs["options"], {**original_options, "threads": 1})
            self.assertIs(milp_planner.milp, solver)
        self.assertNotIn("threads", original_options)

    def test_manifest_relative_checkpoint_and_explicit_seed(self):
        learned = dict(id="fresh-raw-42", benchmark="cash10", controller="raw", interface="flat",
                       checkpoint="fresh-output/final.zip", sha256="a" * 64, training_seed=42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(dict(version=1, blocks=[learned])))
            _, blocks = timing.read_manifest(path)
            self.assertEqual(blocks[0]["checkpoint"], str((Path(directory) / "fresh-output/final.zip").resolve()))
            self.assertEqual(blocks[0]["training_seed"], 42)
        del learned["training_seed"]
        with self.assertRaisesRegex(ValueError, "training_seed"):
            timing.validate_block(learned)


if __name__ == "__main__":
    unittest.main()
