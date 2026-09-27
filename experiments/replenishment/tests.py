"""Functional checks plus four 512-transition smoke runs, never paper evaluation.

All files are created in a temporary directory and removed on exit. Smoke
instances use development namespaces, not the validation or test banks.
"""

from __future__ import annotations

import copy
import json
import tempfile
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.replenishment"


from .envs.environment import (
    CAP,
    CFG,
    D,
    JOINT,
    PF,
    PC,
    COARSE_GIVEN_DEMAND,
    signal_law,
    make_episode,
    posterior,
    LeadTimeReplenishment,
)
from .runtime import (
    CELLS,
    evaluate,
    make_model,
    np,
    policy_hash,
    preserve_rng,
    torch,
)
from .run import train, evaluate_run
from .analyze import summarize, validate_records


def functional_checks():
    for accuracy in (0.75, 0.80, 0.85):
        conditional, joint = signal_law(accuracy)
        np.testing.assert_allclose(joint.sum(axis=1) * 3, COARSE_GIVEN_DEMAND, atol=1e-12)
        np.testing.assert_allclose(conditional.sum(axis=1), 1, atol=1e-12)
    low = make_episode("test", 9242000, 0, 0.75)
    high = make_episode("test", 9242000, 0, 0.85)
    np.testing.assert_array_equal(low[0], high[0])  # Demand category.
    np.testing.assert_array_equal(low[2], high[2])  # Shared coarse signal.
    np.testing.assert_array_equal(low[3], high[3])  # Daily demand.
    np.testing.assert_allclose(JOINT.sum(), 1)
    np.testing.assert_allclose(PF.sum(axis=1), 1)
    np.testing.assert_allclose(PC.sum(axis=1), 1)
    np.testing.assert_allclose(np.diag(PF), CFG["fine_correct"])
    np.testing.assert_allclose(np.diag(PC), CFG["coarse_keeps_fine"])
    np.testing.assert_allclose(np.diag(PF @ PC), 0.57875)
    coarse_error = fine_error = 0.0
    for c in range(3):
        prob = posterior(c)
        coarse_error += JOINT[:, :, c].sum() * (prob @ (D - prob @ D) ** 2)
        for f in range(3):
            prob = posterior(c, f)
            fine_error += JOINT[:, f, c].sum() * (prob @ (D - prob @ D) ** 2)
    assert fine_error < coarse_error

    for identity in range(9244000, 9244004):
        fingerprints = []
        for cell in CELLS:
            interface, information = cell.split("_")
            env = LeadTimeReplenishment(interface, information, "dev", identity)
            obs, _ = env.reset()
            assert obs.shape == (24,) and env.observation_space.contains(obs)
            assert env.inventory == 63.0
            fingerprints.append(env.fingerprint)
            np.testing.assert_array_equal(env.daily.sum(axis=1), D[env.state])
            hidden = copy.deepcopy(env)
            hidden.daily[:] = 0
            hidden.state[:] = 2
            np.testing.assert_array_equal(hidden.obs(), obs)
            total = 0.0
            all_days = []
            for step in range(26):
                week, stage = env.week, env.stage
                obs, reward, done, truncated, info = env.step([-0.3, 0.2, -0.6])
                total += reward
                all_days.extend(info["daily_trace"])
                assert done == (step == 25) and not truncated
                assert env.observation_space.contains(obs)
                if stage == 0:
                    assert info["arrived"] == 0
                    assert len(info["daily_trace"]) == 3
                    if interface == "mmdp":
                        assert info["executed_flexible"] == 0
                    if information == "updated":
                        assert obs[22] == 1 and np.argmax(obs[19:22]) == env.f[week + 1]
                    # The coarse category remains observable after the update.
                    assert np.argmax(obs[16:19]) == env.c[week + 1]
                elif not done:
                    assert obs[6] == 1 and env.reg == env.flex == 0
                    if information == "updated":
                        assert obs[15] == 1 and np.argmax(obs[12:15]) == env.f[week + 1]
                if information == "retained":
                    assert np.all(obs[12:16] == 0) and np.all(obs[19:23] == 0)
            assert env.days == 98 and env.length == 26
            assert [(r["week"], r["day"]) for r in all_days] == [
                (w, d) for w in range(14) for d in range(7)
            ]
            expected = env.totals["revenue"] - sum(
                v for k, v in env.totals.items() if k != "revenue"
            )
            np.testing.assert_allclose(total, expected, rtol=0, atol=1e-8)
            old_fingerprint = env.fingerprint
            env.reset()
            assert env.fingerprint != old_fingerprint  # Fresh episode on every reset.
            fresh = LeadTimeReplenishment(interface, information, "dev", identity)
            fresh.reset()
            assert fresh.fingerprint == old_fingerprint
        assert len(set(fingerprints)) == 1

    # Preserve physical outcomes when a feasible flexible quantity is deferred.
    for early, late in [(0, CAP), (CAP, 0), (12.345, 23.456), (33.6, 33.6)]:
        flat = LeadTimeReplenishment("flat", "updated")
        mmdp = LeadTimeReplenishment("mmdp", "updated")
        flat.reset()
        mmdp.reset()
        _, r1, _, _, _ = flat.step([-0.5, -0.5, 2 * early / CAP - 1])
        _, r2, _, _, _ = mmdp.step([-0.5, -0.5, -1])
        o1, s1, _, _, _ = flat.step([-1, -1, 2 * late / CAP - 1])
        o2, s2, _, _, _ = mmdp.step([-1, -1, 2 * (early + late) / CAP - 1])
        np.testing.assert_allclose(o1, o2, rtol=0, atol=1e-7)
        np.testing.assert_allclose(r1 + s1, r2 + s2, rtol=0, atol=1e-8)
        for _ in range(24):
            a = flat.step([-0.4, -0.2, -1])
            b = mmdp.step([-0.4, -0.2, -1])
            np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-7)
            np.testing.assert_allclose(a[1], b[1], rtol=0, atol=1e-8)

    # Equal pending totals with different supplier use remain distinguishable.
    first = LeadTimeReplenishment()
    first.reset()
    second = copy.deepcopy(first)
    first.step([2 * 10 / 168 - 1, 2 * 10 / 168 - 1, -1])
    second.step([-1, -1, 2 * 10 / CAP - 1])
    np.testing.assert_allclose(first.reg + first.flex, second.reg + second.flex)
    assert first.obs()[6] != second.obs()[6]


def smoke_checks():
    initial_hashes = []
    with tempfile.TemporaryDirectory(prefix="mmdp-replenishment-smoke-") as directory:
        root = Path(directory)
        for cell in CELLS:
            path = root / cell
            train(cell, 42, path, smoke=True)
            meta = json.loads((path / "run.json").read_text())
            initial_hashes.append(meta["initial_policy_hash"])
            assert meta["actual_steps"] == 512 and meta["parameter_count"] == 39943
            assert meta["selection"]["step"] in [256, 512]
            output = root / f"{cell}.jsonl"
            evaluate_run(path, output, trust_checkpoint=True, smoke=True)
            rows = [json.loads(x) for x in output.read_text().splitlines()]
            validate_records(rows, "dev", [9243010, 9243011], {})
            try:
                evaluate_run(path, root / "wrong-mode.jsonl", trust_checkpoint=True)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    "Smoke checkpoints must not enter formal evaluation."
                )
            try:
                evaluate_run(path, output, trust_checkpoint=True, smoke=True)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Existing evaluations must not be overwritten.")
        assert len(set(initial_hashes)) == 1

    # Evaluation must leave learner RNG and policy tensors unchanged.
    env = LeadTimeReplenishment()
    model = make_model(env, 42)
    rng_before = torch.get_rng_state().clone()
    weights_before = policy_hash(model)
    evaluate(model, "flat_updated", "dev", [9243000, 9243001])
    assert torch.equal(rng_before, torch.get_rng_state())
    assert weights_before == policy_hash(model)

    # Constant seed contrasts produce exact, zero-width bootstrap intervals.
    matrix = np.tile([10.0, 20.0, 30.0, 45.0], (10, 1))
    stats = summarize(matrix, matrix)
    for row in stats:
        assert row["ci_low"] == row["mean"] == row["ci_high"]
    interaction = next(r for r in stats if r["estimand"] == "interaction_secondary")
    assert interaction["mean"] == 5.0


if __name__ == "__main__":
    functional_checks()
    smoke_checks()
    print(
        "PASS: environment, signals, capacity, deferral, accounting, four 512-step smoke runs, selected-checkpoint evaluation, and paired statistics. No formal training or test-bank evaluation."
    )
