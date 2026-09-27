"""Train one policy or evaluate a trusted validation-selected checkpoint.

The root reproduce.py command trains all forty policies before testing any of
them. Standalone commands let users inspect or resume individual completed runs.
Outputs must be new; partial runs are never silently overwritten or resumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.replenishment"


from .runtime import (
    CELLS,
    ROOT,
    FixedRewardScale,
    Replenishment,
    PPO,
    dump,
    evaluate,
    make_model,
    np,
    policy_hash,
    sha,
    torch,
)
from stable_baselines3.common.callbacks import BaseCallback

PROTOCOL = json.loads((ROOT / "configs/experiment.json").read_text())


def source_hashes():
    """Relative paths only: keep generated metadata portable and anonymous."""
    paths = list(ROOT.rglob("*.py"))
    paths += list((ROOT / "configs").glob("*.json"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def train(cell, seed, output, smoke=False, fine_accuracy=0.80):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    interface, information = cell.split("_")
    split = "smoke" if smoke else "train"
    env = FixedRewardScale(Replenishment(interface, information, split, seed, fine_accuracy=fine_accuracy))
    model = make_model(env, seed)
    initial_hash = policy_hash(model)
    parameter_count = sum(p.numel() for p in model.policy.parameters())
    assert parameter_count == PROTOCOL["parameter_count"]
    ids = list(range(9243000, 9243004)) if smoke else PROTOCOL["validation_ids"]
    validation_split = "dev" if smoke else "validation"
    cadence = 256 if smoke else 5000
    total_steps = 512 if smoke else PROTOCOL["nominal_steps"]
    # Record initialization as a diagnostic, preserving all learner RNG states.
    initial_rows = evaluate(model, cell, validation_split, ids, fine_accuracy=fine_accuracy)
    history = []
    selected = None
    selected_path = output / "selected.zip"

    with (output / "validation.jsonl").open("x") as validation_file:

        class Validation(BaseCallback):
            def _on_step(self):
                nonlocal selected
                if self.num_timesteps % cadence:
                    return True
                rows = evaluate(self.model, cell, validation_split, ids, fine_accuracy=fine_accuracy)
                for row in rows:
                    validation_file.write(
                        json.dumps(dict(step=self.num_timesteps, **row)) + "\n"
                    )
                validation_file.flush()
                item = {
                    "step": self.num_timesteps,
                    "mean": float(np.mean([r["return"] for r in rows])),
                }
                history.append(item)
                # Strict improvement keeps the earliest checkpoint on a tie.
                if selected is None or item["mean"] > selected["mean"]:
                    self.model.save(selected_path)
                    selected = dict(item)
                print(json.dumps(dict(cell=cell, seed=seed, **item)), flush=True)
                return True

        model.learn(total_timesteps=total_steps, callback=Validation())

    expected_steps = 512 if smoke else PROTOCOL["actual_steps"]
    assert model.num_timesteps == expected_steps
    loaded = PPO.load(selected_path, device="cpu")  # Created by this invocation.
    assert loaded.num_timesteps == selected["step"]
    dump(
        output / "run.json",
        {
            "status": "complete",
            "smoke": smoke,
            "cell": cell,
            "seed": seed,
            "fine_accuracy": fine_accuracy,
            "actual_steps": model.num_timesteps,
            "parameter_count": parameter_count,
            "initial_policy_hash": initial_hash,
            "final_policy_hash": policy_hash(model),
            "initial_validation_mean": float(
                np.mean([r["return"] for r in initial_rows])
            ),
            "selection": dict(
                selected, checkpoint="selected.zip", sha256=sha(selected_path)
            ),
            "validation_history": history,
            "validation_ids": ids,
            "validation_split": validation_split,
            "validation_sha256": sha(output / "validation.jsonl"),
            "source_hashes": source_hashes(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    )
    env.close()


def evaluate_run(run, output, trust_checkpoint=False, smoke=False):
    if not trust_checkpoint:
        raise ValueError(
            "Model archives can execute code; pass --trust-checkpoint only for trusted runs."
        )
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((run / "run.json").read_text())
    if record["status"] != "complete" or record["smoke"] != smoke:
        raise ValueError("Incomplete run or smoke/formal mismatch.")
    # A checksum verifies identity, not whether a model archive is trustworthy.
    path = run / "selected.zip"
    if (
        record["selection"]["checkpoint"] != "selected.zip"
        or sha(path) != record["selection"]["sha256"]
    ):
        raise ValueError("Selected checkpoint does not match the run record.")
    if record["source_hashes"] != source_hashes():
        raise ValueError(
            "Run sources/configurations differ from the current implementation."
        )
    if sha(run / "validation.jsonl") != record["validation_sha256"]:
        raise ValueError("Validation records changed after training.")
    model = PPO.load(path, device="cpu")
    if model.num_timesteps != record["selection"]["step"]:
        raise ValueError(
            "Checkpoint transition count differs from the selected checkpoint."
        )
    ids = [9243010, 9243011] if smoke else PROTOCOL["test_ids"]
    split = "dev" if smoke else "test"
    rows = evaluate(model, record["cell"], split, ids, fine_accuracy=record.get("fine_accuracy", 0.80))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(
                        cell=record["cell"],
                        training_seed=record["seed"],
                        checkpoint_sha256=sha(path),
                        smoke=smoke,
                        **row,
                    )
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", help="Train and select using validation only."
    )
    training.add_argument("--cell", choices=CELLS, required=True)
    training.add_argument("--seed", type=int, choices=PROTOCOL["seeds"], required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--fine-accuracy", type=float, choices=(0.75, 0.80, 0.85), default=0.80)
    testing = commands.add_parser(
        "evaluate", help="Evaluate one trusted selected checkpoint."
    )
    testing.add_argument("--run", type=Path, required=True)
    testing.add_argument("--output", type=Path, required=True)
    testing.add_argument("--trust-checkpoint", action="store_true")
    testing.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    package = ROOT.parents[1]
    target = args.output.resolve()
    if target == package or package in target.parents:
        parser.error("Keep all generated outputs outside the source package.")
    if args.command == "train":
        train(args.cell, args.seed, args.output, args.smoke, args.fine_accuracy)
    else:
        evaluate_run(args.run, args.output, args.trust_checkpoint, args.smoke)


if __name__ == "__main__":
    main()
