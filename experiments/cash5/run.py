"""Explicit training/rollout entrypoints for the five-account benchmark."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.cash5"

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    choices = ["Flat", "True"] + [f"{x}{i}" for x in ["U", "C"] for i in range(1, 6)]
    t = sub.add_parser("train")
    t.add_argument("--setting", required=True, choices=choices)
    t.add_argument("--seed", type=int, required=True, choices=range(42, 52))
    t.add_argument("--output", required=True, type=Path)
    t.add_argument(
        "--smoke", action="store_true", help="256 transitions only; not a paper result"
    )
    e = sub.add_parser("evaluate-final")
    e.add_argument("--setting", required=True, choices=choices)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--sha256", required=True)
    e.add_argument(
        "--trust-checkpoint",
        action="store_true",
        required=True,
        help="Only load your own trusted model archive; pickle loading can execute code",
    )
    e.add_argument("--output", required=True, type=Path)
    learning = sub.add_parser("evaluate-learning", help="Offline fixed20 curves from saved scheduled checkpoints; no training or selection")
    learning.add_argument("--run", required=True, type=Path)
    learning.add_argument("--setting", required=True, choices=choices)
    learning.add_argument("--seed", required=True, type=int)
    learning.add_argument("--inventory-sha256", required=True, help="SHA-256 of the trusted run's checkpoint_inventory.json")
    learning.add_argument("--reward-convention", required=True, choices=("native", "vec-float32"), help="Native rewards or per-step float32 transport, both summed as Python floats")
    learning.add_argument("--trust-checkpoint", action="store_true", required=True, help="Only deserialize your own or independently trusted scheduled checkpoints")
    learning.add_argument("--output", required=True, type=Path, help="New directory outside the source package and training run")
    learning.add_argument("--smoke", action="store_true", help="Permit short fixture checkpoints; not a paper result")
    learning.add_argument("--steps", type=int, nargs="+", help="Custom scheduled timesteps, only with --smoke")
    r = sub.add_parser("random")
    r.add_argument("--output", required=True, type=Path)
    r.add_argument(
        "--smoke",
        action="store_true",
        help="One environment and one replicate per setting",
    )
    m = sub.add_parser("generate-masks")
    m.add_argument("--output-directory", required=True, type=Path)
    a = p.parse_args()
    if hasattr(a, "output") and a.output.exists():
        raise FileExistsError(a.output)
    if a.command == "train":
        steps = 256 if a.smoke else 500000
        if a.setting in ["Flat", "True"]:
            from .training import train_policy

            train_policy(
                "flat" if a.setting == "Flat" else "mmdp",
                a.seed,
                steps,
                output_root=a.output,
            )
        else:
            from .training import train_mask

            train_mask(a.setting, a.seed, steps, a.output)
        (a.output / "REPRODUCTION_MODE.json").write_text(
            json.dumps(dict(smoke=a.smoke, nominal_steps=steps)) + "\n"
        )
    elif a.command == "evaluate-final":
        from .evaluation import initialize_worker, load_frozen_model, evaluate_episode

        initialize_worker()
        meta = dict(
            setting_id=a.setting,
            training_seed=a.seed,
            checkpoint_path=a.checkpoint,
            checkpoint_sha256=a.sha256,
            actual_steps=500224,
        )
        model = load_frozen_model(meta)
        with a.output.open("x") as handle:
            for seed in range(980000, 980200):
                row, _ = evaluate_episode(model, a.setting, a.seed, seed, meta)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    elif a.command == "evaluate-learning":
        from .learning import evaluate_run
        from .evaluation import initialize_worker

        initialize_worker()
        metadata = evaluate_run(a.run, a.setting, a.seed, a.inventory_sha256, a.output,
                                reward_convention=a.reward_convention, trusted=a.trust_checkpoint,
                                smoke=a.smoke, steps=a.steps)
        print(json.dumps({key: metadata[key] for key in ("provenance", "setting_id", "training_seed", "points_recorded", "episode_rows", "last_scheduled_step", "training_endpoint_actual_steps", "reward_convention", "smoke_only")}, sort_keys=True))
    elif a.command == "random":
        from .random_policy import episode

        with a.output.open("x") as handle:
            for seed in range(980000, 980001 if a.smoke else 980200):
                for setting in ["Flat", "True", "C1", "C2", "C3", "C4", "C5"]:
                    for rep in range(1 if a.smoke else 10):
                        row, _ = episode(setting, seed, rep)
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
    else:
        from . import generate_masks

        a.output_directory.mkdir(parents=True, exist_ok=False)
        (a.output_directory / "design.md").write_bytes(
            (generate_masks.ROOT / "design.md").read_bytes()
        )
        generate_masks.ROOT = a.output_directory
        generate_masks.main()


if __name__ == "__main__":
    main()
