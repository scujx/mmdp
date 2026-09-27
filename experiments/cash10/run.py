"""Train one fresh policy or evaluate a trusted policy, with explicit outputs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fresh training; never resume or select a checkpoint")
    evaluate = commands.add_parser("evaluate", help="Raw and/or Search on paired synthetic instances")
    for sub in (train, evaluate):
        sub.add_argument("--policy", choices=("flat", "mmdp", "flat-aa", "mmdp-aa"), required=True)
        sub.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--smoke", action="store_true", help="Only one 512-transition rollout; not a paper result")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--sha256", required=True, help="Digest independently obtained from a trusted source")
    evaluate.add_argument("--trust-checkpoint", action="store_true", help="Acknowledge that SB3 archives can execute serialized code")
    evaluate.add_argument("--seed-start", type=int, default=994000)
    evaluate.add_argument("--instances", type=int, default=200)
    evaluate.add_argument("--caps", type=int, nargs="+", choices=(0, 576, 1152, 2304), default=[0])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new directory")
    if args.command == "evaluate":
        if args.instances < 1 or len(args.caps) != len(set(args.caps)):
            parser.error("Use positive instance counts and unique caps")
        if not args.trust_checkpoint:
            parser.error("Load only a trusted checkpoint; explicit acknowledgement is required")
        if not args.checkpoint.is_file() or sha(args.checkpoint) != args.sha256.lower():
            parser.error("Checkpoint SHA-256 mismatch or missing file")
        metadata_path = args.checkpoint.with_name("run.json")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("policy") != args.policy or metadata.get("checkpoint_sha256") != args.sha256.lower():
                parser.error("Checkpoint metadata does not match the requested policy/hash")
    # One CPU thread avoids machine-dependent oversubscription in this small task.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import gymnasium
    import numpy
    import stable_baselines3
    from . import runtime
    args.output.mkdir(parents=True, exist_ok=False)
    versions = dict(numpy=numpy.__version__, torch=torch.__version__,
                    gymnasium=gymnasium.__version__, stable_baselines3=stable_baselines3.__version__)
    if args.command == "train":
        from .validation import RawValidationCallback
        model = runtime.make_model(args.policy, args.seed)
        try:
            validation = RawValidationCallback(
                args.policy, args.seed, args.output, smoke_only=args.smoke,
                interval=256 if args.smoke else 5_000,
                final_timestep=512 if args.smoke else 2_000_000)
            if args.smoke:
                model.learn(512, callback=validation)
            else:
                # Carry optimizer and timestep state across the two training segments.
                model.learn(1_500_000, callback=validation)
                if model.num_timesteps != 1_500_160:
                    raise RuntimeError("Unexpected prefix rollout boundary")
                model.learn(499_840, callback=validation, reset_num_timesteps=False)
            expected = 512 if args.smoke else 2_000_384
            if model.num_timesteps != expected:
                raise RuntimeError("Unexpected final rollout boundary")
            if validation.points_recorded != validation.final_timestep // validation.interval:
                raise RuntimeError("Incomplete validation grid")
            checkpoint = args.output / "final.zip"
            model.save(checkpoint)
            save_json(args.output / "validation.json", validation.metadata())
            save_json(args.output / "run.json", dict(
                policy=args.policy, training_seed=args.seed, actual_transitions=model.num_timesteps,
                smoke_only=args.smoke, selected_by_validation=False,
                checkpoint="final.zip", checkpoint_sha256=sha(checkpoint), versions=versions,
                training_protocol="two_segment_ppo_training_from_scratch",
                validation=validation.metadata(),
                aa_zero_handling="zero_passthrough" if args.policy.endswith("-aa") else "not_applicable"))
        finally:
            model.get_env().close()
    else:
        model = runtime.PPO.load(args.checkpoint, device="cpu")
        env = runtime.make_env(args.policy, args.seed_start)
        try:
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError("Checkpoint spaces do not match the chosen interface")
        finally:
            env.close()
        rows = [runtime.evaluate_episode(model, args.policy, seed, cap)
                for seed in range(args.seed_start, args.seed_start + args.instances)
                for cap in args.caps]
        with (args.output / "episodes.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_json(args.output / "evaluation.json", dict(
            policy=args.policy, checkpoint_sha256=args.sha256.lower(), versions=versions,
            actual_training_transitions=model.num_timesteps, training_seed=model.seed,
            environment_seed_start=args.seed_start, instances=args.instances,
            reset_ordinal=2, caps=args.caps, means={str(cap): float(numpy.mean(
                [row["episode_return"] for row in rows if row["planning_cap"] == cap])) for cap in args.caps},
            note="One policy evaluation; not a training-seed confidence interval"))
    print(f"Completed {args.command}; outputs are in the requested directory.")


if __name__ == "__main__":
    main()
