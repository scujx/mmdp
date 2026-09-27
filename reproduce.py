"""Single entry point for smoke checks or fresh synthetic experiment reruns.

No dependencies are installed automatically. Full mode is deliberately explicit;
it trains models from scratch rather than requiring pretrained checkpoints.
"""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ("replenishment", "cash5", "cash10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--benchmark", choices=("all", *BENCHMARKS), default="all")
    parser.add_argument("--output", type=Path, help="New output root, required for full mode")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing or writing files")
    for benchmark in BENCHMARKS:
        parser.add_argument(f"--{benchmark}-python", default=sys.executable,
                            help=f"Python interpreter with {benchmark} dependencies installed")
    args = parser.parse_args()
    selected = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    output = args.output.resolve() if args.output else None
    if args.mode == "full":
        if output is None:
            parser.error("Full mode requires --output pointing to a new directory")
        if output.exists():
            parser.error("Output exists; use a new directory (no implicit resume or overwrite)")
        if output == ROOT or ROOT in output.parents:
            parser.error("Keep generated outputs outside the source package")
    elif output is not None:
        parser.error("Smoke tests use temporary directories; --output is only for full mode")

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"

    def run(benchmark, *command):
        python = getattr(args, benchmark + "_python")
        cmd = [python, "-B", *map(str, command)]
        print(shlex.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    # Check every selected environment before starting any costly work.
    for benchmark in selected:
        version = "2.7.1"
        run(benchmark, "-c", "import stable_baselines3 as sb3; "
            f"assert sb3.__version__ == {version!r}, "
            f"'Install experiments/{benchmark}/requirements.txt in this interpreter'")

    if args.mode == "smoke":
        for benchmark in selected:
            run(benchmark, "-m", f"experiments.{benchmark}.tests")
            if benchmark in ("cash5", "cash10"):
                run(benchmark, "-m", f"experiments.{benchmark}.milp", "--smoke")
            if benchmark == "cash5":
                run(benchmark, "-m", "experiments.cash5.learning_tests")
            if benchmark == "cash10":
                run(benchmark, "-m", "experiments.cash10.validation_tests")
                run(benchmark, "-m", "experiments.controller_timing_tests")
                if args.dry_run:
                    run(benchmark, "-m", "experiments.cash10.random_policy", "--smoke",
                        "--output", Path(tempfile.gettempdir()) / "mmdp-cash10-random-smoke" / "result")
                else:
                    with tempfile.TemporaryDirectory(prefix="mmdp-cash10-random-") as temporary:
                        run(benchmark, "-m", "experiments.cash10.random_policy", "--smoke",
                            "--output", Path(temporary) / "result")
        print("Smoke checks complete; these are not paper-performance results.")
        return

    print("Full fresh rerun: 40 replenishment, 120 cash5, and 40 cash10 models "
          "if all benchmarks are selected. Execution is sequential.", flush=True)
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=False)

    def digest(checkpoint):
        # In full mode we load only checkpoints just produced by this invocation.
        if args.dry_run:
            return "SHA256_OF_NEW_CHECKPOINT"
        return hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    for benchmark in selected:
        dest = output / benchmark
        if not args.dry_run:
            dest.mkdir()
        if benchmark == "replenishment":
            selected_runs = []
            for cell in ("flat_retained", "flat_updated", "mmdp_retained", "mmdp_updated"):
                for seed in range(42, 52):
                    model_dir = dest / "runs" / f"{cell}_{seed}"
                    run(benchmark, "-m", "experiments.replenishment.run", "train", "--cell", cell,
                        "--seed", seed, "--output", model_dir)
                    selected_runs.append((model_dir, cell, seed))
            # Test only after every policy has been selected on validation.
            for model_dir, cell, seed in selected_runs:
                run(benchmark, "-m", "experiments.replenishment.run", "evaluate", "--run", model_dir,
                    "--trust-checkpoint", "--output", dest / "evaluations" / f"{cell}_{seed}.jsonl")
            run(benchmark, "-m", "experiments.replenishment.analyze", "--runs", dest / "runs",
                "--evaluations", dest / "evaluations", "--output", dest / "analysis")
        elif benchmark == "cash5":
            evaluations = []
            settings = ["Flat", "True"] + [f"{group}{i}" for group in ("C", "U") for i in range(1, 6)]
            for setting in settings:
                for seed in range(42, 52):
                    model_dir = dest / f"{setting}_{seed}"
                    run(benchmark, "-m", "experiments.cash5.run", "train", "--setting", setting,
                        "--seed", seed, "--output", model_dir)
                    checkpoint = model_dir / "final_model.zip"
                    evaluation = dest / f"{setting}_{seed}.jsonl"
                    run(benchmark, "-m", "experiments.cash5.run", "evaluate-final", "--setting", setting,
                        "--seed", seed, "--checkpoint", checkpoint, "--sha256", digest(checkpoint),
                        "--trust-checkpoint", "--output", evaluation)
                    evaluations.append(evaluation)
                    if setting in ("Flat", "True"):
                        # Offline curve scoring cannot change training or select an endpoint.
                        run(benchmark, "-m", "experiments.cash5.run", "evaluate-learning",
                            "--run", model_dir, "--setting", setting, "--seed", seed,
                            "--inventory-sha256", digest(model_dir / "checkpoint_inventory.json"),
                            "--reward-convention", "native", "--trust-checkpoint",
                            "--output", dest / "learning" / f"{setting}_{seed}")
            run(benchmark, "-m", "experiments.cash5.run", "random", "--output", dest / "random.jsonl")
            run(benchmark, "-m", "experiments.cash5.analyze", "--controller", "ppo", "--input",
                *evaluations, "--output", dest / "ppo-statistics.json")
            run(benchmark, "-m", "experiments.cash5.analyze", "--controller", "random", "--input",
                dest / "random.jsonl", "--output", dest / "random-statistics.json")
            run(benchmark, "-m", "experiments.cash5.milp", "--output", dest / "milp")
        else:
            evaluations = []
            for policy in ("flat", "mmdp", "flat-aa", "mmdp-aa"):
                for seed in range(42, 52):
                    model_dir = dest / f"{policy}_{seed}"
                    run(benchmark, "-m", "experiments.cash10.run", "train", "--policy", policy,
                        "--seed", seed, "--output", model_dir)
                    checkpoint = model_dir / "final.zip"
                    evaluation = dest / f"{policy}_{seed}_evaluation"
                    run(benchmark, "-m", "experiments.cash10.run", "evaluate", "--policy", policy,
                        "--checkpoint", checkpoint, "--sha256", digest(checkpoint), "--trust-checkpoint",
                        "--caps", 0, 576, 1152, 2304, "--output", evaluation)
                    evaluations.append(evaluation)
            run(benchmark, "-m", "experiments.cash10.analyze", "--input", *evaluations,
                "--output", dest / "statistics.json")
            run(benchmark, "-m", "experiments.cash10.milp", "--output", dest / "milp")
            run(benchmark, "-m", "experiments.cash10.random_policy", "--output", dest / "random")
    print("Dry run complete; no files written." if args.dry_run else "Fresh experiment rerun complete.")


if __name__ == "__main__":
    main()
