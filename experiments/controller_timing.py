"""Measure serial CPU controller time on a shared subset of cash test instances.

This is a fresh local timing entry point, not a source of portable fixed latency
values. Learned-policy modes require a newly trained, explicitly trusted model.
"""

from __future__ import annotations

import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import contextlib
import hashlib
import json
import math
import platform
import statistics
import time
import warnings
from pathlib import Path

OFFSETS = (8, 41, 53, 69, 77, 78, 84, 92, 99, 122,
           126, 128, 130, 132, 158, 161, 187, 191, 193, 199)
SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = "reconstructed_fresh_run_tool_not_recovered_historical_timing"
SCHEDULER_SEEDS = {"cash5": 2026092401, "cash10": 2026092308}
WARMUP_SEEDS = {"cash5": (981180, 981181), "cash10": (995006, 995007)}


def runtime_imports():
    # Import the pinned scientific dependencies only for an actual timing run.
    global ForecastMeanMILPRollingHorizon, milp5, random5, route_masks, milp10, runtime10
    from experiments.milp_planner import ForecastMeanMILPRollingHorizon
    from experiments.cash5 import milp as milp5
    from experiments.cash5 import random_policy as random5
    from experiments.cash5.wrappers import route_masks
    from experiments.cash10 import milp as milp10
    from experiments.cash10 import runtime as runtime10


@contextlib.contextmanager
def single_thread_highs():
    """Runtime-only solver scheduler override, restored after timing."""
    from experiments import milp_planner
    native = milp_planner.milp

    def solve(*args, **kwargs):
        kwargs["options"] = dict(kwargs.get("options") or {}, threads=1)
        return native(*args, **kwargs)

    milp_planner.milp = solve
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads.*", category=RuntimeWarning)
            yield
    finally:
        milp_planner.milp = native


def jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_model(path, sha256, trusted):
    if not trusted or not path or not sha256:
        raise ValueError("Learned policies require --checkpoint, --sha256, and --trust-checkpoint")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256.lower():
        raise ValueError("Checkpoint checksum mismatch")
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    model.policy.set_training_mode(False)
    return model


def episode_cash5(controller, setting, seed, replicate, model):
    routes = milp5.route_lists() if controller == "milp" else None
    env = route_masks.make_env(setting, seed) if controller != "milp" else None
    base = env.base_env if env is not None else milp5.CashFlowEnv(milp5.CashFlowEnvConfig(seed=int(seed)))
    if env is None:
        base.reset()
        base.reset()
        planner = ForecastMeanMILPRollingHorizon()
    else:
        obs, _ = env.reset()
    def choose():
        nonlocal obs
        result, kind = action_cash5(controller, setting, seed, replicate, model,
                                    env, base, routes, planner if controller == "milp" else None,
                                    obs if env is not None else None)
        if kind == "wrapper":
            obs = result[0]
        return result, kind
    try:
        return timed_loop(base, choose, 22)
    finally:
        if env is not None:
            env.close()


def action_cash5(controller, setting, seed, replicate, model, env, base, routes, planner, obs):
    if controller == "ppo":
        action, _ = model.predict(obs, deterministic=True)
        return env.step(action), "wrapper"
    if controller == "random":
        snapshot = random5.public_snapshot(base)
        choices = tuple(tuple(edge) for edge in route_masks.current_routes(env))
        action = random5.random_action(snapshot, choices, seed, replicate)
        return random5.execute_semantic(base, action), "base"
    available = routes[setting] if base.stage == 0 else list(base.cfg.stage2_edges)
    view = milp5.VisibleRouteView(base, available)
    (index, ratio), decision = planner.select_action(view)
    full = list(base.get_current_edges())
    mapped = len(full) if index == len(available) else full.index(available[index])
    result = base.step((mapped, ratio))
    planner.note_transition()
    return result, "base"


def episode_cash10(controller, policy, seed, model):
    env = runtime10.make_env(policy, seed) if controller != "milp" else None
    if env is None:
        base = milp10.CashFlowEnvV2(milp10.CashFlowEnvConfigV2(seed=int(seed)))
        base.reset()
        base.reset()
        planner = ForecastMeanMILPRollingHorizon()
    else:
        obs, _ = env.reset()
        base = env.base_env
    def choose():
        nonlocal obs
        result, kind = action_cash10(controller, policy, model, env, base,
                                     planner if controller == "milp" else None,
                                     obs if env is not None else None)
        if kind == "wrapper":
            obs = result[0]
        return result, kind
    try:
        return timed_loop(base, choose, 34)
    finally:
        if env is not None:
            env.close()


def action_cash10(controller, policy, model, env, base, planner, obs):
    if controller == "raw":
        action, _ = model.predict(obs, deterministic=True)
        return env.step(action), "wrapper"
    if controller == "search":
        state = runtime10.public_state(env)
        selected = runtime10.search.select_dfm_q9_action(
            model=model, current_obs=obs.copy(), state=state,
            method_spec=runtime10.search.METHOD_SPECS[runtime10.FAMILIES[policy]],
            wrapper_factory=runtime10.WRAPPERS[policy], transition_cap=2304)
        return env.step(selected.action), "wrapper"
    interface = "Flat" if policy == "flat" else "MMDP"
    full = [tuple(edge) for edge in base.get_current_edges()]
    available = full if interface == "Flat" else milp10.mmdp_routes(base)
    view = milp10.RestrictedRouteView(base, available)
    (index, ratio), decision = planner.select_action(view)
    mapped = len(full) if decision["decision"]["kind"] == "STOP" else full.index(available[index])
    result = base.step((mapped, ratio))
    planner.note_transition()
    return result, "base"


def timed_loop(base, choose_and_step, limit, clock=None):
    """Time decisions, excluding only physical steps of this real environment.

    Hypothetical environments created by Search are not patched. Bookkeeping,
    hashing, and return checks occur outside each measured decision interval.
    """
    clock = time.perf_counter_ns if clock is None else clock
    physical_ns = 0
    native_step = base.step
    had_local_step = "step" in vars(base)
    previous_local_step = vars(base).get("step")
    actions = []
    initial_hash = fingerprint(base.get_observation_dict())

    def measured_step(action):
        nonlocal physical_ns
        # Capture the action outside the physical timer; conversion occurs later.
        actions.append(action)
        start = clock()
        result = native_step(action)
        physical_ns += clock() - start
        return result

    base.step = measured_step
    total_return = 0.0
    decisions = []
    try:
        for length in range(1, limit + 1):
            stage = int(base.stage)
            step = int(base.step_in_stage)
            before_ns, before_actions = physical_ns, len(actions)
            start = clock()
            result, kind = choose_and_step()
            elapsed_ns = clock() - start
            environment_ns = physical_ns - before_ns
            if len(actions) != before_actions + 1:
                raise RuntimeError("Each decision must advance the real environment exactly once")
            if environment_ns > elapsed_ns:
                raise RuntimeError("Physical environment time exceeds decision time")
            if kind == "wrapper":
                obs, reward, done, truncated, info = result
                if truncated:
                    raise RuntimeError("Unexpected truncation")
            else:
                obs, reward, done, info = result
            if not math.isfinite(float(reward)):
                raise RuntimeError("Non-finite episode reward")
            total_return += float(reward)
            decisions.append(dict(decision_ordinal=length, stage=stage, step_in_stage=step,
                                  controller_ms=(elapsed_ns - environment_ns) / 1e6,
                                  environment_ms=environment_ns / 1e6,
                                  executed_action=jsonable(actions[-1]),
                                  action_sha256=fingerprint(actions[-1]), reward=float(reward),
                                  post_observation_sha256=fingerprint(base.get_observation_dict()),
                                  done=bool(done)))
            if done:
                break
        else:
            raise RuntimeError("Episode exceeded its documented decision limit")
        return {"controller_ms": sum(row["controller_ms"] for row in decisions),
                "environment_ms": physical_ns / 1e6, "return": total_return,
                "decisions": length, "reset_ordinal": 2,
                "initial_observation_sha256": initial_hash, "actions_sha256": fingerprint(actions),
                "decision_rows": decisions}
    finally:
        if had_local_step:
            base.step = previous_local_step
        else:
            del base.step


def validate_block(block):
    benchmark, controller, interface = (block.get(key) for key in ("benchmark", "controller", "interface"))
    settings = {"cash5": ("Flat", "True", "C1", "C2", "C3", "C4", "C5"),
                "cash10": ("flat", "mmdp", "flat-aa", "mmdp-aa")}
    controllers = {"cash5": ("ppo", "random", "milp"), "cash10": ("raw", "search", "milp")}
    if benchmark not in settings or controller not in controllers[benchmark] or interface not in settings[benchmark]:
        raise ValueError("Invalid benchmark/controller/interface timing block")
    if benchmark == "cash10" and controller == "milp" and interface not in ("flat", "mmdp"):
        raise ValueError("Ten-account MILP uses flat or mmdp")
    if not isinstance(block.get("id"), str) or not block["id"].strip():
        raise ValueError("Every timing block requires a nonempty id")
    replicate = block.get("random_replicate", 0)
    if type(replicate) is not int or replicate not in range(10):
        raise ValueError("Random replicate must be 0-9")
    block["random_replicate"] = replicate
    learned = controller in ("ppo", "raw", "search")
    if learned:
        if not block.get("checkpoint") or not isinstance(block.get("sha256"), str):
            raise ValueError("Learned blocks require a checkpoint and SHA-256")
        digest = block["sha256"].lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Require a complete hexadecimal checkpoint SHA-256")
        block["sha256"] = digest
        if type(block.get("training_seed")) is not int:
            raise ValueError("Manifest learned blocks require an explicit training_seed")
    elif block.get("checkpoint") or block.get("sha256") or block.get("training_seed") is not None:
        raise ValueError("Non-learning blocks must not specify a checkpoint or training seed")
    if controller != "random" and replicate:
        raise ValueError("random_replicate applies only to Random")
    return block


def read_manifest(path):
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("blocks"), list) or not data["blocks"]:
        raise ValueError("Require a version-1 manifest with a nonempty blocks list")
    blocks = []
    for raw in data["blocks"]:
        if not isinstance(raw, dict):
            raise ValueError("Manifest blocks must be objects")
        block = validate_block(dict(raw))
        if block.get("checkpoint"):
            checkpoint = Path(block["checkpoint"])
            block["checkpoint"] = str((path.parent / checkpoint).resolve()) if not checkpoint.is_absolute() else str(checkpoint.resolve())
        blocks.append(block)
    identities = set()
    for block in blocks:
        identity = (block["benchmark"], block["controller"], block["interface"],
                    block.get("training_seed"), block["random_replicate"])
        if identity in identities:
            raise ValueError("Duplicate controller/interface/replicate timing block")
        identities.add(identity)
    if len({block["id"] for block in blocks}) != len(blocks):
        raise ValueError("Timing block ids must be unique")
    if len({block["benchmark"] for block in blocks}) != 1:
        raise ValueError("Run each benchmark separately with its own timing manifest")
    return data, blocks


def preload_models(blocks, trusted, cache=None):
    """Verify and load all models before any warm-up or measured round."""
    models = {}
    cache = {} if cache is None else cache
    for block in blocks:
        if block["controller"] not in ("ppo", "raw", "search"):
            models[block["id"]] = None
            continue
        key = (block["checkpoint"], block["sha256"])
        if key not in cache:
            cache[key] = load_model(Path(key[0]), key[1], trusted)
        model = cache[key]
        if model.seed != block["training_seed"]:
            raise ValueError("Checkpoint training seed does not match the timing manifest")
        block["actual_training_transitions"] = int(model.num_timesteps)
        env = (route_masks.make_env(block["interface"], 880000) if block["benchmark"] == "cash5"
               else runtime10.make_env(block["interface"], 880000))
        try:
            if model.observation_space != env.observation_space or model.action_space != env.action_space:
                raise ValueError("Checkpoint spaces do not match the timing block")
        finally:
            env.close()
        models[block["id"]] = model
    return models


def run_episode(block, seed, model):
    if block["benchmark"] == "cash5":
        return episode_cash5(block["controller"], block["interface"], seed,
                             block["random_replicate"], model)
    return episode_cash10(block["controller"], block["interface"], seed, model)


def assert_same_episode(reference, current):
    if (current["reset_ordinal"] != 2 or current["decisions"] != reference["decisions"]
            or current["initial_observation_sha256"] != reference["initial_observation_sha256"]
            or current["actions_sha256"] != reference["actions_sha256"]
            or not math.isclose(current["return"], reference["return"], rel_tol=0, abs_tol=1e-8)):
        raise RuntimeError("Timed episode identity, return, or length changed across repetitions")
    signatures = lambda row: [(d["decision_ordinal"], d["stage"], d["step_in_stage"],
                               d["action_sha256"], d["post_observation_sha256"], d["reward"], d["done"])
                              for d in row["decision_rows"]]
    if signatures(current) != signatures(reference):
        raise RuntimeError("Timed episode decisions changed across repetitions")
    if len(current["decision_rows"]) != current["decisions"]:
        raise RuntimeError("Incomplete decision timing records")
    if (not math.isclose(current["controller_ms"], sum(d["controller_ms"] for d in current["decision_rows"]), rel_tol=0, abs_tol=1e-8)
            or not math.isclose(current["environment_ms"], sum(d["environment_ms"] for d in current["decision_rows"]), rel_tol=0, abs_tol=1e-8)
            or not math.isclose(current["return"], sum(d["reward"] for d in current["decision_rows"]), rel_tol=0, abs_tol=1e-8)):
        raise RuntimeError("Episode totals do not reconstruct from decision records")


def aggregate_rows(blocks, rows, repetitions):
    """Episode means weight instances then controller replicates equally.

    Per-decision means pool visited decisions after matching their repetition
    medians. They are not averages of episode time divided by episode length.
    """
    summaries = []
    for block in blocks:
        subset = [row for row in rows if row["block_id"] == block["id"]]
        seeds = sorted({row["environment_seed"] for row in subset})
        episode_medians, decision_medians = [], []
        for seed in seeds:
            matched = [row for row in subset if row["environment_seed"] == seed]
            if len(matched) != repetitions or {row["repetition"] for row in matched} != set(range(repetitions)):
                raise RuntimeError("Missing or duplicate timing repetitions")
            for row in matched:
                assert_same_episode(matched[0], row)
            episode_medians.append(statistics.median(row["controller_ms"] for row in matched))
            for ordinal in range(matched[0]["decisions"]):
                decision_medians.append(statistics.median(row["decision_rows"][ordinal]["controller_ms"] for row in matched))
        if not seeds:
            raise RuntimeError("No timing episodes for a block")
        summaries.append(dict(block_id=block["id"], benchmark=block["benchmark"],
                              controller=block["controller"], interface=block["interface"],
                              training_seed=block.get("training_seed"), random_replicate=block["random_replicate"],
                              instances=len(seeds), visited_decisions=len(decision_medians),
                              mean_of_instance_medians_ms_per_episode=statistics.fmean(episode_medians),
                              pooled_median_ms_per_decision=statistics.fmean(decision_medians)))
    groups = []
    for key in sorted({(s["benchmark"], s["controller"], s["interface"]) for s in summaries}):
        members = [s for s in summaries if (s["benchmark"], s["controller"], s["interface"]) == key]
        total_decisions = sum(s["visited_decisions"] for s in members)
        groups.append(dict(benchmark=key[0], controller=key[1], interface=key[2],
                           controller_replicates=len(members),
                           mean_ms_per_episode=statistics.fmean(s["mean_of_instance_medians_ms_per_episode"] for s in members),
                           pooled_ms_per_decision=sum(s["pooled_median_ms_per_decision"] * s["visited_decisions"] for s in members) / total_decisions))
    masks = []
    for controller in ("ppo", "random", "milp"):
        members = [g for g in groups if g["benchmark"] == "cash5" and g["controller"] == controller and g["interface"] in ("C1", "C2", "C3", "C4", "C5")]
        if len(members) == 5:
            replicate_sets = [{(s["training_seed"], s["random_replicate"]) for s in summaries
                               if s["benchmark"] == "cash5" and s["controller"] == controller and s["interface"] == f"C{k}"}
                              for k in range(1, 6)]
            if any(values != replicate_sets[0] for values in replicate_sets[1:]):
                raise RuntimeError("C1-C5 Mask aggregation requires matched controller replicates")
            mask_blocks = [s for s in summaries if s["benchmark"] == "cash5" and s["controller"] == controller
                           and s["interface"] in ("C1", "C2", "C3", "C4", "C5")]
            mask_decisions = sum(s["visited_decisions"] for s in mask_blocks)
            masks.append(dict(benchmark="cash5", controller=controller, interface="Mask",
                              masks=["C1", "C2", "C3", "C4", "C5"],
                              controller_replicates_per_mask=members[0]["controller_replicates"],
                              mean_ms_per_episode=statistics.fmean(g["mean_ms_per_episode"] for g in members),
                              pooled_ms_per_decision=sum(s["pooled_median_ms_per_decision"] * s["visited_decisions"]
                                                         for s in mask_blocks) / mask_decisions))
    return dict(blocks=summaries, controller_interfaces=groups, masks=masks)


def run_suite(blocks, models, repetitions=3, scheduler_seed=None, smoke=False, runner=run_episode):
    if scheduler_seed is None:
        scheduler_seed = SCHEDULER_SEEDS[blocks[0]["benchmark"]]
    if type(repetitions) is not int or repetitions < 1 or type(scheduler_seed) is not int:
        raise ValueError("Require positive integer repetitions and an integer scheduler seed")
    import numpy as np
    scheduler = np.random.Generator(np.random.PCG64(scheduler_seed))
    seeds_by_block = {b["id"]: [(980000 if b["benchmark"] == "cash5" else 994000) + offset
                              for offset in (OFFSETS[:1] if smoke else OFFSETS)] for b in blocks}
    rows, schedule, references = [], [], {}
    for repetition in range(repetitions):
        round_schedule = []
        for block_order, index in enumerate(scheduler.permutation(len(blocks))):
            block = blocks[int(index)]
            warmups = WARMUP_SEEDS[block["benchmark"]] if repetition == 0 else WARMUP_SEEDS[block["benchmark"]][:1]
            for seed in warmups:
                runner(block, seed, models[block["id"]])
            seeds = [int(seed) for seed in scheduler.permutation(seeds_by_block[block["id"]])]
            round_schedule.append(dict(block_id=block["id"], warmup_environment_seeds=list(warmups), environment_seeds=seeds))
            for instance_order, seed in enumerate(seeds):
                episode = runner(block, seed, models[block["id"]])
                identity = (block["id"], seed)
                if identity in references:
                    assert_same_episode(references[identity], episode)
                else:
                    assert_same_episode(episode, episode)
                    references[identity] = episode
                rows.append(dict(block_id=block["id"], benchmark=block["benchmark"],
                                 controller=block["controller"], interface=block["interface"],
                                 training_seed=block.get("training_seed"), checkpoint_sha256=block.get("sha256"),
                                 random_replicate=block["random_replicate"], repetition=repetition,
                                 block_order=block_order, instance_order=instance_order,
                                 environment_seed=seed, **episode))
        schedule.append(dict(repetition=repetition, blocks=round_schedule))
    return dict(provenance=PROVENANCE, benchmark="suite", smoke=smoke,
                timing_scope="serial controller plus wrapper/planner; only real physical environment steps excluded",
                aggregation="median repetitions per instance/replicate; equal instances then replicates; per-decision pooling after medians",
                scheduler=dict(algorithm="NumPy PCG64", seed=scheduler_seed),
                platform=platform.platform(), repetitions=repetitions, reset_ordinal=2,
                highs_threads=1, search_cap=2304,
                warmup="Two off-bank episodes per first-round block; one per later-round block; excluded",
                protocol_basis="Recovered serial audit scheduling and warm-up protocol; fresh execution, not archived measurements",
                instance_offsets=list(OFFSETS[:1] if smoke else OFFSETS),
                manifest_blocks=blocks, schedule=schedule, summaries=aggregate_rows(blocks, rows, repetitions), rows=rows)


def validate_output(path):
    output = path.resolve()
    if output.exists() or output == SOURCE_ROOT or SOURCE_ROOT in output.parents:
        raise ValueError("Use a new output path outside the source package")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Version-1 JSON timing blocks; checkpoint paths are relative to this file")
    parser.add_argument("--benchmark", choices=("cash5", "cash10"))
    parser.add_argument("--controller", choices=("ppo", "random", "milp", "raw", "search"))
    parser.add_argument("--interface", help="Cash5: Flat, True, C1-C5; Cash10: flat, mmdp, flat-aa, mmdp-aa")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--random-replicate", type=int, default=0)
    parser.add_argument("--repetitions", type=int, help="Default: manifest repetitions or three")
    parser.add_argument("--scheduler-seed", type=int, help="Independent, recorded block/instance-order RNG seed")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = validate_output(args.output)
        if args.manifest:
            if any(value is not None for value in (args.benchmark, args.controller, args.interface, args.checkpoint, args.sha256)) or args.random_replicate:
                parser.error("Use either --manifest or single-controller options")
            manifest, blocks = read_manifest(args.manifest.resolve())
        else:
            if not all((args.benchmark, args.controller, args.interface)):
                parser.error("Require --manifest or --benchmark/--controller/--interface")
            manifest = {}
            blocks = [dict(id="single", benchmark=args.benchmark, controller=args.controller,
                           interface=args.interface, random_replicate=args.random_replicate,
                           checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                           sha256=args.sha256)]
            # Legacy CLI learns the seed from its explicitly trusted checkpoint.
            if args.controller in ("ppo", "raw", "search"):
                blocks[0]["training_seed"] = 0
            validate_block(blocks[0])
        repetitions = args.repetitions if args.repetitions is not None else manifest.get("repetitions", 3)
        scheduler_seed = args.scheduler_seed if args.scheduler_seed is not None else manifest.get("scheduler_seed", SCHEDULER_SEEDS[blocks[0]["benchmark"]])
        if type(repetitions) is not int or repetitions < 1 or type(scheduler_seed) is not int:
            parser.error("Require positive integer repetitions and an integer scheduler seed")
        if any(b["controller"] in ("ppo", "raw", "search") for b in blocks) and not args.trust_checkpoint:
            parser.error("Learned timing blocks require --trust-checkpoint")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        runtime_imports()
        if not args.manifest and args.controller in ("ppo", "raw", "search"):
            model = load_model(args.checkpoint, args.sha256, args.trust_checkpoint)
            blocks[0]["training_seed"] = model.seed
            validate_block(blocks[0])
            # Reuse the loaded model while retaining the same space/seed checks.
            models = preload_models(blocks, args.trust_checkpoint,
                                    cache={(blocks[0]["checkpoint"], blocks[0]["sha256"]): model})
        else:
            models = preload_models(blocks, args.trust_checkpoint)
        with single_thread_highs():
            result = run_suite(blocks, models, repetitions, scheduler_seed, args.smoke)
        import numpy
        import scipy
        import stable_baselines3
        result["versions"] = dict(numpy=numpy.__version__, scipy=scipy.__version__,
                                  torch=torch.__version__, stable_baselines3=stable_baselines3.__version__,
                                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads())
        result["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.manifest:
            result["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        else:
            result.update(benchmark=args.benchmark, controller=args.controller, interface=args.interface,
                          instances=result["summaries"]["blocks"][0]["instances"],
                          mean_of_instance_medians_ms_per_episode=result["summaries"]["blocks"][0]["mean_of_instance_medians_ms_per_episode"])
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("rows", "schedule", "manifest_blocks")}, allow_nan=False))


if __name__ == "__main__":
    main()
