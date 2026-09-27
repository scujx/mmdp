"""Training and evaluation helpers; no training, evaluation, or file writes on import."""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from .envs.cash_flow_env_v2 import CashFlowEnvConfigV2, CashFlowEnvV2
from .wrappers.flat_cash_gym_env_v2 import FlatCashGymEnvV2
from .wrappers.mmdp_cash_gym_env_v2 import MMDPCashGymEnvV2
from .wrappers.flat_aa import FlatCashGymEdgeOnlyEnvV2
from .wrappers.mmdp_aa import MMDPCashGymEdgeOnlyEnvV2
from . import search

WRAPPERS = {"flat": FlatCashGymEnvV2, "mmdp": MMDPCashGymEnvV2,
            "flat-aa": FlatCashGymEdgeOnlyEnvV2, "mmdp-aa": MMDPCashGymEdgeOnlyEnvV2}
FAMILIES = {"flat": "flat_ppo", "mmdp": "mmdp_ppo",
            "flat-aa": "flat_aa_ppo", "mmdp-aa": "mmdp_aa_ppo"}
CONFIG = json.loads((Path(__file__).resolve().parent / "configs/experiment.json").read_text())
CAPS = (0, 576, 1152, 2304)


def make_env(policy, seed):
    # The wrapper performs reset 1; the first explicit reset starts instance 2.
    return WRAPPERS[policy](CashFlowEnvV2(CashFlowEnvConfigV2(seed=int(seed))))


def make_model(policy, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = VecMonitor(DummyVecEnv([lambda: make_env(policy, seed)]))
    return PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
               **copy.deepcopy(CONFIG["configuration"]))


def public_state(env):
    """Only observation-visible values enter the planner; no latent snapshot."""
    p = env.base_env.get_observation_dict()
    return search.VisibleState(
        stage=int(p["stage"]), step_in_stage=int(p["step_in_stage"]),
        balances=tuple(float(x) for x in p["balances"]),
        visible_forecast=tuple(float(x) for x in p["predicted_remaining_outflow"]),
        current_routes=tuple(tuple(x) for x in env._get_mmdp_edges_for_current_stage()),
        shock_target_account=int(p["shock_target_account"]), shock_size=float(p["shock_size"]))


def _install_strict_search_checks():
    """Validate policy probabilities and candidate-count limits."""
    score_reader, candidate_builder = search._discrete_scores, search.build_candidates

    def checked_scores(model, obs, ledger):
        values = score_reader(model, obs, ledger)
        assert values is not None and values.ndim == 1 and np.isfinite(values).all()
        assert (values >= 0).all() and abs(values.sum() - 1) < 1e-5
        return values

    def checked_candidates(model, obs, state, method_spec, ledger):
        candidates, meta = candidate_builder(model, obs, state, method_spec, ledger)
        assert len(candidates) <= method_spec.max_candidates
        assert len({x.semantic_key() for x in candidates}) == len(candidates)
        assert sum(x.kind == "stop" for x in candidates) == 1
        if method_spec.discrete and len(state.current_routes) + 1 > method_spec.max_candidates:
            assert meta["scores"] is not None and len(meta["scores"]) >= len(state.current_routes)
        return candidates, meta

    search._discrete_scores, search.build_candidates = checked_scores, checked_candidates


_install_strict_search_checks()


def evaluate_episode(model, policy, seed, cap=0):
    """Evaluate a complete episode, pairing Raw/Search by construction seed."""
    if cap not in CAPS:
        raise ValueError("Use Raw (0), or planning cap 576, 1152, 2304")
    env = make_env(policy, seed)
    was_training = model.policy.training
    totals = dict(episode_return=0., transfer_fee=0., step_penalty=0.,
                  zero_transfer_penalty=0., terminal_yield=0., terminal_gap_penalty=0.)
    planning = max_planning = forwards = 0
    try:
        obs, _ = env.reset()
        initial_observation = obs.tolist()
        for length in range(1, 35):
            if cap:
                state = public_state(env)
                selected = search.select_dfm_q9_action(
                    model=model, current_obs=obs.copy(), state=state,
                    method_spec=search.METHOD_SPECS[FAMILIES[policy]],
                    wrapper_factory=WRAPPERS[policy], transition_cap=cap)
                ledger = selected.ledger.public_record()
                assert ledger["planning_transitions"] <= cap
                assert ledger["scenario_worlds_scored"] == ledger["candidates_scored"] * (9 if state.stage == 0 else 1)
                assert ledger["terminal_transitions"] == ledger["scenario_worlds_scored"]
                assert ledger["candidates_generated"] == ledger["candidates_scored"] + ledger["candidates_skipped_cap"]
                planning += ledger["planning_transitions"]
                forwards += ledger["policy_forwards"]
                max_planning = max(max_planning, ledger["planning_transitions"])
                action = selected.action
            else:
                action, _ = model.predict(obs, deterministic=True)
            if not env.action_space.contains(action):
                raise RuntimeError("Policy output does not belong to the wrapper's action space")
            obs, reward, done, truncated, info = env.step(action)
            totals["episode_return"] += float(reward)
            for key in ("transfer_fee", "terminal_yield", "terminal_gap_penalty"):
                totals[key] += float(info.get(key, 0.))
            totals["step_penalty"] += 0. if info["stop_action"] else env.base_env.cfg.step_penalty
            totals["zero_transfer_penalty"] += env.base_env.cfg.zero_transfer_penalty * int(not info["stop_action"] and info.get("transfer_amount", 0.) <= 1e-8)
            if done or truncated:
                if truncated or not done:
                    raise RuntimeError("Expected an entire terminated episode")
                break
        else:
            raise RuntimeError("Episode exceeded the documented horizon")
        reconstructed = totals["terminal_yield"] - totals["terminal_gap_penalty"] - totals["transfer_fee"] - totals["step_penalty"] - totals["zero_transfer_penalty"]
        error = abs(totals["episode_return"] - reconstructed)
        if error >= 1e-8:
            raise RuntimeError("Episode reward accounting mismatch")
        return dict(policy=policy, environment_seed=seed, reset_ordinal=2, planning_cap=cap,
                    episode_length=length, planning_transitions=planning,
                    planning_policy_forwards=forwards, max_planning_transitions_per_decision=max_planning,
                    initial_observation_sha256=hashlib.sha256(json.dumps(initial_observation).encode()).hexdigest(),
                    reward_reconstruction_error=error, **totals)
    finally:
        model.policy.set_training_mode(was_training)
        env.close()
