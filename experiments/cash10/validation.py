"""Reconstructed, observational Raw validation; never select a checkpoint."""
import csv
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from . import runtime


def evaluate_raw_episode(model, policy, seed):
    """Score reset two, retaining the archived curve's float32 step sum."""
    env = runtime.make_env(policy, seed)
    native_return = 0.
    curve_return = np.float32(0.)
    try:
        observation, _ = env.reset()
        for length in range(1, 35):
            action, _ = model.predict(observation, deterministic=True)
            if not env.action_space.contains(action):
                raise RuntimeError("Validation action is outside the wrapper's action space")
            observation, reward, done, truncated, _ = env.step(action)
            native_return += float(reward)
            curve_return = np.float32(curve_return + np.float32(reward))
            if done or truncated:
                if truncated or not done:
                    raise RuntimeError("Validation requires a complete terminated episode")
                return dict(curve_return=float(curve_return), native_return=native_return,
                            episode_length=length)
        raise RuntimeError("Validation episode exceeded the documented horizon")
    finally:
        env.close()


@contextmanager
def preserve_training_state(model):
    """Isolate evaluation's global randomness and policy mode from training."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    modes = [(module, module.training) for module in model.policy.modules()]
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        # Restore each module, including any deliberately mixed train/eval modes.
        for module, training in modes:
            module.training = training


class RawValidationCallback(BaseCallback):
    """Record a fixed bank on absolute timesteps across consecutive learn calls.

    The implementation follows the disclosed grid and reset convention. It is
    newly reconstructed, not a recovered historical training callback. Returns
    use the float32 per-step accumulation found in the archived fixed-bank
    callback, alongside native sums retained for transparent reward accounting.
    """
    fields = ("policy", "training_seed", "timesteps", "mean_return", "return_sd",
              "native_mean_return", "mean_episode_length", "episodes", "smoke_only")

    def __init__(self, policy, training_seed, output, *, smoke_only=False,
                 interval=5_000, final_timestep=2_000_000):
        super().__init__()
        if interval < 1 or final_timestep < interval or final_timestep % interval:
            raise ValueError("Validation requires a positive, complete timestep grid")
        if not smoke_only and (interval, final_timestep) != (5_000, 2_000_000):
            raise ValueError("Alternative validation grids are smoke-only")
        self.policy_name = policy
        self.training_seed = int(training_seed)
        self.environment_seeds = tuple(self.training_seed + 1000 + j for j in range(20))
        self.interval = interval
        self.final_timestep = final_timestep
        self.smoke_only = smoke_only
        self.next_timestep = interval
        self.points_recorded = 0
        self.csv_path = Path(output) / "validation.csv"
        with self.csv_path.open("x", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def _on_step(self):
        step = self.model.num_timesteps
        if self.next_timestep > self.final_timestep or step < self.next_timestep:
            return True
        if step != self.next_timestep:
            raise RuntimeError("Validation skipped an absolute timestep; use one training environment")
        with preserve_training_state(self.model):
            episodes = [evaluate_raw_episode(self.model, self.policy_name, seed)
                        for seed in self.environment_seeds]
        returns = [episode["curve_return"] for episode in episodes]
        row = dict(policy=self.policy_name, training_seed=self.training_seed,
                   timesteps=step, mean_return=float(np.mean(returns)),
                   return_sd=float(np.std(returns, ddof=0)),
                   native_mean_return=float(np.mean([episode["native_return"] for episode in episodes])),
                   mean_episode_length=float(np.mean([episode["episode_length"] for episode in episodes])),
                   episodes=len(returns),
                   smoke_only=self.smoke_only)
        with self.csv_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)
        self.points_recorded += 1
        self.next_timestep += self.interval
        return True

    def metadata(self):
        return dict(
            implementation="reconstructed_disclosed_validation_protocol",
            historical_callback_recovered=False,
            file=self.csv_path.name, deterministic=True, planning_cap=0,
            environment_seeds=list(self.environment_seeds), reset_ordinal=2,
            interval=self.interval, first_timestep=self.interval,
            last_timestep=self.final_timestep,
            expected_points=self.final_timestep // self.interval,
            points_recorded=self.points_recorded, absolute_timestep_grid=True,
            shared_across_training_segments=True, checkpoint_selection=False,
            initialization_evaluation=False, smoke_only=self.smoke_only,
            return_accumulation="float32_per_step_sum_then_float64_bank_mean",
            native_return_also_recorded=True, within_bank_sd_ddof=0,
            historical_reward_precision_verified=True,
            precision_evidence="archived cash10 historical_core.py evaluate_episode and ValidationCallback",
            note="Fresh reconstruction of the disclosed grid; does not recover historical validation values.")
