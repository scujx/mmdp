from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MMDPCashGymEnv(gym.Env):
    """
    Stage-structured MMDP wrapper for the five-account cash-flow environment.

    Design:
      - stage 0 action space = stage0_edges (expiring edges only)
      - stage 1 action space = stage1_edges (persistent edges only)
      - STOP is always available

    Observation:
      [stage_onehot(2),
       step_frac(1),
       balances(5),
       pred(5),
       gap(5),
       surplus(5),
       shock_onehot(4),
       shock_size(1)]
    """

    metadata = {"render_modes": []}

    def __init__(self, base_env):
        super().__init__()
        self.base_env = base_env

        self.base_env.reset()

        stage1_all = list(self.base_env.stage1_edges)
        stage2_persistent = list(self.base_env.stage2_edges)
        stage2_set = set(stage2_persistent)

        # theorem-aligned split:
        # stage 0 handles expiring edges only
        # stage 1 handles persistent edges only
        self.stage0_edges = [e for e in stage1_all if e not in stage2_set]
        self.stage1_edges = stage2_persistent

        obs = self._build_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        # continuous 2-d action:
        #   action[0] -> edge selector
        #   action[1] -> ratio
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    # =========================================================
    # Gym API
    # =========================================================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # base_env itself already uses config seed;
            # do not silently override unless you want a fuller refactor
            pass

        self.base_env.reset()
        obs = self._build_obs()

        info = {
            "mmdp_stage": int(self.base_env.stage),
            "stage0_edge_count": len(self.stage0_edges),
            "stage1_edge_count": len(self.stage1_edges),
            "current_mmdp_edge_count": len(self._get_mmdp_edges_for_current_stage()),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        stage_before = int(self.base_env.stage)
        full_current_edges = self.base_env.get_current_edges()
        mmdp_edges = self._get_mmdp_edges_for_current_stage()

        local_idx, ratio = self._decode_action(action, len(mmdp_edges))

        # STOP
        if local_idx == len(mmdp_edges):
            _, reward, done, info = self.base_env.step((len(full_current_edges), ratio))
            decoded_edge = "STOP"
            decoded_full_edge_index = len(full_current_edges)
            stop_selected_by_wrapper = True
        else:
            chosen_edge = mmdp_edges[local_idx]
            if chosen_edge not in full_current_edges:
                raise RuntimeError(
                    f"MMDP wrapper selected edge {chosen_edge}, but it is not valid in "
                    f"base env stage {self.base_env.stage}. Current base edges = {full_current_edges}"
                )
            full_idx = full_current_edges.index(chosen_edge)
            _, reward, done, info = self.base_env.step((full_idx, ratio))
            decoded_edge = chosen_edge
            decoded_full_edge_index = int(full_idx)
            stop_selected_by_wrapper = False

        if done:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            terminated = True
            truncated = False
        else:
            next_obs = self._build_obs()
            terminated = False
            truncated = False

        # -----------------------------------------------------
        # Wrapper decode info
        # -----------------------------------------------------
        info["decoded_edge_index_local"] = int(local_idx)
        info["decoded_edge_index_full"] = int(decoded_full_edge_index)
        info["decoded_edge"] = decoded_edge
        info["decoded_ratio"] = float(ratio)
        info["wrapper_stop_selected"] = bool(stop_selected_by_wrapper)

        info["mmdp_stage"] = int(info.get("stage_before", stage_before))
        info["mmdp_stage_after"] = int(self.base_env.stage if not done else 1)
        info["mmdp_edge_count"] = len(mmdp_edges)
        info["stage0_edge_count"] = len(self.stage0_edges)
        info["stage1_edge_count"] = len(self.stage1_edges)

        # -----------------------------------------------------
        # Pass through base-environment diagnostics.
        # -----------------------------------------------------
        info["transition_reason"] = info.get("transition_reason", None)

        info["stop_count_stage0"] = int(getattr(self.base_env, "stop_count_stage0", 0))
        info["stop_count_stage1"] = int(getattr(self.base_env, "stop_count_stage1", 0))
        info["total_stop_count"] = int(getattr(self.base_env, "total_stop_count", 0))

        info["zero_transfer_count"] = int(
            getattr(self.base_env, "zero_transfer_count", 0)
        )

        info["stage0_steps_used"] = int(getattr(self.base_env, "stage0_steps_used", 0))
        info["stage1_steps_used"] = int(getattr(self.base_env, "stage1_steps_used", 0))

        transition_history = getattr(self.base_env, "transition_history", [])
        info["transition_history"] = list(transition_history)

        # helpful per-step flags
        info["is_zero_transfer"] = bool(
            info.get("transfer_amount", 0.0) <= 1e-8
            and not info.get("stop_action", False)
        )
        info["episode_done"] = bool(done)

        # -----------------------------------------------------
        # When episode finishes, attach full episode summary
        # -----------------------------------------------------
        if done:
            ep_summary = self.base_env.episode_summary()

            # keep nested summary for debugging / later parsing
            info["episode_summary"] = ep_summary

            # also flatten a few load-bearing fields for convenience
            info["ep_stop_count_stage0"] = int(ep_summary.get("stop_count_stage0", 0))
            info["ep_stop_count_stage1"] = int(ep_summary.get("stop_count_stage1", 0))
            info["ep_total_stop_count"] = int(ep_summary.get("total_stop_count", 0))
            info["ep_zero_transfer_count"] = int(
                ep_summary.get("zero_transfer_count", 0)
            )
            info["ep_stage0_steps_used"] = int(ep_summary.get("stage0_steps_used", 0))
            info["ep_stage1_steps_used"] = int(ep_summary.get("stage1_steps_used", 0))
            info["ep_total_transfer_amount"] = float(
                ep_summary.get("total_transfer_amount", 0.0)
            )
            info["ep_total_transfer_fee"] = float(
                ep_summary.get("total_transfer_fee", 0.0)
            )
            info["ep_terminal_yield"] = float(ep_summary.get("terminal_yield", 0.0))
            info["ep_terminal_gap_penalty"] = float(
                ep_summary.get("terminal_gap_penalty", 0.0)
            )
            info["ep_terminal_reward"] = (
                float(ep_summary.get("terminal_reward", 0.0))
                if ep_summary.get("terminal_reward", None) is not None
                else None
            )
            info["ep_transition_history"] = ep_summary.get("transition_history", [])

        return next_obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None

    # =========================================================
    # Observation
    # =========================================================
    def _build_obs(self) -> np.ndarray:
        obs_dict = self.base_env.get_observation_dict()

        stage = int(obs_dict["stage"])
        step_in_stage = int(obs_dict["step_in_stage"])
        balances = np.asarray(obs_dict["balances"], dtype=np.float32)
        pred = np.asarray(obs_dict["predicted_remaining_outflow"], dtype=np.float32)

        stage_onehot = np.zeros(2, dtype=np.float32)
        stage_onehot[stage] = 1.0

        max_steps = (
            float(self.base_env.cfg.stage1_max_steps)
            if stage == 0
            else float(self.base_env.cfg.stage2_max_steps)
        )
        step_frac = np.array([step_in_stage / max(max_steps, 1.0)], dtype=np.float32)

        scale_cash = 100.0
        balances_norm = balances / scale_cash
        pred_norm = pred / scale_cash

        gap = np.maximum(pred - balances, 0.0)
        surplus = np.maximum(balances - pred, 0.0)
        gap_norm = gap / scale_cash
        surplus_norm = surplus / scale_cash

        shock_target = int(obs_dict.get("shock_target_account", -1))
        shock_size = float(obs_dict.get("shock_size", 0.0))

        shock_onehot = np.zeros(4, dtype=np.float32)  # [none, A2, A3, A4]
        if shock_target == -1:
            shock_onehot[0] = 1.0
        else:
            local_map = {2: 1, 3: 2, 4: 3}
            shock_onehot[local_map[shock_target]] = 1.0

        shock_size_arr = np.array([shock_size / scale_cash], dtype=np.float32)

        obs = np.concatenate(
            [
                stage_onehot,
                step_frac,
                balances_norm,
                pred_norm,
                gap_norm,
                surplus_norm,
                shock_onehot,
                shock_size_arr,
            ],
            axis=0,
        ).astype(np.float32)

        return obs

    # =========================================================
    # Edge sets
    # =========================================================
    def _get_mmdp_edges_for_current_stage(self):
        if self.base_env.stage == 0:
            return self.stage0_edges
        return self.stage1_edges

    # =========================================================
    # Action decoding
    # =========================================================
    def _decode_action(self, action: np.ndarray, num_current_edges: int):
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))

        # edge selector
        num_choices = num_current_edges + 1  # +1 for STOP
        selector = (a0 + 1.0) / 2.0
        edge_idx = int(np.floor(selector * num_choices))
        edge_idx = min(edge_idx, num_choices - 1)

        # ratio in [0, 1]
        ratio = (a1 + 1.0) / 2.0
        ratio = float(np.clip(ratio, 0.0, 1.0))

        return edge_idx, ratio

    # =========================================================
    # Snapshot / Restore passthrough
    # =========================================================
    def get_snapshot(self):
        return self.base_env.get_snapshot()

    def load_snapshot(self, snapshot):
        self.base_env.load_snapshot(snapshot)

    def export_planner_state(self):
        return self.get_snapshot()

    def load_planner_state(self, snapshot):
        self.load_snapshot(snapshot)
