from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlatCashGymEnv(gym.Env):
    """
    Flat MDP wrapper for the five-account cash-flow environment.

    Design:
      - stage 0: flat agent searches over the union of stage1/stage2 edges
      - stage 1: flat agent searches over stage2 valid edges only
      - STOP is always available
      - no invalid-edge penalty trick

    Compared with MMDP:
      - Flat stage 0 must already search over future-useful edges as well
      - MMDP stage 0 only searches over stage1 admissible edges

    Observation is aligned with MMDP:
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

        # stage-0 flat action set = union of both stages

        self.stage0_edges = list(self.base_env.stage1_edges)
        self.stage1_edges = list(self.base_env.stage2_edges)

        obs = self._build_obs(reset=True)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        # continuous action:
        # action[0] in [-1,1] -> edge selector over current edge set + STOP
        # action[1] in [-1,1] -> transfer ratio in [0,1]
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
            pass

        self.base_env.reset()
        obs = self._build_obs(reset=False)
        info = {
            "flat_stage": int(self.base_env.stage),
            "stage0_edge_count": len(self.stage0_edges),
            "stage1_edge_count": len(self.stage1_edges),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        current_edges = self._get_flat_current_edges()
        edge_idx, ratio = self._decode_action(action, num_edges=len(current_edges))

        # STOP
        if edge_idx == len(current_edges):
            _, reward, done, info = self.base_env.step(
                (len(self.base_env.get_current_edges()), ratio)
            )
            next_obs, terminated, truncated = self._format_transition(done)

            info["decoded_edge_index"] = int(edge_idx)
            info["decoded_edge"] = "STOP"
            info["decoded_ratio"] = float(ratio)
            info["flat_stage"] = int(info.get("stage_before", self.base_env.stage))
            info["flat_edge_count"] = len(current_edges)
            return next_obs, float(reward), terminated, truncated, info

        chosen_edge = current_edges[edge_idx]

        # chosen edge must be executable in base env CURRENT stage
        base_current_edges = self.base_env.get_current_edges()
        if chosen_edge not in base_current_edges:
            raise RuntimeError(
                f"Flat wrapper selected edge {chosen_edge}, but it is not valid in base env stage {self.base_env.stage}. "
                f"Current base edges = {base_current_edges}"
            )

        local_edge_idx = base_current_edges.index(chosen_edge)
        _, reward, done, info = self.base_env.step((local_edge_idx, ratio))

        next_obs, terminated, truncated = self._format_transition(done)
        info["decoded_edge_index"] = int(edge_idx)
        info["decoded_edge"] = chosen_edge
        info["decoded_ratio"] = float(ratio)
        info["flat_stage"] = int(info.get("stage_before", self.base_env.stage))
        info["flat_edge_count"] = len(current_edges)
        return next_obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None

    # =========================================================
    # Edge sets
    # =========================================================
    def _get_flat_current_edges(self):
        if self.base_env.stage == 0:
            return self.stage0_edges
        return self.stage1_edges

    # =========================================================
    # Observation
    # =========================================================
    def _build_obs(self, reset: bool = False) -> np.ndarray:
        if reset:
            self.base_env.reset()

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

        # shock feature
        shock_target = int(obs_dict.get("shock_target_account", -1))
        shock_size = float(obs_dict.get("shock_size", 0.0))

        shock_onehot = np.zeros(4, dtype=np.float32)  # [none, A2, A3, A4]
        if shock_target == -1:
            shock_onehot[0] = 1.0
        else:
            local_map = {2: 1, 3: 2, 4: 3}
            shock_onehot[local_map[shock_target]] = 1.0

        shock_size_arr = np.array([shock_size / scale_cash], dtype=np.float32)

        return np.concatenate(
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

    # =========================================================
    # Action decoding
    # =========================================================
    def _decode_action(self, action: np.ndarray, num_edges: int):
        a0 = float(np.clip(action[0], -1.0, 1.0))
        a1 = float(np.clip(action[1], -1.0, 1.0))

        num_choices = num_edges + 1  # +1 for STOP
        selector = (a0 + 1.0) / 2.0
        edge_idx = int(np.floor(selector * num_choices))
        edge_idx = min(edge_idx, num_choices - 1)

        ratio = (a1 + 1.0) / 2.0
        ratio = float(np.clip(ratio, 0.0, 1.0))

        return edge_idx, ratio

    # =========================================================
    # Transition formatting
    # =========================================================
    def _format_transition(self, done: bool):
        if done:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            terminated = True
            truncated = False
        else:
            next_obs = self._build_obs(reset=False)
            terminated = False
            truncated = False
        return next_obs, terminated, truncated
