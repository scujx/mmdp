"""Action abstraction: a zero amount executes a zero transfer, not STOP."""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MMDPCashGymEdgeOnlyEnvV2(gym.Env):
    """
    Edge-only MMDP wrapper for CashFlowEnvV2.

    Clean design:
      - stage 0 action space = expiring edges only
      - stage 1 action space = persistent edges only
      - agent outputs only a discrete edge choice
      - STOP is always available
      - transfer amount is computed by base_env heuristic executor
      - wrapper converts heuristic amount into ratio for base_env.step((edge_idx, ratio))

    Important:
      - fixed-size Discrete(max_stage_edge_count + 1)
      - indices >= current_edge_count are interpreted as STOP
      - no modulo remapping
      - zero/small heuristic amount is treated as STOP
    """
    metadata = {'render_modes': []}

    def __init__(self, base_env, min_amount_threshold: float=1e-06):
        super().__init__()
        super().__init__()
        self.base_env = base_env
        self.min_amount_threshold = float(min_amount_threshold)
        self.edge_only_wrapper = True
        self.base_env.reset()
        stage1_all = list(self.base_env.stage1_edges)
        stage2_persistent = list(self.base_env.stage2_edges)
        stage2_set = set(stage2_persistent)
        self.stage0_edges = sorted([e for e in stage1_all if e not in stage2_set], key=lambda x: (x[0], x[1]))
        self.stage1_edges = sorted(stage2_persistent, key=lambda x: (x[0], x[1]))
        obs = self._build_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
        self.max_stage_edge_count = max(len(self.stage0_edges), len(self.stage1_edges))
        self.action_space = spaces.Discrete(self.max_stage_edge_count + 1)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            pass
        self.base_env.reset()
        obs = self._build_obs()
        info = {'mmdp_stage': int(self.base_env.stage), 'stage0_edge_count': len(self.stage0_edges), 'stage1_edge_count': len(self.stage1_edges), 'current_mmdp_edge_count': len(self._get_mmdp_edges_for_current_stage()), 'edge_only_wrapper': True}
        return (obs, info)

    def step(self, action):
        stage_before = int(self.base_env.stage)
        full_current_edges = self.base_env.get_current_edges()
        mmdp_edges = self._get_mmdp_edges_for_current_stage()
        local_idx = self._decode_action(action, len(mmdp_edges))
        if local_idx >= len(mmdp_edges):
            stop_ratio = 1.0
            (_, reward, done, info) = self.base_env.step((len(full_current_edges), stop_ratio))
            decoded_edge = 'STOP'
            decoded_full_edge_index = len(full_current_edges)
            wrapper_stop_selected = True
            heuristic_amount = 0.0
            heuristic_ratio = stop_ratio
            edge_selected_but_skipped = False
        else:
            chosen_edge = mmdp_edges[local_idx]
            if chosen_edge not in full_current_edges:
                raise RuntimeError(f'MMDP edge-only wrapper selected edge {chosen_edge}, but it is not valid in base env stage {self.base_env.stage}. Current base edges = {full_current_edges}')
            (source, target) = chosen_edge
            full_idx = full_current_edges.index(chosen_edge)
            heuristic_amount = self.base_env.compute_heuristic_amount_for_edge(source=source, target=target, stage_aware=True)
            if heuristic_amount <= self.min_amount_threshold:
                (_, reward, done, info) = self.base_env.step((full_idx, 0.0))
                decoded_edge = chosen_edge
                decoded_full_edge_index = int(full_idx)
                wrapper_stop_selected = False
                heuristic_ratio = 0.0
                edge_selected_but_skipped = False
            else:
                heuristic_ratio = self._amount_to_ratio(source, target, heuristic_amount)
                (_, reward, done, info) = self.base_env.step((full_idx, heuristic_ratio))
                decoded_edge = chosen_edge
                decoded_full_edge_index = int(full_idx)
                wrapper_stop_selected = False
                edge_selected_but_skipped = False
        if done:
            next_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            terminated = True
            truncated = False
        else:
            next_obs = self._build_obs()
            terminated = False
            truncated = False
        info['decoded_edge_index_local'] = int(local_idx)
        info['decoded_edge_index_full'] = int(decoded_full_edge_index)
        info['decoded_edge'] = decoded_edge
        info['wrapper_stop_selected'] = bool(wrapper_stop_selected)
        info['heuristic_amount'] = float(heuristic_amount)
        info['heuristic_ratio'] = float(heuristic_ratio)
        info['edge_selected_but_skipped'] = bool(edge_selected_but_skipped)
        info['mmdp_stage'] = int(info.get('stage_before', stage_before))
        info['mmdp_stage_after'] = int(self.base_env.stage)
        info['mmdp_edge_count'] = len(mmdp_edges)
        info['stage0_edge_count'] = len(self.stage0_edges)
        info['stage1_edge_count'] = len(self.stage1_edges)
        info['transition_reason'] = info.get('transition_reason', None)
        info['stop_count_stage0'] = int(getattr(self.base_env, 'stop_count_stage0', 0))
        info['stop_count_stage1'] = int(getattr(self.base_env, 'stop_count_stage1', 0))
        info['total_stop_count'] = int(getattr(self.base_env, 'total_stop_count', 0))
        info['zero_transfer_count'] = int(getattr(self.base_env, 'zero_transfer_count', 0))
        info['stage0_steps_used'] = int(getattr(self.base_env, 'stage0_steps_used', 0))
        info['stage1_steps_used'] = int(getattr(self.base_env, 'stage1_steps_used', 0))
        transition_history = getattr(self.base_env, 'transition_history', [])
        info['transition_history'] = list(transition_history)
        info['is_zero_transfer'] = bool(info.get('transfer_amount', 0.0) <= 1e-08 and (not info.get('stop_action', False)))
        info['episode_done'] = bool(done)
        info['edge_only_wrapper'] = True
        if done:
            ep_summary = self.base_env.episode_summary()
            info['episode_summary'] = ep_summary
            info['ep_stop_count_stage0'] = int(ep_summary.get('stop_count_stage0', 0))
            info['ep_stop_count_stage1'] = int(ep_summary.get('stop_count_stage1', 0))
            info['ep_total_stop_count'] = int(ep_summary.get('total_stop_count', 0))
            info['ep_zero_transfer_count'] = int(ep_summary.get('zero_transfer_count', 0))
            info['ep_stage0_steps_used'] = int(ep_summary.get('stage0_steps_used', 0))
            info['ep_stage1_steps_used'] = int(ep_summary.get('stage1_steps_used', 0))
            info['ep_total_transfer_amount'] = float(ep_summary.get('total_transfer_amount', 0.0))
            info['ep_total_transfer_fee'] = float(ep_summary.get('total_transfer_fee', 0.0))
            info['ep_terminal_yield'] = float(ep_summary.get('terminal_yield', 0.0))
            info['ep_terminal_gap_penalty'] = float(ep_summary.get('terminal_gap_penalty', 0.0))
            info['ep_terminal_reward'] = float(ep_summary.get('terminal_reward', 0.0)) if ep_summary.get('terminal_reward', None) is not None else None
            info['ep_transition_history'] = ep_summary.get('transition_history', [])
        return (next_obs, float(reward), terminated, truncated, info)

    def render(self):
        return None

    def close(self):
        return None

    def _build_obs(self) -> np.ndarray:
        obs_dict = self.base_env.get_observation_dict()
        stage = int(obs_dict['stage'])
        step_in_stage = int(obs_dict['step_in_stage'])
        balances = np.asarray(obs_dict['balances'], dtype=np.float32)
        pred = np.asarray(obs_dict['predicted_remaining_outflow'], dtype=np.float32)
        stage_onehot = np.zeros(2, dtype=np.float32)
        stage_onehot[stage] = 1.0
        max_steps = float(self.base_env.cfg.stage1_max_steps) if stage == 0 else float(self.base_env.cfg.stage2_max_steps)
        step_frac = np.array([step_in_stage / max(max_steps, 1.0)], dtype=np.float32)
        scale_cash = 100.0
        balances_norm = balances / scale_cash
        pred_norm = pred / scale_cash
        gap = np.maximum(pred - balances, 0.0)
        surplus = np.maximum(balances - pred, 0.0)
        gap_norm = gap / scale_cash
        surplus_norm = surplus / scale_cash
        shock_target = int(obs_dict.get('shock_target_account', -1))
        shock_size = float(obs_dict.get('shock_size', 0.0))
        operational_indices = tuple(self.base_env.cfg.operational_indices)
        shock_onehot = np.zeros(1 + len(operational_indices), dtype=np.float32)
        if shock_target == -1:
            shock_onehot[0] = 1.0
        else:
            local_map = {acct_idx: i + 1 for (i, acct_idx) in enumerate(operational_indices)}
            shock_onehot[local_map[shock_target]] = 1.0
        shock_size_arr = np.array([shock_size / scale_cash], dtype=np.float32)
        obs = np.concatenate([stage_onehot, step_frac, balances_norm, pred_norm, gap_norm, surplus_norm, shock_onehot, shock_size_arr], axis=0).astype(np.float32)
        return obs

    def _get_mmdp_edges_for_current_stage(self):
        if self.base_env.stage == 0:
            return self.stage0_edges
        return self.stage1_edges

    def _decode_action(self, action, num_current_edges: int) -> int:
        """
        Fixed Discrete action space:
          - 0 .. max_stage_edge_count-1 : local edge slot
          - max_stage_edge_count        : explicit STOP
          - overflow                    : STOP
        """
        a = int(action)
        if a < 0:
            return num_current_edges
        if a >= num_current_edges:
            return num_current_edges
        return a

    def _amount_to_ratio(self, source: int, target: int, amount: float) -> float:
        amount = float(max(amount, 0.0))
        pred = self.base_env._current_predicted_remaining_outflow()
        balances = self.base_env.balances
        if balances is None:
            return 0.0
        own_need = float(pred[source])
        source_balance = float(balances[source])
        source_surplus = max(source_balance - own_need, 0.0)
        if target == self.base_env.cfg.investment_idx:
            denom = source_surplus
        else:
            target_gap = max(float(pred[target]) - float(balances[target]), 0.0)
            denom = min(source_surplus, target_gap)
        if denom <= 1e-08:
            return 0.0
        ratio = amount / denom
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return ratio

    def get_snapshot(self):
        return self.base_env.get_snapshot()

    def load_snapshot(self, snapshot):
        self.base_env.load_snapshot(snapshot)

    def export_planner_state(self):
        return self.get_snapshot()

    def load_planner_state(self, snapshot):
        self.load_snapshot(snapshot)
