from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from .cash_flow_generator import CashFlowGeneratorConfig, SyntheticCashFlowGenerator


@dataclass
class CashFlowEnvConfig:
    # =========================================================
    # Basic structure
    # =========================================================
    seed: int = 42
    num_accounts: int = 5
    stage1_max_steps: int = 10
    stage2_max_steps: int = 10

    # Account roles
    # 0 = master, 1 = investment, 2/3/4 = operational
    master_idx: int = 0
    investment_idx: int = 1
    operational_indices: Tuple[int, int, int] = (2, 3, 4)

    # =========================================================
    # Initial balances
    # =========================================================
    init_balance_low: Tuple[float, ...] = (140.0, 20.0, 25.0, 20.0, 20.0)
    init_balance_high: Tuple[float, ...] = (180.0, 40.0, 45.0, 40.0, 35.0)

    # =========================================================
    # Reward settings
    # =========================================================
    transfer_fee_rate: float = 0.001
    step_penalty: float = 20.0

    # End-of-day yield rates for positive balances
    account_yield_rate: Tuple[float, ...] = (0.003, 0.010, 0.001, 0.001, 0.001)

    # Gap penalty rates for negative balances
    account_gap_penalty: Tuple[float, ...] = (4.0, 2.0, 10.0, 8.0, 8.0)

    yield_weight: float = 1.0
    gap_weight: float = 1.0

    # Optional shaping for moving money into investment
    investment_shaping_weight: float = 0.0

    # =========================================================
    # Stage-specific edge sets
    # Stage 1: master <-> ops, master <-> investment
    # Stage 2: ops fully connected, master <-> investment
    # =========================================================
    stage1_edges: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (1, 0),  # master <-> investment
        (0, 2),
        (2, 0),  # master <-> A2
        (0, 3),
        (3, 0),  # master <-> A3
        (0, 4),
        (4, 0),  # master <-> A4
        (2, 3),
        (3, 2),  # ops fully connected
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    )

    stage2_edges: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (1, 0),  # master <-> investment
        (2, 3),
        (3, 2),  # ops fully connected
        (2, 4),
        (4, 2),
        (3, 4),
        (4, 3),
    )


class CashFlowEnv:
    """
    Five-account two-stage cash-flow-control environment.

    Structure:
      - One trajectory = one day
      - Two decision stages:
          stage 0: seasonal forecast only
          stage 1: localized demand shock revealed
      - Demand is settled only once at the very end
      - Transfers arrive immediately
      - End-of-day reward = yield on positive balances - shortage penalty on negative balances

    Action:
      (edge_index, ratio)
        - edge_index in [0, len(current_edges)] where len(current_edges) means STOP
        - ratio in [0, 1]
    """

    def __init__(self, config: CashFlowEnvConfig = CashFlowEnvConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)

        self.account_names = ["A0_master", "A1_invest", "A2_op", "A3_op", "A4_op"]

        # dynamic state
        self.stage = 0
        self.step_in_stage = 0
        self.done = False
        self.balances: np.ndarray | None = None

        # generator outputs
        self.base_total_demand: np.ndarray | None = None
        self.true_total_outflow: np.ndarray | None = None
        self.seasonal_base_forecast: np.ndarray | None = None
        self.updated_total_forecast: np.ndarray | None = None
        self.shock_target_account: int = -1
        self.shock_size: float = 0.0
        self.shock_vector: np.ndarray | None = None

        # accounting
        self.total_transfer_fee = 0.0
        self.total_transfer_amount = 0.0
        self.terminal_yield = 0.0
        self.terminal_gap_penalty = 0.0

        # transition / behavior stats
        self.stop_count_stage0 = 0
        self.stop_count_stage1 = 0
        self.total_stop_count = 0
        self.zero_transfer_count = 0
        self.stage0_steps_used = 0
        self.stage1_steps_used = 0
        self.transition_history: List[Dict] = []

        self.last_info: Dict = {}

        # generator
        gen_cfg = CashFlowGeneratorConfig(
            seed=self.cfg.seed,
            num_accounts=self.cfg.num_accounts,
            master_idx=self.cfg.master_idx,
            investment_idx=self.cfg.investment_idx,
            operational_indices=self.cfg.operational_indices,
        )
        self.generator = SyntheticCashFlowGenerator(gen_cfg)

        self.stage1_edges = list(self.cfg.stage1_edges)
        self.stage2_edges = list(self.cfg.stage2_edges)

    # =========================================================
    # Reset
    # =========================================================
    def reset(self):
        self.stage = 0
        self.step_in_stage = 0
        self.done = False

        self.total_transfer_fee = 0.0
        self.total_transfer_amount = 0.0
        self.terminal_yield = 0.0
        self.terminal_gap_penalty = 0.0
        self.stop_count_stage0 = 0
        self.stop_count_stage1 = 0
        self.total_stop_count = 0

        self.zero_transfer_count = 0

        self.stage0_steps_used = 0
        self.stage1_steps_used = 0

        self.transition_history = []
        self.last_info = {}

        # initial balances
        low = np.asarray(self.cfg.init_balance_low, dtype=np.float64)
        high = np.asarray(self.cfg.init_balance_high, dtype=np.float64)
        self.balances = self.rng.uniform(low, high)

        # one-day demand generation
        payload = self.generator.generate_day()
        self.base_total_demand = payload["base_total_demand"].astype(np.float64)
        self.true_total_outflow = payload["true_total_outflow"].astype(np.float64)
        self.seasonal_base_forecast = payload["seasonal_base_forecast"].astype(
            np.float64
        )
        self.updated_total_forecast = payload["updated_total_forecast"].astype(
            np.float64
        )
        self.shock_target_account = int(payload["shock_target_account"])
        self.shock_size = float(payload["shock_size"])
        self.shock_vector = payload["shock_vector"].astype(np.float64)

        return self.get_observation_dict()

    # =========================================================
    # Observation
    # =========================================================
    def get_current_edges(self) -> List[Tuple[int, int]]:
        return self.stage1_edges if self.stage == 0 else self.stage2_edges

    def _current_predicted_remaining_outflow(self) -> np.ndarray:
        if self.stage == 0:
            return self.seasonal_base_forecast.copy()
        return self.updated_total_forecast.copy()

    def get_observation_dict(self) -> Dict:
        pred = self._current_predicted_remaining_outflow()

        return {
            "stage": self.stage,
            "step_in_stage": self.step_in_stage,
            "balances": self.balances.copy(),
            "predicted_remaining_outflow": pred.copy(),
            "current_edges": self.get_current_edges().copy(),
            "edge_count": len(self.get_current_edges()),
            "shock_target_account": (
                self.shock_target_account if self.stage == 1 else -1
            ),
            "shock_size": self.shock_size if self.stage == 1 else 0.0,
        }

    def get_state(self) -> np.ndarray:
        """
        Unified state vector:
          [stage_onehot(2),
           step_frac,
           balances(5),
           predicted_outflow(5),
           gap(5),
           surplus(5),
           shock_onehot(4),   # none / A2 / A3 / A4
           shock_size]
        """
        pred = self._current_predicted_remaining_outflow()
        balances = self.balances.copy()

        gap = np.maximum(pred - balances, 0.0)
        surplus = np.maximum(balances - pred, 0.0)

        stage_onehot = np.zeros(2, dtype=np.float32)
        stage_onehot[self.stage] = 1.0

        max_steps = (
            self.cfg.stage1_max_steps if self.stage == 0 else self.cfg.stage2_max_steps
        )
        step_frac = np.array([self.step_in_stage / max(max_steps, 1)], dtype=np.float32)

        shock_onehot = np.zeros(4, dtype=np.float32)  # [none, A2, A3, A4]
        if self.stage == 0 or self.shock_target_account == -1:
            shock_onehot[0] = 1.0
            shock_size = 0.0
        else:
            local_map = {2: 1, 3: 2, 4: 3}
            shock_onehot[local_map[self.shock_target_account]] = 1.0
            shock_size = self.shock_size

        return np.concatenate(
            [
                stage_onehot,
                step_frac,
                balances.astype(np.float32),
                pred.astype(np.float32),
                gap.astype(np.float32),
                surplus.astype(np.float32),
                shock_onehot.astype(np.float32),
                np.array([shock_size], dtype=np.float32),
            ]
        )

    # =========================================================
    # Step
    # =========================================================
    def step(self, action: Tuple[int, float]):
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() first.")

        edge_index, ratio = action
        ratio = float(np.clip(ratio, 0.0, 1.0))
        current_edges = self.get_current_edges()

        info = {
            "stage_before": self.stage,
            "step_in_stage_before": self.step_in_stage,
            "balances_before": self.balances.copy(),
        }

        # STOP
        if edge_index == len(current_edges):
            reward = 0.0
            info["stop_action"] = True
            if self.stage == 0:
                self.stop_count_stage0 += 1
            else:
                self.stop_count_stage1 += 1
            self.total_stop_count += 1
            done = self._advance_stage_or_finish(info, transition_reason="stop_action")

            if done:
                terminal_reward = self.compute_terminal_reward()
                reward += terminal_reward
                info["terminal_reward"] = terminal_reward
                info["terminal_yield"] = self.terminal_yield
                info["terminal_gap_penalty"] = self.terminal_gap_penalty
                info["balances_after_terminal_settlement"] = self.balances.copy()

            obs = None if done else self.get_observation_dict()
            return obs, float(reward), done, info

        if edge_index < 0 or edge_index >= len(current_edges):
            raise ValueError(
                f"Invalid edge_index={edge_index}, valid range is [0, {len(current_edges)}], "
                f"or STOP={len(current_edges)}"
            )

        source, target = current_edges[edge_index]
        amount = self._heuristic_transfer_amount(source, target, ratio)

        # immediate transfer
        self.balances[source] -= amount
        self.balances[target] += amount

        transfer_fee = self.cfg.transfer_fee_rate * amount
        shaping = 0.0
        if target == self.cfg.investment_idx:
            shaping = self.cfg.investment_shaping_weight * amount

        step_reward = -transfer_fee + shaping - self.cfg.step_penalty

        self.total_transfer_fee += transfer_fee
        self.total_transfer_amount += amount

        info.update(
            {
                "stop_action": False,
                "edge_index": edge_index,
                "edge": (source, target),
                "ratio": ratio,
                "transfer_amount": amount,
                "transfer_fee": transfer_fee,
                "balances_after_transfer": self.balances.copy(),
            }
        )

        if amount <= 1e-8:
            self.zero_transfer_count += 1

        self.step_in_stage += 1
        max_steps = (
            self.cfg.stage1_max_steps if self.stage == 0 else self.cfg.stage2_max_steps
        )

        if self.step_in_stage >= max_steps:
            done = self._advance_stage_or_finish(
                info, transition_reason="stage_step_limit"
            )
        else:
            done = False

        if done:
            terminal_reward = self.compute_terminal_reward()
            step_reward += terminal_reward
            info["terminal_reward"] = terminal_reward
            info["terminal_yield"] = self.terminal_yield
            info["terminal_gap_penalty"] = self.terminal_gap_penalty
            info["balances_after_terminal_settlement"] = self.balances.copy()

        obs = None if done else self.get_observation_dict()
        return obs, float(step_reward), done, info

    # =========================================================
    # Stage transition
    # =========================================================
    def _advance_stage_or_finish(self, info: Dict, transition_reason: str) -> bool:
        """
        stage 0 -> stage 1:
            no demand settlement, only information reveal

        stage 1 -> done:
            settle full-day true demand once, then terminal reward
        """
        info["transition_reason"] = transition_reason

        if self.stage == 0:
            self.stage0_steps_used = self.step_in_stage

            info["stage0_steps_used"] = self.stage0_steps_used
            info["stage2_reveal_shock_target_account"] = self.shock_target_account
            info["stage2_reveal_shock_size"] = self.shock_size
            info["updated_total_forecast"] = self.updated_total_forecast.copy()

            self.transition_history.append(
                {
                    "from_stage": 0,
                    "to_stage": 1,
                    "reason": transition_reason,
                    "steps_used": int(self.stage0_steps_used),
                }
            )

            self.stage = 1
            self.step_in_stage = 0
            return False

        # stage 1 -> episode end
        self.stage1_steps_used = self.step_in_stage
        info["stage1_steps_used"] = self.stage1_steps_used

        self.transition_history.append(
            {
                "from_stage": 1,
                "to_stage": "done",
                "reason": transition_reason,
                "steps_used": int(self.stage1_steps_used),
            }
        )

        # finish episode: settle whole-day demand once
        self._apply_outflow(self.true_total_outflow)
        info["true_total_outflow_realized"] = self.true_total_outflow.copy()
        info["balances_after_demand_settlement"] = self.balances.copy()

        self.done = True
        return True

    # =========================================================
    # Terminal reward
    # =========================================================
    def compute_terminal_reward(self) -> float:
        positive_bal = np.maximum(self.balances, 0.0)
        negative_bal = np.maximum(-self.balances, 0.0)

        yield_rates = np.asarray(self.cfg.account_yield_rate, dtype=np.float64)
        gap_penalties = np.asarray(self.cfg.account_gap_penalty, dtype=np.float64)

        self.terminal_yield = float(np.sum(positive_bal * yield_rates))
        self.terminal_gap_penalty = float(np.sum(negative_bal * gap_penalties))

        return (
            self.cfg.yield_weight * self.terminal_yield
            - self.cfg.gap_weight * self.terminal_gap_penalty
        )

    # =========================================================
    # Transfer amount rule
    # =========================================================
    def _heuristic_transfer_amount(
        self, source: int, target: int, ratio: float
    ) -> float:
        """
        amount = ratio * min(source_surplus, target_gap)
        if target is investment:
            amount = ratio * source_surplus
        """
        pred = self._current_predicted_remaining_outflow()

        own_need = pred[source]
        source_surplus = max(self.balances[source] - own_need, 0.0)

        if target == self.cfg.investment_idx:
            amount = ratio * source_surplus
        else:
            target_gap = max(pred[target] - self.balances[target], 0.0)
            amount = ratio * min(source_surplus, target_gap)

        amount = max(0.0, min(amount, max(self.balances[source], 0.0)))
        return float(amount)

    # =========================================================
    # Helper
    # =========================================================
    def _apply_outflow(self, outflow_vec: np.ndarray):
        self.balances = self.balances - outflow_vec

    # =========================================================
    # Utilities
    # =========================================================
    def episode_summary(self) -> Dict:
        terminal_reward = self.compute_terminal_reward() if self.done else None

        return {
            "config": asdict(self.cfg),
            "done": self.done,
            "balances_final": None if self.balances is None else self.balances.copy(),
            "base_total_demand": (
                None
                if self.base_total_demand is None
                else self.base_total_demand.copy()
            ),
            "true_total_outflow": (
                None
                if self.true_total_outflow is None
                else self.true_total_outflow.copy()
            ),
            "seasonal_base_forecast": (
                None
                if self.seasonal_base_forecast is None
                else self.seasonal_base_forecast.copy()
            ),
            "updated_total_forecast": (
                None
                if self.updated_total_forecast is None
                else self.updated_total_forecast.copy()
            ),
            "shock_target_account": self.shock_target_account,
            "shock_size": self.shock_size,
            "shock_vector": (
                None if self.shock_vector is None else self.shock_vector.copy()
            ),
            "total_transfer_fee": self.total_transfer_fee,
            "total_transfer_amount": self.total_transfer_amount,
            "terminal_yield": self.terminal_yield,
            "terminal_gap_penalty": self.terminal_gap_penalty,
            "terminal_reward": terminal_reward,
            "stop_count_stage0": self.stop_count_stage0,
            "stop_count_stage1": self.stop_count_stage1,
            "total_stop_count": self.total_stop_count,
            "zero_transfer_count": self.zero_transfer_count,
            "stage0_steps_used": self.stage0_steps_used,
            "stage1_steps_used": self.stage1_steps_used,
            "transition_history": self.transition_history,
        }

    # =========================================================
    # Snapshot / Restore
    # =========================================================
    def get_snapshot(self) -> Dict:
        """
        Deep-copy current env state for search / rollback.
        """
        return {
            # basic dynamic state
            "stage": int(self.stage),
            "step_in_stage": int(self.step_in_stage),
            "done": bool(self.done),
            "balances": None if self.balances is None else self.balances.copy(),
            # generator outputs
            "base_total_demand": (
                None
                if self.base_total_demand is None
                else self.base_total_demand.copy()
            ),
            "true_total_outflow": (
                None
                if self.true_total_outflow is None
                else self.true_total_outflow.copy()
            ),
            "seasonal_base_forecast": (
                None
                if self.seasonal_base_forecast is None
                else self.seasonal_base_forecast.copy()
            ),
            "updated_total_forecast": (
                None
                if self.updated_total_forecast is None
                else self.updated_total_forecast.copy()
            ),
            "shock_target_account": int(self.shock_target_account),
            "shock_size": float(self.shock_size),
            "shock_vector": (
                None if self.shock_vector is None else self.shock_vector.copy()
            ),
            # accounting
            "total_transfer_fee": float(self.total_transfer_fee),
            "total_transfer_amount": float(self.total_transfer_amount),
            "terminal_yield": float(self.terminal_yield),
            "terminal_gap_penalty": float(self.terminal_gap_penalty),
            # transition / behavior stats
            "stop_count_stage0": int(self.stop_count_stage0),
            "stop_count_stage1": int(self.stop_count_stage1),
            "total_stop_count": int(self.total_stop_count),
            "zero_transfer_count": int(self.zero_transfer_count),
            "stage0_steps_used": int(self.stage0_steps_used),
            "stage1_steps_used": int(self.stage1_steps_used),
            "transition_history": [dict(x) for x in self.transition_history],
            "last_info": (
                dict(self.last_info) if isinstance(self.last_info, dict) else {}
            ),
        }

    def load_snapshot(self, snapshot: Dict) -> None:
        """
        Restore env state from snapshot.
        """
        self.stage = int(snapshot["stage"])
        self.step_in_stage = int(snapshot["step_in_stage"])
        self.done = bool(snapshot["done"])
        self.balances = (
            None
            if snapshot["balances"] is None
            else np.asarray(snapshot["balances"], dtype=np.float64).copy()
        )

        self.base_total_demand = (
            None
            if snapshot["base_total_demand"] is None
            else np.asarray(snapshot["base_total_demand"], dtype=np.float64).copy()
        )
        self.true_total_outflow = (
            None
            if snapshot["true_total_outflow"] is None
            else np.asarray(snapshot["true_total_outflow"], dtype=np.float64).copy()
        )
        self.seasonal_base_forecast = (
            None
            if snapshot["seasonal_base_forecast"] is None
            else np.asarray(snapshot["seasonal_base_forecast"], dtype=np.float64).copy()
        )
        self.updated_total_forecast = (
            None
            if snapshot["updated_total_forecast"] is None
            else np.asarray(snapshot["updated_total_forecast"], dtype=np.float64).copy()
        )
        self.shock_target_account = int(snapshot["shock_target_account"])
        self.shock_size = float(snapshot["shock_size"])
        self.shock_vector = (
            None
            if snapshot["shock_vector"] is None
            else np.asarray(snapshot["shock_vector"], dtype=np.float64).copy()
        )

        self.total_transfer_fee = float(snapshot["total_transfer_fee"])
        self.total_transfer_amount = float(snapshot["total_transfer_amount"])
        self.terminal_yield = float(snapshot["terminal_yield"])
        self.terminal_gap_penalty = float(snapshot["terminal_gap_penalty"])

        self.stop_count_stage0 = int(snapshot["stop_count_stage0"])
        self.stop_count_stage1 = int(snapshot["stop_count_stage1"])
        self.total_stop_count = int(snapshot["total_stop_count"])
        self.zero_transfer_count = int(snapshot["zero_transfer_count"])
        self.stage0_steps_used = int(snapshot["stage0_steps_used"])
        self.stage1_steps_used = int(snapshot["stage1_steps_used"])
        self.transition_history = [dict(x) for x in snapshot["transition_history"]]
        self.last_info = (
            dict(snapshot["last_info"])
            if isinstance(snapshot["last_info"], dict)
            else {}
        )

    def export_planner_state(self):
        return self.get_snapshot()

    def load_planner_state(self, snapshot):
        self.load_snapshot(snapshot)


if __name__ == "__main__":
    cfg = CashFlowEnvConfig(seed=123)
    env = CashFlowEnv(cfg)

    obs = env.reset()
    print("===== RESET =====")
    print("obs:", obs)
    print("state shape:", env.get_state().shape)
    print("stage1 edges:", env.get_current_edges())

    done = False
    total_reward = 0.0

    while not done:
        current_edges = env.get_current_edges()
        pred = obs["predicted_remaining_outflow"]
        balances = obs["balances"]

        candidate = None
        best_gap = 0.0
        for edge_idx, (u, v) in enumerate(current_edges):
            if v in cfg.operational_indices:
                gap = max(pred[v] - balances[v], 0.0)
                if gap > best_gap:
                    best_gap = gap
                    candidate = edge_idx

        if candidate is None:
            action = (len(current_edges), 1.0)  # STOP
        else:
            action = (candidate, 0.8)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        print("\nstep reward:", reward, "done:", done)
        print("info:", info)

    print("\n===== SUMMARY =====")
    print(env.episode_summary())
    print("total reward incl. terminal:", total_reward)
