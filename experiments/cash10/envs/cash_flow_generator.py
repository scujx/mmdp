from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple, Dict


@dataclass
class CashFlowGeneratorConfig:
    # =========================================================
    # Basic structure
    # =========================================================
    seed: int = 42
    num_accounts: int = 5
    master_idx: int = 0
    investment_idx: int = 1
    operational_indices: Tuple[int, int, int] = (2, 3, 4)

    # =========================================================
    # Base demand (seasonal / low-frequency component)
    # =========================================================
    op_base_demand_mean: Tuple[float, ...] = (55.0, 40.0, 35.0)   # A2, A3, A4
    op_base_demand_std: Tuple[float, ...] = (10.0, 8.0, 7.0)

    # =========================================================
    # Stage-1 seasonal forecast noise
    # =========================================================
    stage1_forecast_noise_ratio: float = 0.20

    # =========================================================
    # Stage-2 shock design
    # One localized shock hits exactly one operational account
    # =========================================================
    shock_prob: float = 0.90
    shock_target_probs: Tuple[float, float, float] = (0.40, 0.35, 0.25)  # on A2/A3/A4
    shock_mean: float = 35.0
    shock_std: float = 12.0

    # Optional reveal noise in stage 2
    stage2_reveal_noise_ratio: float = 0.00


class SyntheticCashFlowGenerator:
    """
    Two-stage synthetic cash-flow generator.

    Structure:
      - Stage 1 observes a seasonal / low-frequency forecast of total demand
      - Before Stage 2, a localized demand shock is revealed
      - The shock only hits one operational account
      - Stage 2 observes an updated total-demand forecast after the shock reveal

    Returned payload:
      {
        "base_total_demand": np.ndarray shape [num_accounts],
        "true_total_outflow": np.ndarray shape [num_accounts],
        "seasonal_base_forecast": np.ndarray shape [num_accounts],
        "updated_total_forecast": np.ndarray shape [num_accounts],
        "shock_target_account": int,   # actual account index in {2,3,4}, or -1 if no shock
        "shock_size": float,
        "shock_vector": np.ndarray shape [num_accounts],
        "config": dict,
      }
    """

    def __init__(self, config: CashFlowGeneratorConfig = CashFlowGeneratorConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)

    # =========================================================
    # Main public API
    # =========================================================
    def generate_day(self) -> Dict:
        base_total_demand = self._generate_base_total_demand()
        seasonal_base_forecast = self._make_stage1_seasonal_forecast(base_total_demand)

        shock_target_account, shock_size, shock_vector = self._generate_localized_shock()
        true_total_outflow = base_total_demand + shock_vector
        updated_total_forecast = self._make_stage2_updated_forecast(true_total_outflow)

        payload = {
            "base_total_demand": base_total_demand.astype(np.float32),
            "true_total_outflow": true_total_outflow.astype(np.float32),
            "seasonal_base_forecast": seasonal_base_forecast.astype(np.float32),
            "updated_total_forecast": updated_total_forecast.astype(np.float32),
            "shock_target_account": int(shock_target_account),
            "shock_size": float(shock_size),
            "shock_vector": shock_vector.astype(np.float32),
            "config": asdict(self.cfg),
        }
        return payload

    # =========================================================
    # Internal generators
    # =========================================================
    def _generate_base_total_demand(self) -> np.ndarray:
        demand = np.zeros(self.cfg.num_accounts, dtype=np.float64)

        for local_j, acct_idx in enumerate(self.cfg.operational_indices):
            mean = self.cfg.op_base_demand_mean[local_j]
            std = self.cfg.op_base_demand_std[local_j]
            demand[acct_idx] = max(0.0, self.rng.normal(mean, std))

        return demand

    def _make_stage1_seasonal_forecast(self, base_total_demand: np.ndarray) -> np.ndarray:
        forecast = np.zeros(self.cfg.num_accounts, dtype=np.float64)

        for idx in self.cfg.operational_indices:
            noise_std = self.cfg.stage1_forecast_noise_ratio * max(base_total_demand[idx], 1.0)
            forecast[idx] = max(
                0.0,
                base_total_demand[idx] + self.rng.normal(0.0, noise_std),
            )

        return forecast

    def _generate_localized_shock(self) -> Tuple[int, float, np.ndarray]:
        shock_vector = np.zeros(self.cfg.num_accounts, dtype=np.float64)

        has_shock = self.rng.uniform() < self.cfg.shock_prob
        if not has_shock:
            return -1, 0.0, shock_vector

        local_target = int(
            self.rng.choice(
                np.arange(len(self.cfg.operational_indices)),
                p=np.asarray(self.cfg.shock_target_probs, dtype=np.float64)
                / np.sum(np.asarray(self.cfg.shock_target_probs, dtype=np.float64)),
            )
        )
        target_account = int(self.cfg.operational_indices[local_target])

        shock_size = max(0.0, self.rng.normal(self.cfg.shock_mean, self.cfg.shock_std))
        shock_vector[target_account] = shock_size

        return target_account, shock_size, shock_vector

    def _make_stage2_updated_forecast(self, true_total_outflow: np.ndarray) -> np.ndarray:
        forecast = np.zeros(self.cfg.num_accounts, dtype=np.float64)

        for idx in self.cfg.operational_indices:
            noise_std = self.cfg.stage2_reveal_noise_ratio * max(true_total_outflow[idx], 1.0)
            forecast[idx] = max(
                0.0,
                true_total_outflow[idx] + self.rng.normal(0.0, noise_std),
            )

        return forecast


if __name__ == "__main__":
    cfg = CashFlowGeneratorConfig(seed=123)
    gen = SyntheticCashFlowGenerator(cfg)

    payload = gen.generate_day()

    print("base_total_demand       =", np.round(payload["base_total_demand"], 2))
    print("seasonal_base_forecast  =", np.round(payload["seasonal_base_forecast"], 2))
    print("shock_target_account    =", payload["shock_target_account"])
    print("shock_size              =", round(payload["shock_size"], 2))
    print("shock_vector            =", np.round(payload["shock_vector"], 2))
    print("updated_total_forecast  =", np.round(payload["updated_total_forecast"], 2))
    print("true_total_outflow      =", np.round(payload["true_total_outflow"], 2))
