"""Fixed stage-zero route-mask controls with a raw Gaussian PPO interface.

Controlled settings share the action decoder, observations, transitions, rewards,
transfer executor, stage-one routes, and reset mechanism. Each uses the MMDP
wrapper with its stage-zero route list set once after construction.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "configs"
SOURCE_ROOT = Path(__file__).resolve().parents[1]
from ..envs.cash_flow_env import CashFlowEnv, CashFlowEnvConfig  # noqa: E402
from ..envs.cash_flow_generator import SyntheticCashFlowGenerator  # noqa: E402
from .flat_cash_gym_env import FlatCashGymEnv  # noqa: E402
from .mmdp_cash_gym_env import MMDPCashGymEnv  # noqa: E402

for _class in (CashFlowEnv, SyntheticCashFlowGenerator, FlatCashGymEnv, MMDPCashGymEnv):
    if (
        not Path(inspect.getfile(_class))
        .resolve()
        .is_relative_to(SOURCE_ROOT.resolve())
    ):
        raise ImportError(
            f"Cash5 source collision: {_class} from {inspect.getfile(_class)}"
        )

CFG = CashFlowEnvConfig()
GLOBAL_ROUTES = tuple(tuple(route) for route in CFG.stage1_edges)
STAGE1_ROUTES = tuple(tuple(route) for route in CFG.stage2_edges)
TRUE_STAGE0_ROUTES = tuple(
    route for route in GLOBAL_ROUTES if route not in STAGE1_ROUTES
)
assert (
    len(GLOBAL_ROUTES) == 14
    and len(STAGE1_ROUTES) == 8
    and len(TRUE_STAGE0_ROUTES) == 6
)


def canonical_routes(routes):
    """Reject reordering/duplicates: mask membership filters the global catalogue."""
    requested = tuple(tuple(int(account) for account in route) for route in routes)
    assert requested and len(requested) == len(set(requested))
    assert set(requested).issubset(GLOBAL_ROUTES)
    ordered = tuple(route for route in GLOBAL_ROUTES if route in set(requested))
    assert requested == ordered, "Route order must follow the global catalogue"
    return ordered


class ControlledCashEnv(MMDPCashGymEnv):
    """Shared local decoder/executor with a fixed, canonical stage-zero whitelist."""

    def __init__(self, base_env, stage0_routes):
        ordered = canonical_routes(stage0_routes)
        assert tuple(base_env.stage1_edges) == GLOBAL_ROUTES
        assert tuple(base_env.stage2_edges) == STAGE1_ROUTES
        super().__init__(base_env)  # The parent constructor performs one reset.
        self.stage0_edges = list(ordered)
        self.fixed_stage0_routes = ordered
        self.reset_ordinal = 1
        assert tuple(self.stage1_edges) == STAGE1_ROUTES
        assert (
            self.observation_space.shape == (28,)
            and self.observation_space.dtype == np.float32
        )
        assert self.action_space.shape == (2,) and self.action_space.dtype == np.float32
        assert np.array_equal(
            self.action_space.low, np.array([-1.0, -1.0], dtype=np.float32)
        )
        assert np.array_equal(
            self.action_space.high, np.array([1.0, 1.0], dtype=np.float32)
        )

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.reset_ordinal += 1
        # Logging only. This field is never concatenated into the observation.
        info["canonical_reset_ordinal"] = self.reset_ordinal
        assert tuple(self.stage0_edges) == self.fixed_stage0_routes
        assert tuple(self.stage1_edges) == STAGE1_ROUTES
        return observation, info


def load_inventory(inventory=None):
    if inventory is None:
        return json.loads((ROOT / "mask_inventory.json").read_text())
    if isinstance(inventory, (str, Path)):
        return json.loads(Path(inventory).read_text())
    if not isinstance(inventory, dict):
        raise TypeError("inventory must be a dictionary or JSON path")
    return inventory


def setting_routes(setting_id, inventory=None):
    inventory = load_inventory(inventory)
    assert inventory["schema_version"] == 1
    assert (
        tuple(tuple(route) for route in inventory["catalogue_routes"]) == GLOBAL_ROUTES
    )
    assert (
        tuple(tuple(route) for route in inventory["persistent_routes"]) == STAGE1_ROUTES
    )
    assert (
        tuple(tuple(route) for route in inventory["true_routes"]) == TRUE_STAGE0_ROUTES
    )
    settings = inventory["settings"]
    if isinstance(settings, list):
        indexed = {row["setting_id"]: row for row in settings}
        assert len(indexed) == len(settings), "Duplicate setting_id"
    else:
        assert isinstance(settings, dict)
        indexed = settings
    row = indexed[setting_id]
    routes = canonical_routes(row["stage0_routes"])
    assert list(row["stage0_indices"]) == [
        GLOBAL_ROUTES.index(route) for route in routes
    ]
    if "stage1_routes" in row:
        assert tuple(tuple(route) for route in row["stage1_routes"]) == STAGE1_ROUTES
    return routes


def make_env(setting_id, seed, inventory=None):
    routes = setting_routes(setting_id, inventory)
    env = ControlledCashEnv(CashFlowEnv(CashFlowEnvConfig(seed=int(seed))), routes)
    # Setting identifiers are diagnostics and never policy features.
    env.setting_id = str(setting_id)
    return env


def current_routes(env):
    return env.stage0_edges if env.base_env.stage == 0 else env.stage1_edges


def encode_semantic_action(env, route, ratio):
    """Map a semantic route/ratio into the float32 Gaussian-wrapper encoding."""
    routes = current_routes(env)
    index = len(routes) if route == "STOP" else routes.index(tuple(route))
    assert 0 <= float(ratio) <= 1
    return np.array(
        [2.0 * (index + 0.5) / (len(routes) + 1) - 1.0, 2.0 * float(ratio) - 1.0],
        dtype=np.float32,
    )


def to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def exogenous_payload(env):
    """Audit-only initial instance fields, excluded from the policy interface."""
    base = env.base_env
    return dict(
        initial_balances=base.balances.copy(),
        base_total_demand=base.base_total_demand.copy(),
        true_total_outflow=base.true_total_outflow.copy(),
        seasonal_base_forecast=base.seasonal_base_forecast.copy(),
        updated_total_forecast=base.updated_total_forecast.copy(),
        shock_target_account=int(base.shock_target_account),
        shock_size=float(base.shock_size),
        shock_vector=base.shock_vector.copy(),
    )
