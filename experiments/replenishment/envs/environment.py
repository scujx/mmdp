"""Finite-state weekly forecast channel with real lead time and daily lost sales."""

from pathlib import Path
import hashlib, json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

CFG = json.loads(
    (Path(__file__).resolve().parents[1] / "configs/environment.json").read_text()
)
D = np.asarray(CFG["weekly_demand"], dtype=int)
CAP = CFG["flex_capacity"]
# Auxiliary channels generate the shared coarse-signal distribution.
# Their intermediate categories are not exposed to the policy.
AUXILIARY_FINE_CHANNEL = np.full((3, 3), 0.075)
np.fill_diagonal(AUXILIARY_FINE_CHANNEL, 0.85)
AUXILIARY_COARSE_CHANNEL = np.full((3, 3), 0.175)
np.fill_diagonal(AUXILIARY_COARSE_CHANNEL, 0.65)
COARSE_GIVEN_DEMAND = AUXILIARY_FINE_CHANNEL @ AUXILIARY_COARSE_CHANNEL
MAIN_FINE_ACCURACY = 0.80
COARSE_ACCURACY = 0.57875


def signal_law(fine_accuracy):
    """Construct D -> F -> C while preserving the fixed coarse-given-demand law."""
    p = float(fine_accuracy)
    if p not in (0.75, 0.80, 0.85):
        raise ValueError("Supported fine accuracies are 0.75, 0.80, and 0.85")
    fine = np.full((3, 3), (1 - p) / 2)
    np.fill_diagonal(fine, p)
    q = (COARSE_ACCURACY - (1 - p) / 2) / (p - (1 - p) / 2)
    degradation = np.full((3, 3), (1 - q) / 2)
    np.fill_diagonal(degradation, q)
    if not np.allclose(fine @ degradation, COARSE_GIVEN_DEMAND, atol=1e-12):
        raise ValueError("Fine accuracy changed the coarse-signal law")
    joint = fine[:, :, None] * degradation[None, :, :] / 3
    conditional = fine[:, :, None] * degradation[None, :, :] / COARSE_GIVEN_DEMAND[:, None, :]
    return conditional, joint


CONDITIONAL_FINE, JOINT = signal_law(MAIN_FINE_ACCURACY)
# Named main-setting matrices are also used by the benchmark's invariant tests.
PF = np.full((3, 3), (1 - MAIN_FINE_ACCURACY) / 2)
np.fill_diagonal(PF, MAIN_FINE_ACCURACY)
KEEP = (COARSE_ACCURACY - (1 - MAIN_FINE_ACCURACY) / 2) / (MAIN_FINE_ACCURACY - (1 - MAIN_FINE_ACCURACY) / 2)
PC = np.full((3, 3), (1 - KEEP) / 2)
np.fill_diagonal(PC, KEEP)
SPLITS = {"dev": 1, "smoke": 2, "train": 3, "validation": 4, "test": 5}
SCHEMA = [
    ("early", 1, "one-hot"),
    ("late", 1, "one-hot"),
    ("remaining_order_weeks", 1, "/13"),
    ("inventory", 1, "/168"),
    ("pending_regular", 1, "/168"),
    ("pending_flexible", 1, "/67.2"),
    ("remaining_flexible_capacity", 1, "/67.2"),
    ("monday_opening_inventory", 1, "/168"),
    ("observed_current_week_demand", 1, "/126"),
    ("current_coarse", 3, "one-hot"),
    ("retained_current_fine", 3, "one-hot"),
    ("current_fine_available", 1, "binary"),
    ("next_coarse", 3, "one-hot"),
    ("next_fine", 3, "one-hot"),
    ("next_fine_available", 1, "binary"),
    ("updated_mode", 1, "binary"),
]


def make_episode(split, identity, ordinal, fine_accuracy=MAIN_FINE_ACCURACY):
    conditional_fine, _ = signal_law(fine_accuracy)
    def rng(stream):
        return np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence(
                    CFG["rng_root"]
                    + [SPLITS[split], int(identity), int(ordinal), stream]
                )
            )
        )

    state = rng(1).integers(0, 3, 14)
    fine = np.array(
        [
            np.searchsorted(np.cumsum(AUXILIARY_FINE_CHANNEL[s]), u)
            for s, u in zip(state, rng(2).random(14))
        ]
    )
    coarse = np.array(
        [
            np.searchsorted(np.cumsum(AUXILIARY_COARSE_CHANNEL[f]), u)
            for f, u in zip(fine, rng(3).random(14))
        ]
    )
    # Given the shared coarse draws, sample the observed fine categories conditionally.
    # This reverse sampling order realizes the specified D -> F -> C joint law.
    fine = np.array(
        [
            np.searchsorted(np.cumsum(conditional_fine[s, :, c]), u)
            for s, c, u in zip(state, coarse, rng(5).random(14))
        ]
    )
    daily_rng = rng(4)
    daily = np.array([daily_rng.multinomial(int(D[s]), np.ones(7) / 7) for s in state])
    fp = hashlib.sha256(
        state.astype("<i8").tobytes()
        + fine.astype("<i8").tobytes()
        + coarse.astype("<i8").tobytes()
        + daily.astype("<i8").tobytes()
    ).hexdigest()
    return state, fine, coarse, daily, fp


def posterior(coarse, fine=None):
    p = JOINT[:, :, coarse].sum(1) if fine is None else JOINT[:, fine, coarse]
    return p / p.sum()


class LeadTimeReplenishment(gym.Env):
    metadata = {}

    def __init__(
        self, interface="flat", information="updated", split="dev", identity=9244000,
        fine_accuracy=MAIN_FINE_ACCURACY,
    ):
        super().__init__()
        assert interface in ["flat", "mmdp"]
        assert information in ["retained", "updated"]
        signal_law(fine_accuracy)
        self.fine_accuracy = float(fine_accuracy)
        self.interface, self.information, self.split, self.identity = (
            interface,
            information,
            split,
            identity,
        )
        self.ordinal = -1
        self.done = True
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        high = np.ones(24, np.float32)
        high[[3, 7]] = (63 + 13 * (168 + CAP)) / 168
        self.observation_space = spaces.Box(
            np.zeros(24, np.float32), high, dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ordinal += 1
        self.state, self.f, self.c, self.daily, self.fingerprint = make_episode(
            self.split, self.identity, self.ordinal, self.fine_accuracy
        )
        self.week = 0
        self.stage = 0
        self.inventory = 63.0
        self.opening = 63.0
        self.reg = 0.0
        self.flex = 0.0
        self.observed = 0.0
        self.done = False
        self.length = 0
        self.return_sum = 0.0
        self.days = 0
        self.arrivals = []
        self.totals = dict(
            revenue=0.0,
            regular_cost=0.0,
            flexible_cost=0.0,
            holding_cost=0.0,
            shortage_cost=0.0,
        )
        return self.obs(), {"instance_fingerprint": self.fingerprint}

    def obs(self):
        x = np.zeros(24, np.float32)
        if self.done:
            return x
        x[self.stage] = 1
        x[2:9] = [
            (13 - self.week) / 13,
            self.inventory / 168,
            self.reg / 168,
            self.flex / CAP,
            (CAP - self.flex) / CAP,
            self.opening / 168,
            self.observed / 126,
        ]
        x[9 + self.c[self.week]] = 1
        x[16 + self.c[self.week + 1]] = 1
        # A previously released delivery-week signal remains public next week.
        current_visible = self.information == "updated" and self.week > 0
        next_visible = self.information == "updated" and self.stage == 1
        if current_visible:
            x[12 + self.f[self.week]] = 1
            x[15] = 1
        if next_visible:
            x[19 + self.f[self.week + 1]] = 1
            x[22] = 1
        x[23] = self.information == "updated"
        return x

    def decode(self, action):
        w = np.clip((np.asarray(action, dtype=np.float64) + 1) / 2, 0, 1)
        reg = 84 * (w[0] + w[1]) if self.stage == 0 else 0.0
        flex = (
            min(CAP * w[2], CAP - self.flex)
            if self.stage == 1 or self.interface == "flat"
            else 0.0
        )
        return float(reg), float(flex)

    def _days(self, week, start, end, parts, trace):
        for day in range(start, end):
            demand = float(self.daily[week, day])
            opening = self.inventory
            sold = min(opening, demand)
            short = demand - sold
            self.inventory -= sold
            self.days += 1
            parts["revenue"] += 6 * sold
            parts["shortage_cost"] += 3 * short
            parts["holding_cost"] += 0.5 * self.inventory
            trace.append(
                dict(
                    week=week,
                    day=day,
                    opening=opening,
                    demand=demand,
                    sales=sold,
                    shortage=short,
                    ending=self.inventory,
                )
            )
            if week == self.week:
                self.observed += demand

    def step(self, action):
        if self.done:
            raise RuntimeError("step after termination")
        week, stage = self.week, self.stage
        reg, flex = self.decode(action)
        self.reg += reg
        self.flex += flex
        parts = dict(
            revenue=0.0,
            regular_cost=reg,
            flexible_cost=1.5 * flex,
            holding_cost=0.0,
            shortage_cost=0.0,
        )
        trace = []
        arrival = 0.0
        if stage == 0:
            self._days(week, 0, 3, parts, trace)
            self.stage = 1
        else:
            self._days(week, 3, 7, parts, trace)
            arrival = self.reg + self.flex
            self.inventory += arrival
            self.arrivals.append((week + 1, arrival))
            self.reg = 0.0
            self.flex = 0.0
            if week == 12:
                self._days(13, 0, 7, parts, trace)
                self.done = True
            else:
                self.week += 1
                self.stage = 0
                self.opening = self.inventory
                self.observed = 0.0
        reward = parts["revenue"] - sum(v for k, v in parts.items() if k != "revenue")
        for k, v in parts.items():
            self.totals[k] += v
        self.return_sum += reward
        self.length += 1
        info = dict(
            week=week,
            stage=stage,
            executed_regular=reg,
            executed_flexible=flex,
            arrived=arrival,
            arrival_week=week + 1 if stage else None,
            daily_trace=trace,
            **parts
        )
        return self.obs(), float(reward), self.done, False, info
