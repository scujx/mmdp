"""Episode logging and training-only reward scaling."""

import gymnasium as gym
from ..envs.environment import LeadTimeReplenishment


class Replenishment(LeadTimeReplenishment):
    """Log episode identity without altering the physical environment."""

    def __init__(self, interface, information, split, identity, log=None, fine_accuracy=0.80):
        super().__init__(interface, information, split, identity, fine_accuracy)
        self.log = log

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        if self.log:
            self.log(
                dict(
                    split=self.split,
                    identity=self.identity,
                    ordinal=self.ordinal,
                    fingerprint=self.fingerprint,
                )
            )
        return result


class FixedRewardScale(gym.RewardWrapper):
    """Scale only the learner reward; leave observations, info and RNG unchanged."""

    def reward(self, reward):
        return 0.01 * reward
