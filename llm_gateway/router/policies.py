from __future__ import annotations

from abc import ABC, abstractmethod

from llm_gateway.config import ClassifierThresholds, Tier


class RoutingPolicy(ABC):
    """Decides which tier a request should *start* on.

    The Router still owns fallback/escalation; a policy only picks the first tier.
    """

    @abstractmethod
    def choose_tier(self, prompt: str, score: float) -> Tier:
        raise NotImplementedError


def tier_for_score(score: float, thresholds: ClassifierThresholds) -> Tier:
    if score < thresholds.cheap_max:
        return Tier.CHEAP
    if score < thresholds.mid_max:
        return Tier.MID
    return Tier.FRONTIER


class ThresholdPolicy(RoutingPolicy):
    """Maps the classifier's complexity score to a tier via fixed thresholds."""

    def __init__(self, thresholds: ClassifierThresholds | None = None):
        self.thresholds = thresholds or ClassifierThresholds()

    def choose_tier(self, prompt: str, score: float) -> Tier:
        return tier_for_score(score, self.thresholds)
