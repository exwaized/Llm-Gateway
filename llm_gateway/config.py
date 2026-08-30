from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    CHEAP = "cheap"
    MID = "mid"
    FRONTIER = "frontier"


# Ascending cost order used for fallback/escalation chains.
TIER_ORDER = [Tier.CHEAP, Tier.MID, Tier.FRONTIER]


@dataclass
class ProviderConfig:
    name: str
    tier: Tier
    cost_per_1k_input: float
    cost_per_1k_output: float
    timeout_s: float = 30.0


@dataclass
class CacheConfig:
    similarity_threshold: float = 0.92
    ttl_seconds: float = 3600.0
    max_entries: int = 5000


@dataclass
class BreakerConfig:
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    half_open_max_calls: int = 1


@dataclass
class ClassifierThresholds:
    # score in [0, 1]; below cheap_max -> CHEAP, below mid_max -> MID, else FRONTIER
    cheap_max: float = 0.35
    mid_max: float = 0.70

    def __post_init__(self) -> None:
        if not (0.0 <= self.cheap_max <= self.mid_max <= 1.0):
            raise ValueError(
                f"invalid thresholds: require 0 <= cheap_max <= mid_max <= 1, "
                f"got cheap_max={self.cheap_max}, mid_max={self.mid_max}"
            )

    @staticmethod
    def from_env() -> "ClassifierThresholds":
        """Reads GATEWAY_CHEAP_MAX / GATEWAY_MID_MAX, falling back to the defaults."""
        defaults = ClassifierThresholds()
        cheap_max = float(os.environ.get("GATEWAY_CHEAP_MAX", defaults.cheap_max))
        mid_max = float(os.environ.get("GATEWAY_MID_MAX", defaults.mid_max))
        return ClassifierThresholds(cheap_max=cheap_max, mid_max=mid_max)


@dataclass
class GatewayConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    thresholds: ClassifierThresholds = field(default_factory=ClassifierThresholds)
    max_workers: int = 8
