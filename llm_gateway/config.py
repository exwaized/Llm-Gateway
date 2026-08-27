from __future__ import annotations

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


@dataclass
class GatewayConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    breaker: BreakerConfig = field(default_factory=BreakerConfig)
    thresholds: ClassifierThresholds = field(default_factory=ClassifierThresholds)
    max_workers: int = 8
