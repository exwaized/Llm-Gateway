from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    prompt_preview: str
    cache_hit: bool
    tier: str | None
    provider: str | None
    cost_usd: float
    latency_s: float
    escalated: bool
    timestamp: float = field(default_factory=time.time)


class Telemetry:
    """Thread-safe accounting sink. One instance shared across a Gateway."""

    def __init__(self):
        self._lock = threading.RLock()
        self._records: list[RequestRecord] = []

    def record(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)

    def summary(self) -> dict:
        with self._lock:
            records = list(self._records)

        if not records:
            return {"requests": 0}

        total = len(records)
        cache_hits = sum(1 for r in records if r.cache_hit)
        total_cost = sum(r.cost_usd for r in records)
        escalations = sum(1 for r in records if r.escalated)

        tier_counts: dict[str, int] = {}
        for r in records:
            if r.tier:
                tier_counts[r.tier] = tier_counts.get(r.tier, 0) + 1

        return {
            "requests": total,
            "cache_hit_rate": cache_hits / total,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / total, 6),
            "escalation_rate": escalations / total,
            "tier_distribution": tier_counts,
        }
