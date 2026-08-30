"""Upstash Redis-backed Telemetry.

Same record()/summary() surface as Telemetry, so Gateway can use either
interchangeably. Records persist in a capped Redis list instead of an
in-process list, so /telemetry reflects total traffic across every
serverless instance rather than whatever happened to hit the instance
serving the request.
"""

from __future__ import annotations

import json

from llm_gateway.telemetry import RequestRecord

_LIST_KEY = "telemetry:records"
_MAX_RECORDS = 5000


class RedisTelemetry:
    def __init__(self, client, max_records: int = _MAX_RECORDS):
        self._client = client
        self._max_records = max_records

    def record(self, rec: RequestRecord) -> None:
        payload = json.dumps(
            {
                "prompt_preview": rec.prompt_preview,
                "cache_hit": rec.cache_hit,
                "tier": rec.tier,
                "provider": rec.provider,
                "cost_usd": rec.cost_usd,
                "latency_s": rec.latency_s,
                "escalated": rec.escalated,
                "timestamp": rec.timestamp,
            }
        )
        self._client.lpush(_LIST_KEY, payload)
        self._client.ltrim(_LIST_KEY, 0, self._max_records - 1)

    def summary(self) -> dict:
        raw_records = self._client.lrange(_LIST_KEY, 0, -1) or []
        records = [json.loads(r) for r in raw_records]

        if not records:
            return {"requests": 0}

        total = len(records)
        cache_hits = sum(1 for r in records if r["cache_hit"])
        total_cost = sum(r["cost_usd"] for r in records)
        escalations = sum(1 for r in records if r["escalated"])

        tier_counts: dict[str, int] = {}
        for r in records:
            if r["tier"]:
                tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1

        return {
            "requests": total,
            "cache_hit_rate": cache_hits / total,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd": round(total_cost / total, 6),
            "escalation_rate": escalations / total,
            "tier_distribution": tier_counts,
        }
