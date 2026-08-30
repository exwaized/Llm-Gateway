from fake_redis import FakeRedis
from llm_gateway.redis_telemetry import RedisTelemetry
from llm_gateway.telemetry import RequestRecord


def test_summary_empty():
    telemetry = RedisTelemetry(FakeRedis())
    assert telemetry.summary() == {"requests": 0}


def test_summary_aggregates_records():
    telemetry = RedisTelemetry(FakeRedis())
    telemetry.record(
        RequestRecord(
            prompt_preview="a", cache_hit=True, tier=None, provider=None, cost_usd=0.0, latency_s=0.01, escalated=False
        )
    )
    telemetry.record(
        RequestRecord(
            prompt_preview="b",
            cache_hit=False,
            tier="cheap",
            provider="mock-cheap",
            cost_usd=0.001,
            latency_s=0.02,
            escalated=False,
        )
    )
    telemetry.record(
        RequestRecord(
            prompt_preview="c",
            cache_hit=False,
            tier="frontier",
            provider="mock-frontier",
            cost_usd=0.01,
            latency_s=0.03,
            escalated=True,
        )
    )

    summary = telemetry.summary()
    assert summary["requests"] == 3
    assert summary["cache_hit_rate"] == 1 / 3
    assert summary["total_cost_usd"] == round(0.0 + 0.001 + 0.01, 6)
    assert summary["escalation_rate"] == 1 / 3
    assert summary["tier_distribution"] == {"cheap": 1, "frontier": 1}


def test_shared_client_sees_records_across_instances():
    client = FakeRedis()
    telemetry_a = RedisTelemetry(client)
    telemetry_a.record(
        RequestRecord(
            prompt_preview="a", cache_hit=False, tier="cheap", provider="p", cost_usd=0.1, latency_s=0.1, escalated=False
        )
    )

    telemetry_b = RedisTelemetry(client)
    assert telemetry_b.summary()["requests"] == 1


def test_max_records_caps_list_length():
    client = FakeRedis()
    telemetry = RedisTelemetry(client, max_records=2)
    for i in range(5):
        telemetry.record(
            RequestRecord(
                prompt_preview=str(i),
                cache_hit=False,
                tier="cheap",
                provider="p",
                cost_usd=0.0,
                latency_s=0.0,
                escalated=False,
            )
        )

    assert telemetry.summary()["requests"] == 2
