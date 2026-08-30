import time

import pytest

from fake_redis import FakeRedis
from llm_gateway.breaker.circuit_breaker import BreakerState, CircuitOpenError
from llm_gateway.breaker.redis_breaker import RedisCircuitBreaker, RedisCircuitBreakerRegistry
from llm_gateway.config import BreakerConfig


def test_trips_open_after_failure_threshold():
    breaker = RedisCircuitBreaker("p", BreakerConfig(failure_threshold=3, recovery_timeout_s=60), FakeRedis())

    for _ in range(3):
        breaker.before_call()
        breaker.record_failure()

    assert breaker.state == BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_success_resets_failure_count():
    breaker = RedisCircuitBreaker("p", BreakerConfig(failure_threshold=3, recovery_timeout_s=60), FakeRedis())

    breaker.before_call()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_success()

    assert breaker.state == BreakerState.CLOSED
    breaker.before_call()
    breaker.record_failure()
    assert breaker.state == BreakerState.CLOSED


def test_transitions_to_half_open_after_recovery_timeout():
    breaker = RedisCircuitBreaker("p", BreakerConfig(failure_threshold=1, recovery_timeout_s=0.05), FakeRedis())

    breaker.before_call()
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN

    time.sleep(0.1)
    assert breaker.state == BreakerState.HALF_OPEN


def test_half_open_failure_reopens():
    breaker = RedisCircuitBreaker(
        "p", BreakerConfig(failure_threshold=1, recovery_timeout_s=0.05, half_open_max_calls=1), FakeRedis()
    )
    breaker.before_call()
    breaker.record_failure()
    time.sleep(0.1)
    assert breaker.state == BreakerState.HALF_OPEN

    breaker.before_call()
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN


def test_half_open_success_closes():
    breaker = RedisCircuitBreaker(
        "p", BreakerConfig(failure_threshold=1, recovery_timeout_s=0.05, half_open_max_calls=1), FakeRedis()
    )
    breaker.before_call()
    breaker.record_failure()
    time.sleep(0.1)
    assert breaker.state == BreakerState.HALF_OPEN

    breaker.before_call()
    breaker.record_success()
    assert breaker.state == BreakerState.CLOSED


def test_registry_shares_client_and_reports_snapshot():
    client = FakeRedis()
    registry = RedisCircuitBreakerRegistry(BreakerConfig(failure_threshold=1), client)

    a = registry.get("provider-a")
    a.before_call()
    a.record_failure()

    snapshot = registry.snapshot()
    assert snapshot["provider-a"] == BreakerState.OPEN

    # a second registry sharing the same client sees the same state --
    # this is the cross-instance sharing the in-memory registry can't do
    other_registry = RedisCircuitBreakerRegistry(BreakerConfig(failure_threshold=1), client)
    assert other_registry.get("provider-a").state == BreakerState.OPEN
