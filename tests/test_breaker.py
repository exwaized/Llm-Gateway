import time

import pytest

from llm_gateway.breaker.circuit_breaker import BreakerState, CircuitBreaker, CircuitOpenError
from llm_gateway.config import BreakerConfig


def test_trips_open_after_failure_threshold():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=3, recovery_timeout_s=60))

    for _ in range(3):
        breaker.before_call()
        breaker.record_failure()

    assert breaker.state == BreakerState.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_success_resets_failure_count():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=3, recovery_timeout_s=60))

    breaker.before_call()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_failure()
    breaker.before_call()
    breaker.record_success()

    assert breaker.state == BreakerState.CLOSED
    breaker.before_call()
    breaker.record_failure()
    assert breaker.state == BreakerState.CLOSED  # still below threshold post-reset


def test_transitions_to_half_open_after_recovery_timeout():
    breaker = CircuitBreaker("p", BreakerConfig(failure_threshold=1, recovery_timeout_s=0.05))

    breaker.before_call()
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN

    time.sleep(0.1)
    assert breaker.state == BreakerState.HALF_OPEN


def test_half_open_failure_reopens():
    breaker = CircuitBreaker(
        "p", BreakerConfig(failure_threshold=1, recovery_timeout_s=0.05, half_open_max_calls=1)
    )
    breaker.before_call()
    breaker.record_failure()
    time.sleep(0.1)
    assert breaker.state == BreakerState.HALF_OPEN

    breaker.before_call()  # the probe call
    breaker.record_failure()
    assert breaker.state == BreakerState.OPEN


def test_half_open_success_closes():
    breaker = CircuitBreaker(
        "p", BreakerConfig(failure_threshold=1, recovery_timeout_s=0.05, half_open_max_calls=1)
    )
    breaker.before_call()
    breaker.record_failure()
    time.sleep(0.1)
    assert breaker.state == BreakerState.HALF_OPEN

    breaker.before_call()
    breaker.record_success()
    assert breaker.state == BreakerState.CLOSED
