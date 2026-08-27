from __future__ import annotations

import threading
import time
from enum import Enum

from llm_gateway.config import BreakerConfig


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised by CircuitBreaker.before_call() when the breaker is open."""


class CircuitBreaker:
    """Per-provider failure tracker: closed -> open -> half-open -> closed.

    - CLOSED: calls pass through; N consecutive failures trips to OPEN.
    - OPEN: calls are rejected immediately (no network call at all) until
      recovery_timeout_s has elapsed, then one probe is allowed (HALF_OPEN).
    - HALF_OPEN: a limited number of probe calls are allowed through; a
      success closes the breaker, a failure re-opens it and resets the timer.

    All state transitions happen under a single RLock since this object is
    shared across every worker thread dispatching to the same provider.
    """

    def __init__(self, name: str, config: BreakerConfig):
        self.name = name
        self._config = config
        self._lock = threading.RLock()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_calls_in_flight = 0

    @property
    def state(self) -> BreakerState:
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state

    def _maybe_transition_to_half_open_locked(self) -> None:
        if (
            self._state == BreakerState.OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self._config.recovery_timeout_s
        ):
            self._state = BreakerState.HALF_OPEN
            self._half_open_calls_in_flight = 0

    def before_call(self) -> None:
        """Call before dispatching. Raises CircuitOpenError if the call should be skipped."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()

            if self._state == BreakerState.OPEN:
                raise CircuitOpenError(f"circuit open for provider={self.name!r}")

            if self._state == BreakerState.HALF_OPEN:
                if self._half_open_calls_in_flight >= self._config.half_open_max_calls:
                    raise CircuitOpenError(
                        f"circuit half-open probe already in flight for provider={self.name!r}"
                    )
                self._half_open_calls_in_flight += 1

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = BreakerState.CLOSED
            self._opened_at = None
            self._half_open_calls_in_flight = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._state == BreakerState.HALF_OPEN:
                self._half_open_calls_in_flight = 0
                self._trip_locked()
                return

            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.failure_threshold:
                self._trip_locked()

    def _trip_locked(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = time.monotonic()

    def call(self, fn, *args, **kwargs):
        """Convenience wrapper: before_call -> fn -> record_success/record_failure."""
        self.before_call()
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result


class CircuitBreakerRegistry:
    """One CircuitBreaker per provider name, created lazily, thread-safe."""

    def __init__(self, config: BreakerConfig):
        self._config = config
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get(self, provider_name: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(provider_name)
            if breaker is None:
                breaker = CircuitBreaker(provider_name, self._config)
                self._breakers[provider_name] = breaker
            return breaker

    def snapshot(self) -> dict[str, BreakerState]:
        with self._lock:
            return {name: b.state for name, b in self._breakers.items()}
