"""Upstash Redis-backed mirror of CircuitBreaker.

Same before_call()/record_success()/record_failure() surface as
CircuitBreaker/CircuitBreakerRegistry, so Gateway/Router can use either
interchangeably. State lives in Redis keys per provider instead of
instance attributes behind a lock, so failure counts and open/closed
state are shared across every serverless instance instead of each cold
start resetting to CLOSED.

Each transition uses a single atomic Redis primitive (INCR/SET/DELETE)
rather than one lock guarding a read-modify-write, so this is safe
under concurrent instances but not perfectly linearizable: failures
landing in the same instant can each independently decide to trip (or
transition to half-open), which is harmless since those writes are
idempotent, but can very occasionally let one extra probe call through
during half-open. That tradeoff is acceptable here -- a stray extra
probe costs one wasted provider call, not a correctness bug -- and it
avoids needing Lua scripting/WATCH for a component that mainly protects
against sustained failure, not split-second races.
"""

from __future__ import annotations

import logging
import time

from llm_gateway.breaker.circuit_breaker import BreakerState, CircuitOpenError
from llm_gateway.config import BreakerConfig

_logger = logging.getLogger("llm_gateway.breaker")
_NAMES_KEY = "breaker:names"


class RedisCircuitBreaker:
    def __init__(self, name: str, config: BreakerConfig, client):
        self.name = name
        self._config = config
        self._client = client

    def _key(self, suffix: str) -> str:
        return f"breaker:{self.name}:{suffix}"

    def _read_state(self) -> tuple[BreakerState, float | None]:
        raw_state = self._client.get(self._key("state"))
        state = BreakerState(raw_state) if raw_state else BreakerState.CLOSED
        raw_opened_at = self._client.get(self._key("opened_at"))
        opened_at = float(raw_opened_at) if raw_opened_at is not None else None
        return state, opened_at

    def _maybe_transition_to_half_open(self) -> BreakerState:
        state, opened_at = self._read_state()
        if (
            state == BreakerState.OPEN
            and opened_at is not None
            and time.time() - opened_at >= self._config.recovery_timeout_s
        ):
            self._client.set(self._key("state"), BreakerState.HALF_OPEN.value)
            self._client.delete(self._key("half_open_calls"))
            return BreakerState.HALF_OPEN
        return state

    @property
    def state(self) -> BreakerState:
        return self._maybe_transition_to_half_open()

    def before_call(self) -> None:
        state = self._maybe_transition_to_half_open()

        if state == BreakerState.OPEN:
            raise CircuitOpenError(f"circuit open for provider={self.name!r}")

        if state == BreakerState.HALF_OPEN:
            in_flight = self._client.incr(self._key("half_open_calls"))
            if in_flight > self._config.half_open_max_calls:
                raise CircuitOpenError(
                    f"circuit half-open probe already in flight for provider={self.name!r}"
                )

    def record_success(self) -> None:
        self._client.delete(
            self._key("failures"),
            self._key("state"),
            self._key("opened_at"),
            self._key("half_open_calls"),
        )

    def record_failure(self) -> None:
        state, _ = self._read_state()
        if state == BreakerState.HALF_OPEN:
            self._client.delete(self._key("half_open_calls"))
            self._trip()
            return

        count = self._client.incr(self._key("failures"))
        if count >= self._config.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._client.set(self._key("state"), BreakerState.OPEN.value)
        self._client.set(self._key("opened_at"), time.time())
        _logger.warning("circuit breaker tripped", extra={"fields": {"provider": self.name}})


class RedisCircuitBreakerRegistry:
    def __init__(self, config: BreakerConfig, client):
        self._config = config
        self._client = client
        self._breakers: dict[str, RedisCircuitBreaker] = {}

    def get(self, provider_name: str) -> RedisCircuitBreaker:
        breaker = self._breakers.get(provider_name)
        if breaker is None:
            breaker = RedisCircuitBreaker(provider_name, self._config, self._client)
            self._breakers[provider_name] = breaker
            self._client.sadd(_NAMES_KEY, provider_name)
        return breaker

    def snapshot(self) -> dict[str, BreakerState]:
        names = self._client.smembers(_NAMES_KEY) or set()
        return {name: self.get(name).state for name in names}
