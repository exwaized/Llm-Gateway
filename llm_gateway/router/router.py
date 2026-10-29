from __future__ import annotations

from dataclasses import dataclass, field

from llm_gateway.breaker.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from llm_gateway.config import TIER_ORDER, ClassifierThresholds, Tier
from llm_gateway.providers.base import Provider, ProviderError, ProviderResponse
from llm_gateway.router.policies import RoutingPolicy, ThresholdPolicy, tier_for_score  # noqa: F401


class AllProvidersExhaustedError(Exception):
    """Raised when every provider in every tier on the fallback chain failed or was open."""


@dataclass
class RouteAttempt:
    provider: str
    tier: str
    outcome: str  # "circuit_open" | "provider_error" | "success"
    error: str | None = None


@dataclass
class RouteResult:
    response: ProviderResponse
    attempts: list[RouteAttempt] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """True if we didn't succeed on the first attempt (fallback/escalation kicked in)."""
        return len(self.attempts) > 1

    @property
    def escalated(self) -> bool:
        """True if the successful tier differs from the first attempted tier."""
        if not self.attempts:
            return False
        return self.attempts[0].tier != self.attempts[-1].tier


class Router:
    """Picks a tier from a complexity score, then dispatches with fallback.

    Fallback policy: within a tier, try each registered provider in order
    (redundancy at equal cost). If every provider in a tier is open/failing,
    escalate to the next tier up rather than fail the request outright —
    reliability wins over strict cost adherence. Escalation is recorded on
    the RouteResult so callers/telemetry can tell "answered as requested"
    apart from "answered, but had to pay more because of a provider outage".
    """

    def __init__(
        self,
        providers_by_tier: dict[Tier, list[Provider]],
        breaker_registry: CircuitBreakerRegistry,
        thresholds: ClassifierThresholds | None = None,
        policy: RoutingPolicy | None = None,
    ):
        self._providers_by_tier = providers_by_tier
        self._breakers = breaker_registry
        # An explicit policy wins; otherwise keep the original threshold behaviour.
        self._policy = policy or ThresholdPolicy(thresholds)

    def _tiers_from(self, start_tier: Tier) -> list[Tier]:
        start_idx = TIER_ORDER.index(start_tier)
        return TIER_ORDER[start_idx:]

    def route(self, prompt: str, score: float, **kwargs) -> RouteResult:
        start_tier = self._policy.choose_tier(prompt, score)
        attempts: list[RouteAttempt] = []

        for tier in self._tiers_from(start_tier):
            for provider in self._providers_by_tier.get(tier, []):
                breaker = self._breakers.get(provider.name)
                try:
                    breaker.before_call()
                except CircuitOpenError as exc:
                    attempts.append(
                        RouteAttempt(provider.name, tier.value, "circuit_open", str(exc))
                    )
                    continue

                try:
                    response = provider.complete(prompt, **kwargs)
                except ProviderError as exc:
                    breaker.record_failure()
                    attempts.append(
                        RouteAttempt(provider.name, tier.value, "provider_error", str(exc))
                    )
                    continue

                breaker.record_success()
                attempts.append(RouteAttempt(provider.name, tier.value, "success"))
                return RouteResult(response=response, attempts=attempts)

        raise AllProvidersExhaustedError(
            f"no provider available from tier={start_tier.value} through frontier; "
            f"attempts={attempts}"
        )
