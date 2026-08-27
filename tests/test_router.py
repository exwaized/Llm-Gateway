import pytest

from llm_gateway.breaker.circuit_breaker import CircuitBreakerRegistry
from llm_gateway.config import BreakerConfig, ClassifierThresholds, ProviderConfig, Tier
from llm_gateway.providers.mock_provider import MockProvider
from llm_gateway.router.router import AllProvidersExhaustedError, Router, tier_for_score


def make_router(**breaker_kwargs):
    cheap = MockProvider(ProviderConfig("cheap", Tier.CHEAP, 0.0001, 0.0002))
    mid = MockProvider(ProviderConfig("mid", Tier.MID, 0.001, 0.002))
    frontier = MockProvider(ProviderConfig("frontier", Tier.FRONTIER, 0.01, 0.03))
    providers_by_tier = {Tier.CHEAP: [cheap], Tier.MID: [mid], Tier.FRONTIER: [frontier]}
    registry = CircuitBreakerRegistry(BreakerConfig(**breaker_kwargs))
    router = Router(providers_by_tier, registry, ClassifierThresholds())
    return router, providers_by_tier


def test_tier_for_score_buckets():
    thresholds = ClassifierThresholds(cheap_max=0.35, mid_max=0.70)
    assert tier_for_score(0.1, thresholds) == Tier.CHEAP
    assert tier_for_score(0.5, thresholds) == Tier.MID
    assert tier_for_score(0.9, thresholds) == Tier.FRONTIER


def test_low_complexity_routes_to_cheap_tier():
    router, _ = make_router()
    result = router.route("hi", score=0.1)
    assert result.response.tier == "cheap"
    assert not result.escalated
    assert not result.degraded


def test_escalates_when_cheap_tier_provider_fails():
    cheap = MockProvider(ProviderConfig("cheap", Tier.CHEAP, 0.0001, 0.0002), forced_failures=1)
    mid = MockProvider(ProviderConfig("mid", Tier.MID, 0.001, 0.002))
    providers_by_tier = {Tier.CHEAP: [cheap], Tier.MID: [mid], Tier.FRONTIER: []}
    registry = CircuitBreakerRegistry(BreakerConfig(failure_threshold=5))
    router = Router(providers_by_tier, registry, ClassifierThresholds())

    result = router.route("hi", score=0.1)

    assert result.response.tier == "mid"
    assert result.escalated
    assert result.degraded
    assert result.attempts[0].outcome == "provider_error"
    assert result.attempts[-1].outcome == "success"


def test_raises_when_all_tiers_exhausted():
    failing = MockProvider(ProviderConfig("cheap", Tier.CHEAP, 0.0001, 0.0002), forced_failures=99)
    providers_by_tier = {Tier.CHEAP: [failing], Tier.MID: [], Tier.FRONTIER: []}
    registry = CircuitBreakerRegistry(BreakerConfig(failure_threshold=99))
    router = Router(providers_by_tier, registry, ClassifierThresholds())

    with pytest.raises(AllProvidersExhaustedError):
        router.route("hi", score=0.1)


def test_open_circuit_skips_provider_without_calling_it():
    _, providers = make_router()
    cheap = providers[Tier.CHEAP][0]

    registry = CircuitBreakerRegistry(BreakerConfig(failure_threshold=1, recovery_timeout_s=999))
    breaker = registry.get("cheap")
    breaker.before_call()
    breaker.record_failure()

    router = Router(providers, registry, ClassifierThresholds())
    result = router.route("hi", score=0.1)

    assert result.attempts[0].outcome == "circuit_open"
    assert cheap.call_count == 0
    assert result.response.tier == "mid"
