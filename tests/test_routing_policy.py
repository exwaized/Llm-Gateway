from llm_gateway.breaker.circuit_breaker import CircuitBreakerRegistry
from llm_gateway.config import BreakerConfig, ClassifierThresholds, ProviderConfig, Tier
from llm_gateway.providers.mock_provider import MockProvider
from llm_gateway.router import Router, RoutingPolicy, ThresholdPolicy


def _providers():
    return {
        t: [MockProvider(ProviderConfig(f"{t.value}-p", t, 0.001, 0.002), base_latency_s=0)]
        for t in Tier
    }


def test_threshold_policy_matches_thresholds():
    policy = ThresholdPolicy(ClassifierThresholds(cheap_max=0.2, mid_max=0.6))
    assert policy.choose_tier("x", 0.1) == Tier.CHEAP
    assert policy.choose_tier("x", 0.3) == Tier.MID
    assert policy.choose_tier("x", 0.9) == Tier.FRONTIER


def test_router_uses_custom_policy_over_score():
    class AlwaysFrontier(RoutingPolicy):
        def choose_tier(self, prompt, score):
            return Tier.FRONTIER

    router = Router(_providers(), CircuitBreakerRegistry(BreakerConfig()), policy=AlwaysFrontier())
    result = router.route("simple question", 0.0)
    assert result.response.tier == "frontier"


def test_router_default_policy_unchanged():
    router = Router(_providers(), CircuitBreakerRegistry(BreakerConfig()))
    assert router.route("q", 0.0).response.tier == "cheap"
