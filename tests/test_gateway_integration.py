from llm_gateway import Gateway, GatewayConfig, ProviderConfig, Tier
from llm_gateway.classifier import HeuristicClassifier
from llm_gateway.providers.mock_provider import MockProvider


def build_gateway(**breaker_overrides):
    cheap = MockProvider(ProviderConfig("cheap", Tier.CHEAP, 0.0001, 0.0002))
    mid = MockProvider(ProviderConfig("mid", Tier.MID, 0.001, 0.002))
    frontier = MockProvider(ProviderConfig("frontier", Tier.FRONTIER, 0.01, 0.03))
    providers_by_tier = {Tier.CHEAP: [cheap], Tier.MID: [mid], Tier.FRONTIER: [frontier]}

    config = GatewayConfig()
    for key, value in breaker_overrides.items():
        setattr(config.breaker, key, value)

    gateway = Gateway(config, HeuristicClassifier(), providers_by_tier)
    return gateway, providers_by_tier


COMPLEX_PROMPT = (
    "Design a fault-tolerant distributed rate limiter and explain why a "
    "token-bucket approach is preferable to a sliding-window approach here. "
    "Also analyze the trade-offs around clock skew across nodes, and walk "
    "through how you would debug a scenario where the limiter under-throttles "
    "during a thundering herd. Refactor the reference implementation below "
    "to fix the race condition:\n"
    "```python\n"
    "class RateLimiter:\n"
    "    def __init__(self, capacity):\n"
    "        self.capacity = capacity\n"
    "        self.tokens = capacity\n"
    "    def allow(self):\n"
    "        if self.tokens > 0:\n"
    "            self.tokens -= 1\n"
    "            return True\n"
    "        return False\n"
    "```\n"
    "1. consider concurrency 2. consider fairness 3. consider recovery after an outage"
)


def test_simple_query_routes_to_cheap_tier_and_is_not_a_cache_hit():
    gateway, _ = build_gateway()
    with gateway:
        result = gateway.chat("What is the capital of France?")
        assert result.tier == "cheap"
        assert not result.cache_hit
        assert result.cost_usd > 0


def test_duplicate_query_is_served_from_cache():
    gateway, _ = build_gateway()
    with gateway:
        first = gateway.chat("What is the capital of France?")
        second = gateway.chat("What is the capital of France?")

        assert not first.cache_hit
        assert second.cache_hit
        assert second.cost_usd == 0.0
        assert second.text == first.text


def test_complex_query_routes_to_frontier_tier():
    gateway, _ = build_gateway()
    with gateway:
        result = gateway.chat(COMPLEX_PROMPT)
        assert result.tier == "frontier"


def test_provider_failure_triggers_escalation_and_is_tracked_in_telemetry():
    cheap = MockProvider(ProviderConfig("cheap", Tier.CHEAP, 0.0001, 0.0002), forced_failures=1)
    mid = MockProvider(ProviderConfig("mid", Tier.MID, 0.001, 0.002))
    providers_by_tier = {Tier.CHEAP: [cheap], Tier.MID: [mid], Tier.FRONTIER: []}
    config = GatewayConfig()

    with Gateway(config, HeuristicClassifier(), providers_by_tier) as gateway:
        result = gateway.chat("hi")

        assert result.tier == "mid"
        assert result.escalated

        summary = gateway.telemetry.summary()
        assert summary["requests"] == 1
        assert summary["escalation_rate"] == 1.0


def test_scope_key_prevents_cross_tenant_cache_leak():
    gateway, _ = build_gateway()
    with gateway:
        first = gateway.chat("cancel my subscription", scope_key="tenant-a")
        second = gateway.chat("cancel my subscription", scope_key="tenant-b")

        assert not first.cache_hit
        assert not second.cache_hit
