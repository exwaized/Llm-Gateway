"""Runs the gateway end-to-end with mock providers -- no API keys needed.

    python examples/run_demo.py
"""

from llm_gateway import Gateway, GatewayConfig, ProviderConfig, Tier
from llm_gateway.classifier import HeuristicClassifier
from llm_gateway.providers.mock_provider import MockProvider

PROMPTS = [
    "What is the capital of France?",
    "What is the capital of France?",  # exact duplicate -> should hit cache, cost $0
    "Compare the trade-offs between optimistic and pessimistic locking, and "
    "explain why one would be preferred for a high-contention order-matching engine.",
    "Refactor this function to reduce cyclomatic complexity:\n```python\ndef f(x):\n    if x: return 1\n```",
    "Define 'idempotent'.",  # simple again -> should route back to the cheap tier
]


def build_gateway() -> Gateway:
    cheap = MockProvider(
        ProviderConfig("mock-cheap", Tier.CHEAP, cost_per_1k_input=0.0001, cost_per_1k_output=0.0002),
        # the very first call to the cheap tier fails, to demonstrate router
        # fallback/escalation without fully tripping the circuit breaker
        forced_failures=1,
    )
    mid = MockProvider(
        ProviderConfig("mock-mid", Tier.MID, cost_per_1k_input=0.001, cost_per_1k_output=0.002),
    )
    frontier = MockProvider(
        ProviderConfig("mock-frontier", Tier.FRONTIER, cost_per_1k_input=0.01, cost_per_1k_output=0.03),
    )

    providers_by_tier = {
        Tier.CHEAP: [cheap],
        Tier.MID: [mid],
        Tier.FRONTIER: [frontier],
    }

    config = GatewayConfig()
    config.breaker.failure_threshold = 3  # one flaky call shouldn't trip the breaker

    return Gateway(config, HeuristicClassifier(), providers_by_tier)


def main() -> None:
    with build_gateway() as gateway:
        for prompt in PROMPTS:
            result = gateway.chat(prompt)
            print("-" * 72)
            print(f"prompt:      {prompt[:70]!r}")
            print(f"cache_hit:   {result.cache_hit}")
            print(f"tier:        {result.tier}")
            print(f"provider:    {result.provider}")
            print(f"escalated:   {result.escalated}")
            print(f"cost_usd:    {result.cost_usd:.6f}")
            print(f"latency_s:   {result.latency_s:.4f}")
            print(f"text:        {result.text}")

        print("=" * 72)
        print("telemetry summary:")
        for key, value in gateway.telemetry.summary().items():
            print(f"  {key}: {value}")

        print("breaker states:", gateway.breakers.snapshot())


if __name__ == "__main__":
    main()
