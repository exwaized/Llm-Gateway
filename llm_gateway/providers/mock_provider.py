from __future__ import annotations

import random
import time

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.base import Provider, ProviderError, ProviderResponse


class MockProvider(Provider):
    """Simulated model endpoint for tests/demos — no network, no API keys.

    `failure_rate` and `forced_failures` let tests deterministically trip
    the circuit breaker without depending on a real provider being flaky.
    """

    def __init__(
        self,
        config: ProviderConfig,
        base_latency_s: float = 0.01,
        failure_rate: float = 0.0,
        forced_failures: int = 0,
    ):
        super().__init__(config)
        self.base_latency_s = base_latency_s
        self.failure_rate = failure_rate
        self._forced_failures_remaining = forced_failures
        self.call_count = 0

    def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        self.call_count += 1

        if self._forced_failures_remaining > 0:
            self._forced_failures_remaining -= 1
            raise ProviderError(f"{self.name}: forced failure (test injection)")

        if self.failure_rate and random.random() < self.failure_rate:
            raise ProviderError(f"{self.name}: simulated provider error")

        start = time.monotonic()
        time.sleep(self.base_latency_s)
        latency = time.monotonic() - start

        tokens_in = max(1, len(prompt.split()))
        tokens_out = max(1, tokens_in // 2)
        text = f"[{self.name}/{self.tier.value}] response to: {prompt[:60]!r}"

        return ProviderResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=latency,
            provider=self.name,
            tier=self.tier.value,
        )
