from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from llm_gateway.breaker.circuit_breaker import CircuitBreakerRegistry
from llm_gateway.cache.semantic_cache import SemanticCache
from llm_gateway.classifier.base import Classifier
from llm_gateway.config import GatewayConfig, ProviderConfig, Tier
from llm_gateway.providers.base import Provider
from llm_gateway.router.router import AllProvidersExhaustedError, RouteAttempt, Router
from llm_gateway.telemetry import RequestRecord, Telemetry


@dataclass
class GatewayResponse:
    text: str
    cache_hit: bool
    tier: str | None
    provider: str | None
    cost_usd: float
    latency_s: float
    escalated: bool = False
    degraded: bool = False
    attempts: list[RouteAttempt] = field(default_factory=list)


class Gateway:
    """Ties classifier -> semantic cache -> router -> providers together.

    Classification and the cache embedding/lookup are independent of each
    other, so they're dispatched to the shared ThreadPoolExecutor and
    joined before the routing decision is made — the same
    futures-fan-out-then-join pattern used for provider-health scans
    elsewhere in this codebase, kept here so cache lookups never block on
    classification (or vice versa) under load.
    """

    def __init__(
        self,
        config: GatewayConfig,
        classifier: Classifier,
        providers_by_tier: dict[Tier, list[Provider]],
        telemetry: Telemetry | None = None,
        cache: SemanticCache | None = None,
    ):
        self._config = config
        self._classifier = classifier
        self._providers_by_tier = providers_by_tier
        self._cache = cache or SemanticCache(config.cache)
        self._breakers = CircuitBreakerRegistry(config.breaker)
        self._router = Router(providers_by_tier, self._breakers, config.thresholds)
        self._telemetry = telemetry or Telemetry()
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)

        self._provider_configs: dict[str, ProviderConfig] = {
            provider.name: provider.config
            for providers in providers_by_tier.values()
            for provider in providers
        }

    @property
    def telemetry(self) -> Telemetry:
        return self._telemetry

    @property
    def breakers(self) -> CircuitBreakerRegistry:
        return self._breakers

    def chat(self, prompt: str, scope_key: str = "default", **kwargs) -> GatewayResponse:
        start = time.monotonic()

        score_future = self._executor.submit(self._classifier.score, prompt)
        cache_future = self._executor.submit(self._cache.lookup, prompt, scope_key)

        cache_result = cache_future.result()
        if cache_result.hit:
            score_future.cancel()  # best-effort; harmless if already running
            latency = time.monotonic() - start
            self._telemetry.record(
                RequestRecord(
                    prompt_preview=prompt[:60],
                    cache_hit=True,
                    tier=None,
                    provider=None,
                    cost_usd=0.0,
                    latency_s=latency,
                    escalated=False,
                )
            )
            return GatewayResponse(
                text=cache_result.entry.response_text,
                cache_hit=True,
                tier=None,
                provider=None,
                cost_usd=0.0,
                latency_s=latency,
            )

        complexity = score_future.result()

        try:
            route_result = self._router.route(prompt, complexity.score, **kwargs)
        except AllProvidersExhaustedError:
            latency = time.monotonic() - start
            self._telemetry.record(
                RequestRecord(
                    prompt_preview=prompt[:60],
                    cache_hit=False,
                    tier=None,
                    provider=None,
                    cost_usd=0.0,
                    latency_s=latency,
                    escalated=False,
                )
            )
            raise

        response = route_result.response
        self._cache.put(prompt, response.text, scope_key)

        provider_config = self._provider_configs[response.provider]
        cost = response.cost(provider_config)
        latency = time.monotonic() - start

        self._telemetry.record(
            RequestRecord(
                prompt_preview=prompt[:60],
                cache_hit=False,
                tier=response.tier,
                provider=response.provider,
                cost_usd=cost,
                latency_s=latency,
                escalated=route_result.escalated,
            )
        )

        return GatewayResponse(
            text=response.text,
            cache_hit=False,
            tier=response.tier,
            provider=response.provider,
            cost_usd=cost,
            latency_s=latency,
            escalated=route_result.escalated,
            degraded=route_result.degraded,
            attempts=route_result.attempts,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "Gateway":
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()
