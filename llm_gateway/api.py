"""FastAPI wrapper around Gateway.

    uvicorn llm_gateway.api:app

Wires real OpenAI/Anthropic providers when the corresponding API key env
vars are set; falls back to MockProvider in every tier otherwise, so the
server runs out of the box with no keys configured (demo/dev/CI mode).

If UPSTASH_REDIS_REST_URL/TOKEN (or Vercel KV's KV_REST_API_URL/TOKEN)
are set, the semantic cache, circuit-breaker state, telemetry, and the
rate limiter are all backed by that same Upstash Redis instance instead
of in-process state, so they survive serverless cold starts and are
shared across concurrent instances. Without it, everything falls back
to in-process state scoped to a single instance -- fine for local dev.

Auth: set GATEWAY_API_KEY to require a matching `X-API-Key` header on
/chat, /telemetry, and /breakers (not /healthz, so health checks/load
balancer probes keep working unauthenticated). Left unset, those routes
stay open -- fine for local dev, not for a public deployment.

Rate limiting: GATEWAY_RATE_LIMIT_PER_MINUTE (default 60) caps /chat
calls per client IP per 60s window. GATEWAY_MAX_PROMPT_CHARS (default
16000) caps prompt size, bounding worst-case cost/latency per request.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from llm_gateway import ClassifierThresholds, Gateway, GatewayConfig, ProviderConfig, Tier
from llm_gateway.breaker.redis_breaker import RedisCircuitBreakerRegistry
from llm_gateway.cache.redis_cache import RedisSemanticCache, client_from_env
from llm_gateway.classifier import HeuristicClassifier
from llm_gateway.logging_config import configure_logging
from llm_gateway.providers.base import Provider
from llm_gateway.providers.mock_provider import MockProvider
from llm_gateway.ratelimit import RateLimiter
from llm_gateway.redis_telemetry import RedisTelemetry
from llm_gateway.router.router import AllProvidersExhaustedError

_logger = logging.getLogger("llm_gateway.api")

_MAX_PROMPT_CHARS = int(os.environ.get("GATEWAY_MAX_PROMPT_CHARS", "16000"))
_RATE_LIMIT_PER_MINUTE = int(os.environ.get("GATEWAY_RATE_LIMIT_PER_MINUTE", "60"))

_state: dict[str, Any] = {}


def build_gateway(redis_client=None) -> Gateway:
    providers_by_tier: dict[Tier, list[Provider]] = {Tier.CHEAP: [], Tier.MID: [], Tier.FRONTIER: []}

    if os.environ.get("OPENAI_API_KEY"):
        from llm_gateway.providers.openai_provider import OpenAIProvider

        providers_by_tier[Tier.CHEAP].append(
            OpenAIProvider(
                ProviderConfig(
                    "openai-gpt-4o-mini", Tier.CHEAP, cost_per_1k_input=0.00015, cost_per_1k_output=0.0006
                ),
                model="gpt-4o-mini",
            )
        )
        providers_by_tier[Tier.FRONTIER].append(
            OpenAIProvider(
                ProviderConfig(
                    "openai-gpt-4o", Tier.FRONTIER, cost_per_1k_input=0.0025, cost_per_1k_output=0.01
                ),
                model="gpt-4o",
            )
        )

    if os.environ.get("ANTHROPIC_API_KEY"):
        from llm_gateway.providers.anthropic_provider import AnthropicProvider

        providers_by_tier[Tier.MID].append(
            AnthropicProvider(
                ProviderConfig(
                    "anthropic-haiku", Tier.MID, cost_per_1k_input=0.0008, cost_per_1k_output=0.004
                ),
                model="claude-haiku-4-5",
            )
        )
        providers_by_tier[Tier.FRONTIER].append(
            AnthropicProvider(
                ProviderConfig(
                    "anthropic-sonnet", Tier.FRONTIER, cost_per_1k_input=0.003, cost_per_1k_output=0.015
                ),
                model="claude-sonnet-5",
            )
        )

    if not any(providers_by_tier.values()):
        providers_by_tier = {
            Tier.CHEAP: [MockProvider(ProviderConfig("mock-cheap", Tier.CHEAP, 0.0001, 0.0002))],
            Tier.MID: [MockProvider(ProviderConfig("mock-mid", Tier.MID, 0.001, 0.002))],
            Tier.FRONTIER: [MockProvider(ProviderConfig("mock-frontier", Tier.FRONTIER, 0.01, 0.03))],
        }

    config = GatewayConfig(thresholds=ClassifierThresholds.from_env())

    cache = RedisSemanticCache(config.cache, redis_client) if redis_client else None
    breakers = RedisCircuitBreakerRegistry(config.breaker, redis_client) if redis_client else None
    telemetry = RedisTelemetry(redis_client) if redis_client else None

    return Gateway(
        config, HeuristicClassifier(), providers_by_tier, telemetry=telemetry, cache=cache, breakers=breakers
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    redis_client = client_from_env()
    _state["gateway"] = build_gateway(redis_client)
    _state["rate_limiter"] = RateLimiter(limit=_RATE_LIMIT_PER_MINUTE, window_s=60, client=redis_client)
    try:
        yield
    finally:
        _state.pop("gateway").shutdown()
        _state.pop("rate_limiter", None)


def get_gateway() -> Gateway:
    return _state["gateway"]


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.environ.get("GATEWAY_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="missing or invalid API key")


def enforce_rate_limit(request: Request) -> None:
    limiter: RateLimiter = _state["rate_limiter"]
    client_host = request.client.host if request.client else "unknown"
    if not limiter.allow(client_host):
        raise HTTPException(status_code=429, detail="rate limit exceeded, try again shortly")


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=_MAX_PROMPT_CHARS)
    scope_key: str = "default"


class ChatResponse(BaseModel):
    text: str
    cache_hit: bool
    tier: str | None
    provider: str | None
    cost_usd: float
    latency_s: float
    escalated: bool
    degraded: bool


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = get_gateway().chat(request.prompt, scope_key=request.scope_key)
    except AllProvidersExhaustedError as exc:
        _logger.warning("all providers exhausted", extra={"fields": {"scope_key": request.scope_key}})
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _logger.info(
        "chat request served",
        extra={
            "fields": {
                "cache_hit": result.cache_hit,
                "tier": result.tier,
                "provider": result.provider,
                "cost_usd": result.cost_usd,
                "latency_s": result.latency_s,
                "escalated": result.escalated,
                "scope_key": request.scope_key,
            }
        },
    )

    return ChatResponse(
        text=result.text,
        cache_hit=result.cache_hit,
        tier=result.tier,
        provider=result.provider,
        cost_usd=result.cost_usd,
        latency_s=result.latency_s,
        escalated=result.escalated,
        degraded=result.degraded,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/telemetry", dependencies=[Depends(require_api_key)])
def telemetry() -> dict[str, Any]:
    return get_gateway().telemetry.summary()


@app.get("/breakers", dependencies=[Depends(require_api_key)])
def breakers() -> dict[str, str]:
    return {name: state.value for name, state in get_gateway().breakers.snapshot().items()}
