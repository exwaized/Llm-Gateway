"""FastAPI wrapper around Gateway.

    uvicorn llm_gateway.api:app

Wires real OpenAI/Anthropic providers when the corresponding API key env
vars are set; falls back to MockProvider in every tier otherwise, so the
server runs out of the box with no keys configured (demo/dev/CI mode).

If UPSTASH_REDIS_REST_URL/TOKEN (or Vercel KV's KV_REST_API_URL/TOKEN)
are set, the semantic cache is backed by Upstash Redis instead of an
in-process dict, so cache hits survive serverless cold starts and are
shared across concurrent instances. Circuit-breaker state and telemetry
stay in-process either way -- see llm_gateway/cache/redis_cache.py for
why only the cache was worth persisting first.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_gateway import Gateway, GatewayConfig, ProviderConfig, Tier
from llm_gateway.cache.redis_cache import RedisSemanticCache, client_from_env
from llm_gateway.classifier import HeuristicClassifier
from llm_gateway.providers.base import Provider
from llm_gateway.providers.mock_provider import MockProvider
from llm_gateway.router.router import AllProvidersExhaustedError

_state: dict[str, Gateway] = {}


def build_gateway() -> Gateway:
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

    config = GatewayConfig()
    redis_client = client_from_env()
    cache = RedisSemanticCache(config.cache, redis_client) if redis_client else None

    return Gateway(config, HeuristicClassifier(), providers_by_tier, cache=cache)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["gateway"] = build_gateway()
    try:
        yield
    finally:
        _state.pop("gateway").shutdown()


def get_gateway() -> Gateway:
    return _state["gateway"]


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


class ChatRequest(BaseModel):
    prompt: str
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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = get_gateway().chat(request.prompt, scope_key=request.scope_key)
    except AllProvidersExhaustedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

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


@app.get("/telemetry")
def telemetry() -> dict[str, Any]:
    return get_gateway().telemetry.summary()


@app.get("/breakers")
def breakers() -> dict[str, str]:
    return {name: state.value for name, state in get_gateway().breakers.snapshot().items()}
