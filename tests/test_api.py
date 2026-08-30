import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import _state, app


@pytest.fixture
def client(monkeypatch):
    # Force the mock-provider fallback path and open (no-auth) mode
    # regardless of the host environment's own settings, so these tests
    # stay hermetic.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    monkeypatch.delenv("KV_REST_API_URL", raising=False)
    monkeypatch.delenv("KV_REST_API_TOKEN", raising=False)
    monkeypatch.delenv("GATEWAY_API_KEY", raising=False)
    with TestClient(app) as client:
        yield client


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_gateway_response(client):
    resp = client.post("/chat", json={"prompt": "What is the capital of France?"})
    assert resp.status_code == 200

    body = resp.json()
    assert body["text"]
    assert body["cache_hit"] is False
    assert body["tier"] in {"cheap", "mid", "frontier"}
    assert body["provider"] is not None
    assert body["cost_usd"] >= 0


def test_chat_repeated_prompt_hits_cache(client):
    client.post("/chat", json={"prompt": "Define 'idempotent'."})
    resp = client.post("/chat", json={"prompt": "Define 'idempotent'."})

    body = resp.json()
    assert body["cache_hit"] is True
    assert body["cost_usd"] == 0.0


def test_telemetry_and_breakers_reflect_traffic(client):
    client.post("/chat", json={"prompt": "hello there"})

    telemetry = client.get("/telemetry").json()
    assert telemetry["requests"] >= 1

    breakers = client.get("/breakers").json()
    assert all(state == "closed" for state in breakers.values())


def test_chat_rejects_oversized_prompt(client):
    resp = client.post("/chat", json={"prompt": "a" * 20000})
    assert resp.status_code == 422


def test_chat_rejects_empty_prompt(client):
    resp = client.post("/chat", json={"prompt": ""})
    assert resp.status_code == 422


def test_chat_requires_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret-token")

    unauthorized = client.post("/chat", json={"prompt": "hello"})
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/chat", json={"prompt": "hello"}, headers={"X-API-Key": "secret-token"}
    )
    assert authorized.status_code == 200


def test_healthz_never_requires_api_key(client, monkeypatch):
    monkeypatch.setenv("GATEWAY_API_KEY", "secret-token")
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_chat_rate_limit_returns_429_once_exceeded(client):
    from llm_gateway.ratelimit import RateLimiter

    # Swap in a 1-request-per-window limiter just for this test so it
    # doesn't depend on GATEWAY_RATE_LIMIT_PER_MINUTE's import-time value.
    _state["rate_limiter"] = RateLimiter(limit=1, window_s=60)

    first = client.post("/chat", json={"prompt": "first"})
    second = client.post("/chat", json={"prompt": "second"})

    assert first.status_code == 200
    assert second.status_code == 429
