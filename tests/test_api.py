import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import app


@pytest.fixture
def client(monkeypatch):
    # Force the mock-provider fallback path regardless of the host
    # environment's own API keys, so these tests stay hermetic.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
