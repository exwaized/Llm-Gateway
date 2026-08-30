"""Contract tests for OpenAIProvider/AnthropicProvider.

Fakes the openai/anthropic SDK modules via sys.modules rather than
hitting real APIs or depending on exact installed SDK versions -- the
`from openai import OpenAI` / `from anthropic import Anthropic` imports
inside each provider's __init__ are lazy, so patching sys.modules
before construction is enough to intercept them regardless of whether
the real packages are installed.
"""

import sys
from types import SimpleNamespace

import pytest

from llm_gateway.config import ProviderConfig, Tier
from llm_gateway.providers.base import ProviderError


def _install_fake_openai(monkeypatch, create_fn):
    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create_fn))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))


def _install_fake_anthropic(monkeypatch, create_fn):
    class FakeAnthropic:
        def __init__(self, api_key=None):
            self.messages = SimpleNamespace(create=create_fn)

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))


def test_openai_provider_parses_successful_response(monkeypatch):
    def fake_create(**kwargs):
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi there"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
        )

    _install_fake_openai(monkeypatch, fake_create)
    from llm_gateway.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(
        ProviderConfig("openai-mini", Tier.CHEAP, 0.0001, 0.0002), model="gpt-4o-mini", api_key="test"
    )
    response = provider.complete("hello")

    assert response.text == "hi there"
    assert response.tokens_in == 3
    assert response.tokens_out == 2
    assert response.provider == "openai-mini"
    assert response.tier == "cheap"


def test_openai_provider_wraps_sdk_errors(monkeypatch):
    def fake_create(**kwargs):
        raise RuntimeError("rate limited")

    _install_fake_openai(monkeypatch, fake_create)
    from llm_gateway.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(ProviderConfig("openai-mini", Tier.CHEAP, 0.0001, 0.0002), model="gpt-4o-mini")

    with pytest.raises(ProviderError):
        provider.complete("hello")


def test_openai_provider_handles_empty_content_and_missing_usage(monkeypatch):
    def fake_create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None)

    _install_fake_openai(monkeypatch, fake_create)
    from llm_gateway.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(ProviderConfig("openai-mini", Tier.CHEAP, 0.0001, 0.0002), model="gpt-4o-mini")
    response = provider.complete("hello")

    assert response.text == ""
    assert response.tokens_in == 0
    assert response.tokens_out == 0


def test_anthropic_provider_parses_successful_response(monkeypatch):
    def fake_create(**kwargs):
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
        return SimpleNamespace(
            content=[SimpleNamespace(text="hi there")],
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        )

    _install_fake_anthropic(monkeypatch, fake_create)
    from llm_gateway.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(
        ProviderConfig("anthropic-haiku", Tier.MID, 0.0008, 0.004), model="claude-haiku-4-5", api_key="test"
    )
    response = provider.complete("hello")

    assert response.text == "hi there"
    assert response.tokens_in == 3
    assert response.tokens_out == 2
    assert response.provider == "anthropic-haiku"
    assert response.tier == "mid"


def test_anthropic_provider_wraps_sdk_errors(monkeypatch):
    def fake_create(**kwargs):
        raise RuntimeError("overloaded")

    _install_fake_anthropic(monkeypatch, fake_create)
    from llm_gateway.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider(ProviderConfig("anthropic-haiku", Tier.MID, 0.0008, 0.004), model="claude-haiku-4-5")

    with pytest.raises(ProviderError):
        provider.complete("hello")
