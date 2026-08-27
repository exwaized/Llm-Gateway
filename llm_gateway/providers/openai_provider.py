from __future__ import annotations

import time

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.base import Provider, ProviderError, ProviderResponse


class OpenAIProvider(Provider):
    """Thin adapter over the OpenAI SDK. Requires `pip install openai` and OPENAI_API_KEY."""

    def __init__(self, config: ProviderConfig, model: str, api_key: str | None = None):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAIProvider requires the 'openai' package: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.config.timeout_s,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - normalize any SDK error to ProviderError
            raise ProviderError(f"{self.name}: {exc}") from exc
        latency = time.monotonic() - start

        choice = resp.choices[0].message.content or ""
        usage = resp.usage

        return ProviderResponse(
            text=choice,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            latency_s=latency,
            provider=self.name,
            tier=self.tier.value,
        )
