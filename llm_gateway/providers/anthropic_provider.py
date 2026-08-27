from __future__ import annotations

import time

from llm_gateway.config import ProviderConfig
from llm_gateway.providers.base import Provider, ProviderError, ProviderResponse


class AnthropicProvider(Provider):
    """Thin adapter over the Anthropic SDK. Requires `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(
        self,
        config: ProviderConfig,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ):
        super().__init__(config)
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicProvider requires the 'anthropic' package: pip install anthropic"
            ) from exc
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        start = time.monotonic()
        try:
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=kwargs.pop("max_tokens", self._max_tokens),
                messages=[{"role": "user", "content": prompt}],
                timeout=self.config.timeout_s,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - normalize any SDK error to ProviderError
            raise ProviderError(f"{self.name}: {exc}") from exc
        latency = time.monotonic() - start

        text = "".join(block.text for block in resp.content if hasattr(block, "text"))

        return ProviderResponse(
            text=text,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            latency_s=latency,
            provider=self.name,
            tier=self.tier.value,
        )
