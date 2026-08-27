from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_gateway.config import ProviderConfig


class ProviderError(Exception):
    """Raised when a provider call fails (timeout, 5xx, rate limit, etc.)."""


@dataclass
class ProviderResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    provider: str
    tier: str

    def cost(self, config: ProviderConfig) -> float:
        return (
            self.tokens_in / 1000 * config.cost_per_1k_input
            + self.tokens_out / 1000 * config.cost_per_1k_output
        )


class Provider(ABC):
    """A single model endpoint. One Provider instance == one (vendor, model) pair."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def tier(self):
        return self.config.tier

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> ProviderResponse:
        """Send prompt to the model. Must raise ProviderError on failure."""
        raise NotImplementedError
