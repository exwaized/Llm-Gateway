from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ComplexityScore:
    score: float  # normalized to [0, 1]; higher = harder query
    features: dict = field(default_factory=dict)


class Classifier(ABC):
    @abstractmethod
    def score(self, prompt: str) -> ComplexityScore:
        raise NotImplementedError
