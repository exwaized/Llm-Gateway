from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")

SparseVector = dict[str, float]


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> SparseVector:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Bag-of-words term-frequency vector, L2-normalized.

    No model download, no API call, no numpy — good enough to catch
    near-duplicate prompts for cache purposes. Swap in a real embedding
    model (OpenAI/Cohere/sentence-transformers) behind the same interface
    when recall on paraphrases actually matters; this implementation will
    miss "cancel my subscription" == "please cancel my plan" but will
    catch "cancel my subscription" == "cancel my subscription please".
    """

    def embed(self, text: str) -> SparseVector:
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return {}
        counts = Counter(tokens)
        norm = math.sqrt(sum(c * c for c in counts.values()))
        return {term: c / norm for term, c in counts.items()}


def cosine_similarity(a: SparseVector, b: SparseVector) -> float:
    if not a or not b:
        return 0.0
    # iterate the smaller dict for speed
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())
