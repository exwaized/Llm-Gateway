from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from collections import Counter

from llm_gateway.evals.datasets import EvalItem


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()


class Scorer(ABC):
    """Scores a model answer against an item; returns a value in [0, 1]."""

    name: str

    @abstractmethod
    def score(self, item: EvalItem, answer: str) -> float:
        raise NotImplementedError


class ExactMatchScorer(Scorer):
    name = "exact_match"

    def score(self, item: EvalItem, answer: str) -> float:
        if item.reference is None:
            raise ValueError(f"item {item.id!r} has no reference for exact match")
        return float(_normalize(answer) == _normalize(item.reference))


class TokenF1Scorer(Scorer):
    name = "token_f1"

    def score(self, item: EvalItem, answer: str) -> float:
        if item.reference is None:
            raise ValueError(f"item {item.id!r} has no reference for token F1")
        pred = _normalize(answer).split()
        gold = _normalize(item.reference).split()
        if not pred or not gold:
            return float(pred == gold)
        common = sum((Counter(pred) & Counter(gold)).values())
        if common == 0:
            return 0.0
        precision, recall = common / len(pred), common / len(gold)
        return 2 * precision * recall / (precision + recall)
