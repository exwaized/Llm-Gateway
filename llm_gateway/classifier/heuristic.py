from __future__ import annotations

import re

from llm_gateway.classifier.base import Classifier, ComplexityScore

_REASONING_KEYWORDS = (
    "compare", "explain why", "analyze", "analyse", "step by step",
    "prove", "derive", "optimize", "optimise", "architecture", "design a",
    "debug", "refactor", "trade-off", "tradeoff", "root cause", "why does",
    "walk through", "reason about",
)

_SIMPLE_KEYWORDS = (
    "what is", "define", "translate", "capital of", "spell", "who is",
    "when was", "what year", "convert", "synonym for",
)

_CODE_HINTS = re.compile(r"```|def |class |function |import |=>|;\s*$|\{|\}", re.MULTILINE)

_MULTI_PART_HINTS = re.compile(r"\band\b|\balso\b|^\s*\d+[.)]", re.IGNORECASE | re.MULTILINE)


class HeuristicClassifier(Classifier):
    """Zero-dependency complexity scorer: length + lexical signals.

    Cheap enough to run on every request (microseconds), so it's safe to
    call inline before deciding whether the request even needs a model
    call at all. Intended as the default; swap in a trained/embedding
    classifier later behind the same `Classifier` interface without
    touching the router.
    """

    def __init__(
        self,
        length_weight: float = 0.30,
        reasoning_weight: float = 0.30,
        code_weight: float = 0.20,
        multi_part_weight: float = 0.20,
        simple_penalty: float = 0.35,
        length_norm_tokens: int = 150,
    ):
        self.length_weight = length_weight
        self.reasoning_weight = reasoning_weight
        self.code_weight = code_weight
        self.multi_part_weight = multi_part_weight
        self.simple_penalty = simple_penalty
        self.length_norm_tokens = length_norm_tokens

    def score(self, prompt: str) -> ComplexityScore:
        lower = prompt.lower()
        tokens = prompt.split()

        length_score = min(len(tokens) / self.length_norm_tokens, 1.0)

        reasoning_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in lower)
        reasoning_score = min(reasoning_hits / 2, 1.0)

        code_hits = len(_CODE_HINTS.findall(prompt))
        code_score = min(code_hits / 3, 1.0)

        multi_part_hits = len(_MULTI_PART_HINTS.findall(prompt))
        multi_part_score = min(multi_part_hits / 4, 1.0)

        simple_hits = sum(1 for kw in _SIMPLE_KEYWORDS if kw in lower)

        raw = (
            self.length_weight * length_score
            + self.reasoning_weight * reasoning_score
            + self.code_weight * code_score
            + self.multi_part_weight * multi_part_score
        )
        if simple_hits and len(tokens) < 20:
            raw -= self.simple_penalty

        score = max(0.0, min(1.0, raw))

        return ComplexityScore(
            score=score,
            features={
                "tokens": len(tokens),
                "length_score": length_score,
                "reasoning_hits": reasoning_hits,
                "code_hits": code_hits,
                "multi_part_hits": multi_part_hits,
                "simple_hits": simple_hits,
            },
        )
