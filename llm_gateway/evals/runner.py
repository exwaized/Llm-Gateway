from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_gateway.evals.datasets import EvalItem
from llm_gateway.evals.scorers import Scorer
from llm_gateway.providers.base import Provider, ProviderError


class BudgetExceededError(Exception):
    """Raised when cumulative eval spend passes the configured cap."""


@dataclass
class EvalRecord:
    item_id: str
    provider: str
    tier: str
    answer: str | None
    score: float | None
    cost: float
    latency_s: float | None
    tokens_in: int
    tokens_out: int
    error: str | None = None


class EvalRunner:
    """Runs every item against every provider directly (no router, no cache).

    Results are appended to a JSONL file keyed by (item_id, provider), so a
    killed run resumes where it stopped and finished pairs are never re-billed.
    """

    def __init__(
        self,
        providers: list[Provider],
        scorer: Scorer,
        results_path: str | Path,
        max_cost: float | None = None,
    ):
        self._providers = providers
        self._scorer = scorer
        self._path = Path(results_path)
        self._max_cost = max_cost

    def _load_done(self) -> dict[tuple[str, str], EvalRecord]:
        done: dict[tuple[str, str], EvalRecord] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = EvalRecord(**json.loads(line))
                    done[(rec.item_id, rec.provider)] = rec
        return done

    def run(self, items: list[EvalItem]) -> list[EvalRecord]:
        done = self._load_done()
        spent = sum(r.cost for r in done.values())
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("a", encoding="utf-8") as out:
            for item in items:
                for provider in self._providers:
                    if (item.id, provider.name) in done:
                        continue
                    if self._max_cost is not None and spent >= self._max_cost:
                        raise BudgetExceededError(
                            f"spent ${spent:.4f} >= cap ${self._max_cost:.4f}; "
                            f"rerun to resume from {self._path}"
                        )
                    rec = self._call(item, provider)
                    spent += rec.cost
                    out.write(json.dumps(asdict(rec)) + "\n")
                    out.flush()
                    done[(item.id, provider.name)] = rec

        wanted = {(i.id, p.name) for i in items for p in self._providers}
        return [r for k, r in done.items() if k in wanted]

    def _call(self, item: EvalItem, provider: Provider) -> EvalRecord:
        tier = provider.tier.value
        try:
            resp = provider.complete(item.prompt)
        except ProviderError as exc:
            return EvalRecord(item.id, provider.name, tier, None, None, 0.0, None, 0, 0, str(exc))
        return EvalRecord(
            item_id=item.id,
            provider=provider.name,
            tier=tier,
            answer=resp.text,
            score=self._scorer.score(item, resp.text),
            cost=resp.cost(provider.config),
            latency_s=resp.latency_s,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
        )
