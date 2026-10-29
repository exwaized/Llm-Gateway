import json

import pytest

from llm_gateway.config import ProviderConfig, Tier
from llm_gateway.evals import (
    BudgetExceededError,
    EvalItem,
    EvalRunner,
    ExactMatchScorer,
    TokenF1Scorer,
    load_jsonl,
)
from llm_gateway.providers.mock_provider import MockProvider


def _provider(name="cheap-p", tier=Tier.CHEAP):
    return MockProvider(ProviderConfig(name, tier, 0.001, 0.002), base_latency_s=0)


def test_load_jsonl_and_duplicate_ids(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps({"id": 1, "prompt": "p", "reference": "r"}) + "\n\n")
    assert load_jsonl(f)[0].id == "1"
    f.write_text(json.dumps({"id": 1, "prompt": "p"}) + "\n" + json.dumps({"id": 1, "prompt": "q"}))
    with pytest.raises(ValueError, match="duplicate"):
        load_jsonl(f)


def test_exact_match_normalizes():
    item = EvalItem("a", "q", reference="Hello, World!")
    assert ExactMatchScorer().score(item, " hello world ") == 1.0
    assert ExactMatchScorer().score(item, "goodbye") == 0.0


def test_token_f1_partial_credit():
    item = EvalItem("a", "q", reference="the cat sat")
    assert TokenF1Scorer().score(item, "the cat") == pytest.approx(0.8)


def test_runner_resumes_without_rebilling(tmp_path):
    items = [EvalItem(str(i), f"q{i}", reference="x") for i in range(3)]
    provider = _provider()
    path = tmp_path / "r.jsonl"
    EvalRunner([provider], ExactMatchScorer(), path).run(items)
    assert provider.call_count == 3
    records = EvalRunner([provider], ExactMatchScorer(), path).run(items)
    assert provider.call_count == 3
    assert len(records) == 3


def test_runner_budget_cap_then_resume(tmp_path):
    items = [EvalItem(str(i), "word " * 50, reference="x") for i in range(5)]
    provider = _provider()
    path = tmp_path / "r.jsonl"
    with pytest.raises(BudgetExceededError):
        EvalRunner([provider], ExactMatchScorer(), path, max_cost=0.0001).run(items)
    done = len(path.read_text().splitlines())
    assert 0 < done < 5
    assert len(EvalRunner([provider], ExactMatchScorer(), path).run(items)) == 5


def test_runner_records_provider_errors(tmp_path):
    provider = MockProvider(
        ProviderConfig("bad", Tier.CHEAP, 0.001, 0.002), base_latency_s=0, forced_failures=1
    )
    rec = EvalRunner([provider], ExactMatchScorer(), tmp_path / "r.jsonl").run(
        [EvalItem("a", "q", reference="x")]
    )[0]
    assert rec.error and rec.score is None
