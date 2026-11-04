"""Benchmark the gateway on REAL prompts: databricks-dolly-15k (15,011 human-written
instructions across 8 task categories, CC BY-SA 3.0), instead of a hand-built prompt mix.

    python examples/benchmark_real.py path/to/databricks-dolly-15k.jsonl [--n 3000] [--seed 7]

What is real and what is modelled
  real     : the prompts (text, length, category, and the natural lack of repeats)
  modelled : model replies and prices. MockProvider returns canned text and the per-token
             prices are the same representative cheap/mid/frontier prices as benchmark.py,
             so dollar figures are a cost model, not a bill.
  NOT measured: answer quality. Routing a prompt to a cheaper tier saves money only if the
             cheaper model answers it well enough; this benchmark cannot tell you that.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from benchmark import build_gateway, frontier_only_cost  # same pricing/gateway as the synthetic benchmark


def load_prompts(path: str, n: int, seed: int):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows[:n]:
        prompt = r["instruction"] if not r.get("context") else f'{r["instruction"]}\n\n{r["context"]}'
        out.append((prompt, r["category"]))
    return out


def run(path: str, n: int, seed: int) -> None:
    data = load_prompts(path, n, seed)
    prompts = [p for p, _ in data]
    by_cat_tier: dict[str, Counter] = defaultdict(Counter)
    with build_gateway() as gw:
        for prompt, cat in data:
            res = gw.chat(prompt)
            by_cat_tier[cat]["cache" if res.cache_hit else res.tier] += 1
        s = gw.telemetry.summary()

    base = frontier_only_cost(prompts)
    cost = s["total_cost_usd"]
    n_unique = len(set(prompts))
    print("=" * 76)
    print(f"{len(prompts)} real prompts sampled from dolly-15k (seed={seed}), {n_unique} unique")
    print("=" * 76)
    print(f"modelled cost, gateway:        ${cost:.4f}")
    print(f"modelled cost, frontier-only:  ${base:.4f}")
    print(f"modelled savings:              {(base - cost) / base * 100:.1f}%")
    print(f"cache hit rate:                {s['cache_hit_rate'] * 100:.1f}%")
    print(f"escalation rate:               {s['escalation_rate'] * 100:.1f}%")
    print(f"tier distribution:             {s['tier_distribution']}")
    print("-" * 76)
    print("routing by Dolly task category (share of requests):")
    for cat, c in sorted(by_cat_tier.items()):
        tot = sum(c.values())
        print(f"  {cat:<22}{tot:>6}  " + "  ".join(f"{k}={v / tot:.0%}" for k, v in sorted(c.items())))
    print("=" * 76)
    print(json.dumps({"n": len(prompts), "unique": n_unique, "gateway_cost_usd": round(cost, 4),
                      "frontier_only_cost_usd": round(base, 4),
                      "savings_pct": round((base - cost) / base * 100, 1),
                      "cache_hit_rate": round(s["cache_hit_rate"], 4),
                      "tier_distribution": s["tier_distribution"]}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); run(a.path, a.n, a.seed)
