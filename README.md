# LLM Gateway

[![CI](https://github.com/exwaized/Llm-Gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/exwaized/Llm-Gateway/actions/workflows/ci.yml)

A cost-aware LLM inference router: it scores each request for complexity,
routes it to the cheapest model tier that can plausibly handle it, serves
repeat/near-duplicate requests from a semantic cache instead of calling a
model at all, and falls back to a healthier tier when a provider is failing
— all in ~700 lines of dependency-free Python.

Zero required dependencies. Runs entirely offline against `MockProvider` for
tests and demos; `openai`/`anthropic` are optional extras for real traffic.

## Why this exists

Naive LLM integrations send every request to one model. That's simple but
wastes money at scale: a one-line factual lookup and a multi-step debugging
request cost the same, even though a cheap model handles the first one
just as well. This gateway sits between an application and its model
providers and makes three decisions per request that a naive integration
doesn't:

1. **Is this a repeat?** — check a semantic cache before spending anything.
2. **How hard is this, really?** — score complexity and pick the cheapest
   tier likely to succeed, instead of defaulting to the biggest model.
3. **Is the chosen provider currently healthy?** — a per-provider circuit
   breaker skips known-bad providers instantly and escalates to the next
   tier rather than failing the request outright.

See [`docs/why-cost-matters.md`](docs/why-cost-matters.md) for the case that
this problem doesn't go away as context windows grow and per-token prices
fall.

## Architecture

```
request
   │
   ├─────────────┬─────────────────────┐
   │              │                     │
   ▼              ▼                     │
classifier   semantic cache             │   (dispatched concurrently;
(complexity   (embedding similarity     │    cache lookup never blocks
   score)      within a scope_key)      │    on classification)
   │              │                     │
   │        hit ──┴── return cached response, cost = $0
   │
   ▼
router: tier_for_score(score) → try providers in that tier
   │
   ├── success ─────────────────────────► response + telemetry
   │
   └── circuit open / provider error ──► escalate to next tier up
                                          (repeat until a tier succeeds
                                           or every tier is exhausted)
```

- **`classifier/`** — pluggable `Classifier` interface. Ships with a
  zero-dependency heuristic scorer (length + reasoning/code/multi-part
  lexical signals); swap in an embedding or trained classifier behind the
  same interface without touching the router.
- **`cache/`** — `SemanticCache` with a pluggable `Embedder` (ships with a
  dependency-free hashing bag-of-words embedder). Scoped by `scope_key` so
  cache hits never leak across tenants/system-prompts. TTL + FIFO eviction.
- **`router/`** — maps a complexity score to a starting tier, then tries
  each provider in that tier before escalating up the tier ladder. Every
  attempt (success, provider error, circuit-open skip) is recorded so
  telemetry can distinguish "answered as routed" from "degraded and had to
  pay more because a provider was down."
- **`breaker/`** — per-provider circuit breaker (closed → open → half-open),
  so a failing provider is skipped immediately instead of retried into a
  timeout on every request.
- **`telemetry.py`** — thread-safe request accounting: cache hit rate, cost,
  tier distribution, escalation rate.

## Results

Two runnable, offline demos back the numbers below — nothing here requires
API keys.

### Cost: gateway vs. always-call-the-frontier-model

`python examples/benchmark.py` replays a synthetic 300-request support/
assistant workload (55% simple lookups, 30% moderate comparisons, 15%
genuinely hard multi-step/debugging asks, FAQ-skewed via Zipf sampling —
see the script for the exact composition) through the gateway, and compares
it to the cost of sending every request straight to the frontier tier with
no cache:

```
========================================================================
workload: 300 requests (seed=7), wall time 0.367s
========================================================================
                                       gateway    frontier-only baseline
total cost                             $0.0241                   $0.3646
avg cost / request                   $0.000080                 $0.001215
------------------------------------------------------------------------
cost savings:            $0.3404  (93.4% cheaper than frontier-only)
cache hit rate:           78.3%
escalation rate:          0.0%  (cheap/mid failed over to a higher tier)
tier distribution:        {'mid': 3, 'frontier': 5, 'cheap': 57}
latency avg / p95:        1.20ms / 3.05ms
========================================================================
```

Per-tier pricing in the benchmark tracks real-world cheap/mid/frontier
model pricing as of 2026 (`examples/benchmark.py`, `*_PRICE` constants) —
the $ figures are representative of real spend even though the calls
themselves are simulated. The 93.4% figure is workload-dependent (it moves
with cache hit rate and how much traffic is actually simple); re-run with
your own prompt mix to see how it moves.

### Resilience: fallback and escalation under a provider failure

`python examples/run_demo.py` forces the cheap-tier provider's first call
to fail, then sends five prompts through the gateway:

```
prompt:      'What is the capital of France?'
tier:        mid          # escalated from cheap after cheap's call failed
escalated:   True
cost_usd:    0.000012

prompt:      'What is the capital of France?'  (exact repeat)
cache_hit:   True          # served from cache, $0, no model call
cost_usd:    0.000000

...

telemetry summary:
  cache_hit_rate: 0.2
  escalation_rate: 0.2
  tier_distribution: {'mid': 3, 'cheap': 1}
breaker states: {'mock-cheap': CLOSED, 'mock-mid': CLOSED}
```

The circuit breaker recorded the failure but didn't trip (one flaky call is
below `failure_threshold`), so the very next simple prompt still routes back
to the cheap tier — a real outage would need `failure_threshold`
consecutive failures before that provider is skipped entirely.

## Test suite

```
$ pytest
........................                                          [100%]
24 passed in 0.79s
```

Covers circuit breaker state transitions, semantic cache hit/miss/TTL/
eviction, the heuristic classifier's scoring, router tier-selection and
escalation, and end-to-end gateway behavior (`tests/`). Runs in CI on every
push — see the badge above.

## Usage

```python
from llm_gateway import Gateway, GatewayConfig, ProviderConfig, Tier
from llm_gateway.classifier import HeuristicClassifier
from llm_gateway.providers.mock_provider import MockProvider  # or openai_provider / anthropic_provider

providers_by_tier = {
    Tier.CHEAP: [MockProvider(ProviderConfig("cheap", Tier.CHEAP, 0.00015, 0.0006))],
    Tier.MID: [MockProvider(ProviderConfig("mid", Tier.MID, 0.0025, 0.01))],
    Tier.FRONTIER: [MockProvider(ProviderConfig("frontier", Tier.FRONTIER, 0.015, 0.075))],
}

with Gateway(GatewayConfig(), HeuristicClassifier(), providers_by_tier) as gateway:
    result = gateway.chat("What is the capital of France?")
    print(result.tier, result.cost_usd, result.cache_hit)
    print(gateway.telemetry.summary())
```

## Install

```
pip install -e ".[dev]"      # + pytest, for running the test suite
pip install -e ".[openai]"   # to use the real OpenAI provider
pip install -e ".[anthropic]"  # to use the real Anthropic provider
```

## Project layout

```
llm_gateway/
  classifier/   complexity scoring (Classifier interface + HeuristicClassifier)
  cache/        semantic cache + embeddings
  router/       tier selection, fallback, escalation
  breaker/      per-provider circuit breaker
  providers/    provider interface + mock/OpenAI/Anthropic implementations
  gateway.py    ties the above together; the public entry point
  telemetry.py  request accounting / summary stats
  config.py     dataclasses for all tunables
examples/
  run_demo.py   resilience demo (fallback/escalation under a forced failure)
  benchmark.py  cost benchmark (gateway vs. frontier-only baseline)
tests/
```
