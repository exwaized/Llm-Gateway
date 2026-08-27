"""Cost/latency benchmark: gateway (tiering + semantic cache) vs. an
"always call the frontier model, no cache" baseline, on the same synthetic
workload.

Uses MockProvider (no network, no API keys) with per-1k-token pricing set
to roughly track real-world cheap/mid/frontier tiers as of 2026, so the
$ figures are representative even though the calls are simulated.

    python examples/benchmark.py
"""

from __future__ import annotations

import random
import statistics
import time

from llm_gateway import Gateway, GatewayConfig, ProviderConfig, Tier
from llm_gateway.classifier import HeuristicClassifier
from llm_gateway.providers.mock_provider import MockProvider

SEED = 7
N_REQUESTS = 300

# Roughly: cheap ~= a small fast model, mid ~= a general-purpose model,
# frontier ~= a top-of-line reasoning model. $ per 1k tokens.
CHEAP_PRICE = (0.00015, 0.0006)
MID_PRICE = (0.0025, 0.01)
FRONTIER_PRICE = (0.015, 0.075)

# Simulated workload: a support/assistant-style traffic mix. Simple lookups
# dominate real traffic and repeat often -- a handful of FAQs account for
# most of the volume, with a long tail of one-off variants (Zipf-sampled
# below, not uniform, so the cache hit rate reflects real traffic skew
# instead of an artificially small prompt pool).
_COUNTRIES = [
    "France", "Japan", "Germany", "Brazil", "Egypt", "Canada", "Kenya",
    "Norway", "Vietnam", "Peru", "Portugal", "Thailand", "Morocco", "Chile",
    "Poland", "Greece", "Iceland", "Nigeria", "Malaysia", "Finland",
    "Argentina", "Turkey", "Sweden", "Colombia", "Jordan", "Ukraine",
]
_TERMS = [
    "idempotent", "latency", "throughput", "eventual consistency", "checksum",
    "race condition", "backpressure", "cardinality", "quorum", "jitter",
    "sharding", "denormalization", "leader election", "vector clock",
    "tail latency", "cache stampede", "sticky session", "circuit breaker",
]
_CONVERSIONS = [
    ("10 miles", "kilometers"), ("5 kg", "pounds"), ("100 F", "Celsius"),
    ("2 liters", "gallons"), ("1 acre", "square meters"), ("60 mph", "km/h"),
    ("3 cups", "milliliters"), ("12 inches", "centimeters"), ("1 ton", "kg"),
    ("90 minutes", "hours"),
]
_WORDS = [
    "good morning", "thank you", "see you later", "how are you", "welcome",
    "good luck", "excuse me", "happy birthday", "see you soon", "good night",
]
_COMPANIES = [
    "Microsoft", "Toyota", "Nestle", "Samsung", "Siemens", "Shopify",
    "Spotify", "Airbnb", "Intel", "Sony", "Adobe", "Salesforce",
]

SIMPLE_PROMPTS = (
    [f"What is the capital of {c}?" for c in _COUNTRIES]
    + [f"Define '{t}'." for t in _TERMS]
    + [f"Convert {v} to {u}." for v, u in _CONVERSIONS]
    + [f"Translate '{w}' to Spanish." for w in _WORDS]
    + [f"Who is the CEO of {co}?" for co in _COMPANIES]
)

_TECH_PAIRS = [
    ("TCP", "UDP"), ("REST", "gRPC"), ("SQL", "NoSQL"), ("Docker", "a VM"),
    ("OAuth", "SAML"), ("Kafka", "RabbitMQ"), ("REST", "GraphQL"),
    ("JWTs", "server-side sessions"),
]

# Tuned so the heuristic classifier's score lands mid-tier (0.35-0.70): two
# reasoning keywords ("compare", "trade-off") plus enough length to avoid
# falling back into the cheap tier, but no code blocks.
MEDIUM_PROMPTS = [
    f"Compare the practical trade-offs between {a} and {b} for a mid-sized "
    f"production service handling a few thousand requests per second, and "
    f"note when the simpler option is still the right trade-off."
    for a, b in _TECH_PAIRS
] + [
    "Compare the trade-offs of remote work versus in-office work for a "
    "five-person startup, and explain which trade-off matters most early on.",
    "Compare a hash table's collision-handling trade-offs against a simple "
    "sorted array, and explain why one trade-off wins at small scale.",
]

# Tuned to clear the classifier's frontier threshold (score >= 0.70): 150+
# tokens, 2+ reasoning keywords, a real code block (3+ code hints), and a
# numbered list (multi-part hits) -- i.e. genuinely hard, multi-step asks,
# not just "long."
COMPLEX_PROMPTS = [
    "Compare the trade-offs between optimistic and pessimistic locking for a "
    "high-contention order-matching engine, and explain why one is preferred "
    "under heavy write contention. Then walk through the root cause of why "
    "naive optimistic locking degrades under contention, step by step: "
    "1) what happens to retry rates as contention rises, 2) how that affects "
    "tail latency, and 3) what a hybrid approach with pessimistic locking on "
    "hot rows would need to change. Also refactor this function so its "
    "locking strategy is explicit:\n"
    "```python\ndef place_order(book, order):\n    if book.locked:\n        "
    "return None\n    book.apply(order)\n    return book.last_trade\n```",
    "Debug why this recursive function stack-overflows on large inputs, and "
    "propose two fixes, weighing the trade-offs of each: raw recursion depth "
    "versus an explicit stack, and memoization versus iteration. Walk through "
    "the root cause step by step: 1) how Python's call stack limit is hit, "
    "2) why this specific recursion pattern is worse than tail recursion, "
    "and 3) how each fix changes both memory use and readability. Also "
    "rewrite it so it no longer recurses:\n"
    "```python\ndef fib(n):\n    if n < 2:\n        return n\n    "
    "return fib(n - 1) + fib(n - 2)\n```",
    "Design a rate limiter for a multi-tenant API and explain why you chose "
    "the token-bucket algorithm over sliding-window counters and fixed "
    "windows. Analyze the trade-offs step by step: 1) memory per tenant, "
    "2) burst tolerance, and 3) behavior under clock skew across replicas. "
    "Also explain why a naive shared in-memory counter breaks under "
    "horizontal scaling, and derive what state needs to move to a shared "
    "store like Redis:\n"
    "```python\nclass Bucket:\n    def __init__(self, rate):\n        "
    "self.rate = rate\n        self.tokens = rate\n```",
    "Walk through the root cause of a memory leak in a long-running Python "
    "service that uses threading.local() incorrectly, step by step: "
    "1) how thread pool reuse causes stale references to survive, "
    "2) why the garbage collector can't reclaim them, and 3) how this "
    "differs from a leak caused by an ever-growing module-level cache. "
    "Also compare two fixes and explain why one is more robust: clearing "
    "threading.local() explicitly per task versus switching to an explicit "
    "context object. Refactor this to avoid the leak:\n"
    "```python\nclass Worker:\n    def __init__(self):\n        "
    "self.local = threading.local()\n```",
    "Analyze why this cache-aside implementation is vulnerable to a cache "
    "stampede under a thundering herd of concurrent misses, and design a "
    "fix using request coalescing or probabilistic early expiration. "
    "Explain step by step: 1) what happens when a hot key expires under "
    "high concurrency, 2) why per-key locking alone doesn't fully solve it "
    "across multiple app instances, and 3) how the fix changes tail latency "
    "under load. Also refactor this function to add the fix:\n"
    "```python\ndef get(key):\n    value = cache.get(key)\n    if value is "
    "None:\n        value = db.query(key)\n        cache.set(key, value)\n    "
    "return value\n```",
]

# (pool, weight) -- weight is the share of total traffic drawn from that pool
WORKLOAD_MIX = [
    (SIMPLE_PROMPTS, 0.55),
    (MEDIUM_PROMPTS, 0.30),
    (COMPLEX_PROMPTS, 0.15),
]


def _zipf_weights(n: int, s: float = 0.8) -> list[float]:
    return [1 / (i + 1) ** s for i in range(n)]


def build_workload(rng: random.Random, n: int) -> list[str]:
    """Sample prompts Zipf-skewed within each pool -- a few hot FAQs make up
    most of a pool's volume, with a long tail asked once or twice, which is
    how real support/assistant traffic actually distributes."""
    pools, pool_weights = zip(*WORKLOAD_MIX)
    shuffled_pools = []
    for pool in pools:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        shuffled_pools.append((shuffled, _zipf_weights(len(shuffled))))

    prompts = []
    for _ in range(n):
        idx = rng.choices(range(len(shuffled_pools)), weights=pool_weights, k=1)[0]
        items, weights = shuffled_pools[idx]
        prompts.append(rng.choices(items, weights=weights, k=1)[0])
    return prompts


def build_gateway() -> Gateway:
    cheap = MockProvider(
        ProviderConfig("mock-cheap", Tier.CHEAP, *CHEAP_PRICE), base_latency_s=0.002
    )
    mid = MockProvider(ProviderConfig("mock-mid", Tier.MID, *MID_PRICE), base_latency_s=0.006)
    frontier = MockProvider(
        ProviderConfig("mock-frontier", Tier.FRONTIER, *FRONTIER_PRICE), base_latency_s=0.015
    )
    providers_by_tier = {Tier.CHEAP: [cheap], Tier.MID: [mid], Tier.FRONTIER: [frontier]}
    return Gateway(GatewayConfig(), HeuristicClassifier(), providers_by_tier)


def frontier_only_cost(prompts: list[str]) -> float:
    """What the same workload would cost with no tiering and no cache --
    every request sent straight to the frontier model."""
    in_price, out_price = FRONTIER_PRICE
    total = 0.0
    for prompt in prompts:
        tokens_in = max(1, len(prompt.split()))
        tokens_out = max(1, tokens_in // 2)
        total += tokens_in / 1000 * in_price + tokens_out / 1000 * out_price
    return total


def run() -> None:
    rng = random.Random(SEED)
    prompts = build_workload(rng, N_REQUESTS)

    with build_gateway() as gateway:
        latencies = []
        start = time.monotonic()
        for prompt in prompts:
            result = gateway.chat(prompt)
            latencies.append(result.latency_s)
        wall_s = time.monotonic() - start

        summary = gateway.telemetry.summary()

    baseline_cost = frontier_only_cost(prompts)
    gateway_cost = summary["total_cost_usd"]
    savings_usd = baseline_cost - gateway_cost
    savings_pct = (savings_usd / baseline_cost * 100) if baseline_cost else 0.0

    print("=" * 72)
    print(f"workload: {N_REQUESTS} requests (seed={SEED}), wall time {wall_s:.3f}s")
    print("=" * 72)
    print(f"{'':28}{'gateway':>18}{'frontier-only baseline':>26}")
    print(f"{'total cost':28}{'$' + format(gateway_cost, '.4f'):>18}{'$' + format(baseline_cost, '.4f'):>26}")
    print(f"{'avg cost / request':28}{'$' + format(gateway_cost / N_REQUESTS, '.6f'):>18}"
          f"{'$' + format(baseline_cost / N_REQUESTS, '.6f'):>26}")
    print("-" * 72)
    print(f"cost savings:            ${savings_usd:.4f}  ({savings_pct:.1f}% cheaper than frontier-only)")
    print(f"cache hit rate:           {summary['cache_hit_rate'] * 100:.1f}%")
    print(f"escalation rate:          {summary['escalation_rate'] * 100:.1f}%  (cheap/mid failed over to a higher tier)")
    print(f"tier distribution:        {summary['tier_distribution']}")
    print(f"latency avg / p95:        {statistics.mean(latencies) * 1000:.2f}ms / "
          f"{statistics.quantiles(latencies, n=20)[18] * 1000:.2f}ms")
    print("=" * 72)


if __name__ == "__main__":
    run()
