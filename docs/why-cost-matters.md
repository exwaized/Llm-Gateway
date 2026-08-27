# Why cost-aware routing still matters

A common objection to this project: model prices keep falling, context
windows keep growing toward the million-token range, and some vendors offer
large free tiers — so why build infrastructure to control cost?

**Bigger context windows increase spend, they don't reduce it.** Cost scales
with tokens processed. A 1M-token window doesn't get cheaper per request —
it's an invitation to stuff more into every request, so aggregate spend
rises even as capability rises. This is the same dynamic as Jevons paradox:
cheaper/bigger inference increases usage faster than the unit price drops,
so the total bill grows, not shrinks.

**"Unlimited free tier" is a rate-limited on-ramp for individual/dev usage,
not a production SLA.** No vendor eats inference cost at real production
volume indefinitely — free tiers get throttled, capped, or revoked the
moment usage looks commercial. A system that assumes the cheap/free path
stays available, with a defined fallback when it doesn't, is exactly what
this gateway's circuit breaker and multi-provider router are for.

**The frontier keeps moving.** Even if today's frontier model gets cheap
next year, there's always a new, more expensive frontier model people reach
for by default. Complexity-based tiering isn't a bet on today's price
list — it's a standing policy that routes "define idempotent" to a cheap
tier and reserves the expensive model for problems that actually need it,
regardless of what those two price points are at any given moment.

**Cost-awareness buys more than cost.** The same routing/caching/circuit-
breaker layer also gives you:

- **Latency** — a cache hit is ~instant; no model call happens at all.
- **Resilience** — a provider outage triggers automatic fallback to a
  healthy tier instead of a hard failure for the caller.
- **Vendor-lock-in avoidance** — providers are swappable behind one
  interface (`providers/base.py`), so a pricing or capability change from
  one vendor doesn't require rewriting the application.

None of those go away even in a world where token price trends toward zero.
Nobody stopped doing query optimization when storage got cheap; this is the
same argument applied to inference.
