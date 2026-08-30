"""Minimal in-memory stand-in for upstash_redis.Redis, covering just the
commands this codebase's Redis-backed components use (RedisSemanticCache,
RedisCircuitBreaker, RedisTelemetry, RateLimiter). Lets tests exercise
real command usage/logic without a live Upstash instance.
"""

from __future__ import annotations


class FakeRedis:
    def __init__(self):
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._sets: dict[str, set[str]] = {}
        self._lists: dict[str, list[str]] = {}

    # -- strings --
    def get(self, key):
        return self._strings.get(key)

    def set(self, key, value, **kwargs):
        self._strings[key] = str(value)

    def incr(self, key):
        value = int(self._strings.get(key, 0)) + 1
        self._strings[key] = str(value)
        return value

    def expire(self, key, seconds, **kwargs):
        return True

    # -- hashes --
    def hset(self, key, field, value):
        self._hashes.setdefault(key, {})[field] = value

    def hgetall(self, key):
        return dict(self._hashes.get(key, {}))

    def hdel(self, key, *fields):
        h = self._hashes.get(key, {})
        for f in fields:
            h.pop(f, None)

    def hlen(self, key):
        return len(self._hashes.get(key, {}))

    # -- sorted sets --
    def zadd(self, key, scores):
        self._zsets.setdefault(key, {}).update(scores)

    def zcard(self, key):
        return len(self._zsets.get(key, {}))

    def zrange(self, key, start, stop):
        ordered = sorted(self._zsets.get(key, {}).items(), key=lambda kv: kv[1])
        ids = [member for member, _ in ordered]
        stop_idx = len(ids) if stop < 0 else stop + 1
        return ids[start:stop_idx]

    def zrangebyscore(self, key, min_, max_):
        lo = float("-inf") if min_ == "-inf" else float(min_)
        hi = float("inf") if max_ == "+inf" else float(max_)
        ordered = sorted(self._zsets.get(key, {}).items(), key=lambda kv: kv[1])
        return [member for member, score in ordered if lo <= score <= hi]

    def zremrangebyscore(self, key, min_, max_):
        ids = self.zrangebyscore(key, min_, max_)
        for i in ids:
            self._zsets.get(key, {}).pop(i, None)

    def zremrangebyrank(self, key, start, stop):
        ordered = sorted(self._zsets.get(key, {}).items(), key=lambda kv: kv[1])
        ids = [member for member, _ in ordered]
        stop_idx = len(ids) if stop < 0 else stop + 1
        for i in ids[start:stop_idx]:
            self._zsets.get(key, {}).pop(i, None)

    # -- sets --
    def sadd(self, key, *members):
        self._sets.setdefault(key, set()).update(members)

    def smembers(self, key):
        return set(self._sets.get(key, set()))

    # -- lists --
    def lpush(self, key, *elements):
        self._lists.setdefault(key, [])
        for e in elements:
            self._lists[key].insert(0, e)

    def ltrim(self, key, start, stop):
        lst = self._lists.get(key, [])
        stop_idx = len(lst) if stop < 0 else stop + 1
        self._lists[key] = lst[start:stop_idx]

    def lrange(self, key, start, stop):
        lst = self._lists.get(key, [])
        stop_idx = len(lst) if stop < 0 else stop + 1
        return lst[start:stop_idx]

    # -- generic --
    def delete(self, *keys):
        for k in keys:
            self._strings.pop(k, None)
            self._hashes.pop(k, None)
            self._zsets.pop(k, None)
            self._sets.pop(k, None)
            self._lists.pop(k, None)
