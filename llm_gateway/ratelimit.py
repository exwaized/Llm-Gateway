"""Fixed-window rate limiter, keyed by an arbitrary string (e.g. client IP).

Backed by Redis (INCR + EXPIRE on a per-window key) when a client is
supplied, so the limit is enforced across every serverless instance;
falls back to an in-process counter otherwise -- fine for local dev,
but under horizontal scaling each instance would then enforce its own
separate limit rather than a shared one.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, limit: int, window_s: int = 60, client=None):
        self._limit = limit
        self._window_s = window_s
        self._client = client
        self._local_lock = threading.Lock()
        self._local_counts: dict[str, int] = {}
        self._local_window_started: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        if self._client is not None:
            return self._allow_redis(key)
        return self._allow_local(key)

    def _allow_redis(self, key: str) -> bool:
        window = int(time.time() // self._window_s)
        redis_key = f"ratelimit:{key}:{window}"
        count = self._client.incr(redis_key)
        if count == 1:
            self._client.expire(redis_key, self._window_s)
        return count <= self._limit

    def _allow_local(self, key: str) -> bool:
        now = time.time()
        with self._local_lock:
            started = self._local_window_started.get(key)
            if started is None or now - started >= self._window_s:
                self._local_window_started[key] = now
                self._local_counts[key] = 0
            self._local_counts[key] += 1
            return self._local_counts[key] <= self._limit
