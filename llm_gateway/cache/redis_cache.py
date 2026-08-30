"""Upstash Redis-backed semantic cache.

Same lookup()/put()/size()/clear() surface as SemanticCache, so Gateway
can use either interchangeably -- this one just persists entries in
Upstash Redis (REST, no long-lived TCP connection) so cache hits survive
across serverless cold starts and are shared across concurrent instances,
which an in-process cache cannot do.

Layout per scope_key:
  cache:{scope}:entries  -- hash: entry_id -> json(prompt, response_text,
                             embedding, created_at, hits)
  cache:{scope}:order    -- zset: entry_id, score=created_at; lets TTL
                             eviction and FIFO overflow both be range
                             queries instead of a full scan.
  cache:scopes           -- set of every scope_key ever written, so
                             size()/clear() can enumerate scopes without
                             a Redis KEYS scan on every call.

created_at uses wall-clock time.time(), not time.monotonic() -- entries
are compared across processes/instances, where a monotonic clock has no
shared epoch.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from llm_gateway.cache.embeddings import Embedder, HashingEmbedder, cosine_similarity
from llm_gateway.cache.semantic_cache import CacheEntry, CacheLookupResult
from llm_gateway.config import CacheConfig

_SCOPES_KEY = "cache:scopes"


def client_from_env():
    """Builds an upstash_redis.Redis client from env vars, or None if unset.

    Checks Upstash's own naming first, then Vercel's KV integration
    naming (Vercel KV is Upstash-backed and exposes the same REST
    protocol under different env var names).
    """
    try:
        from upstash_redis import Redis
    except ImportError:
        return None

    url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
    token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
    if not url or not token:
        return None
    return Redis(url=url, token=token)


class RedisSemanticCache:
    def __init__(self, config: CacheConfig, client, embedder: Embedder | None = None):
        self._config = config
        self._client = client
        self._embedder = embedder or HashingEmbedder()

    def _entries_key(self, scope_key: str) -> str:
        return f"cache:{scope_key}:entries"

    def _order_key(self, scope_key: str) -> str:
        return f"cache:{scope_key}:order"

    def _evict_expired(self, scope_key: str) -> None:
        if self._config.ttl_seconds <= 0:
            return
        cutoff = time.time() - self._config.ttl_seconds
        expired_ids = self._client.zrangebyscore(self._order_key(scope_key), "-inf", cutoff)
        if expired_ids:
            self._client.hdel(self._entries_key(scope_key), *expired_ids)
        self._client.zremrangebyscore(self._order_key(scope_key), "-inf", cutoff)

    def _evict_overflow(self, scope_key: str) -> None:
        order_key = self._order_key(scope_key)
        count = self._client.zcard(order_key)
        if count <= self._config.max_entries:
            return
        overflow = count - self._config.max_entries
        oldest_ids = self._client.zrange(order_key, 0, overflow - 1)
        if oldest_ids:
            self._client.hdel(self._entries_key(scope_key), *oldest_ids)
        self._client.zremrangebyrank(order_key, 0, overflow - 1)

    def lookup(self, prompt: str, scope_key: str = "default") -> CacheLookupResult:
        query_vec = self._embedder.embed(prompt)

        self._evict_expired(scope_key)
        raw_entries = self._client.hgetall(self._entries_key(scope_key)) or {}

        best_id: str | None = None
        best_entry: CacheEntry | None = None
        best_sim = 0.0
        for entry_id, raw in raw_entries.items():
            data = json.loads(raw)
            sim = cosine_similarity(query_vec, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = entry_id
                best_entry = CacheEntry(
                    prompt=data["prompt"],
                    response_text=data["response_text"],
                    embedding=data["embedding"],
                    scope_key=scope_key,
                    created_at=data["created_at"],
                    hits=data["hits"],
                )

        if best_entry is not None and best_sim >= self._config.similarity_threshold:
            best_entry.hits += 1
            self._client.hset(
                self._entries_key(scope_key),
                best_id,
                json.dumps(
                    {
                        "prompt": best_entry.prompt,
                        "response_text": best_entry.response_text,
                        "embedding": best_entry.embedding,
                        "created_at": best_entry.created_at,
                        "hits": best_entry.hits,
                    }
                ),
            )
            return CacheLookupResult(hit=True, entry=best_entry, similarity=best_sim)

        return CacheLookupResult(hit=False, similarity=best_sim)

    def put(self, prompt: str, response_text: str, scope_key: str = "default") -> None:
        embedding = self._embedder.embed(prompt)
        entry_id = uuid.uuid4().hex
        created_at = time.time()

        self._client.sadd(_SCOPES_KEY, scope_key)
        self._client.hset(
            self._entries_key(scope_key),
            entry_id,
            json.dumps(
                {
                    "prompt": prompt,
                    "response_text": response_text,
                    "embedding": embedding,
                    "created_at": created_at,
                    "hits": 0,
                }
            ),
        )
        self._client.zadd(self._order_key(scope_key), {entry_id: created_at})
        self._evict_overflow(scope_key)

    def size(self) -> int:
        scopes = self._client.smembers(_SCOPES_KEY) or set()
        return sum(self._client.hlen(self._entries_key(scope)) for scope in scopes)

    def clear(self) -> None:
        scopes = self._client.smembers(_SCOPES_KEY) or set()
        for scope in scopes:
            self._client.delete(self._entries_key(scope), self._order_key(scope))
        if scopes:
            self._client.delete(_SCOPES_KEY)
