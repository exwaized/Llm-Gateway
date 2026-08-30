"""Exercises RedisSemanticCache against a minimal in-memory stub standing
in for upstash_redis.Redis, so these tests need no real Upstash instance
-- only that the cache's Redis command usage is well-formed and its
matching/eviction logic is correct.
"""

import time

import pytest

from llm_gateway.cache.redis_cache import RedisSemanticCache, client_from_env
from llm_gateway.config import CacheConfig
from fake_redis import FakeRedis


@pytest.fixture
def cache():
    return RedisSemanticCache(CacheConfig(similarity_threshold=0.9, ttl_seconds=3600, max_entries=5), FakeRedis())


def test_miss_then_hit_on_identical_prompt(cache):
    miss = cache.lookup("What is the capital of France?")
    assert not miss.hit

    cache.put("What is the capital of France?", "Paris", scope_key="default")

    hit = cache.lookup("What is the capital of France?")
    assert hit.hit
    assert hit.entry.response_text == "Paris"


def test_scopes_do_not_leak_into_each_other(cache):
    cache.put("hello", "hi there", scope_key="tenant-a")

    result = cache.lookup("hello", scope_key="tenant-b")
    assert not result.hit


def test_max_entries_evicts_oldest_first(cache):
    for i in range(6):
        cache.put(f"prompt number {i}", f"response {i}", scope_key="default")

    assert cache.size() == 5
    # the oldest entry (prompt number 0) should have been evicted
    result = cache.lookup("prompt number 0")
    assert not result.hit


def test_ttl_expires_entries():
    cache = RedisSemanticCache(CacheConfig(similarity_threshold=0.9, ttl_seconds=0.05, max_entries=100), FakeRedis())
    cache.put("hello there", "hi", scope_key="default")
    time.sleep(0.1)

    result = cache.lookup("hello there")
    assert not result.hit
    assert cache.size() == 0


def test_clear_removes_everything(cache):
    cache.put("a", "b", scope_key="s1")
    cache.put("c", "d", scope_key="s2")
    cache.clear()
    assert cache.size() == 0


def test_client_from_env_returns_none_without_config(monkeypatch):
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    monkeypatch.delenv("KV_REST_API_URL", raising=False)
    monkeypatch.delenv("KV_REST_API_TOKEN", raising=False)
    assert client_from_env() is None
