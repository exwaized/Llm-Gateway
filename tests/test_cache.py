import time

from llm_gateway.cache.semantic_cache import SemanticCache
from llm_gateway.config import CacheConfig


def test_exact_duplicate_is_a_hit():
    cache = SemanticCache(CacheConfig(similarity_threshold=0.9))
    cache.put("What is the capital of France?", "Paris")

    result = cache.lookup("What is the capital of France?")

    assert result.hit
    assert result.entry.response_text == "Paris"


def test_unrelated_prompt_is_a_miss():
    cache = SemanticCache(CacheConfig(similarity_threshold=0.9))
    cache.put("What is the capital of France?", "Paris")

    result = cache.lookup("Write a sorting algorithm in Rust")

    assert not result.hit


def test_scope_key_isolates_entries():
    cache = SemanticCache(CacheConfig(similarity_threshold=0.9))
    cache.put("cancel my subscription", "done for tenant A", scope_key="tenant-a")

    result = cache.lookup("cancel my subscription", scope_key="tenant-b")

    assert not result.hit


def test_ttl_expires_entries():
    cache = SemanticCache(CacheConfig(similarity_threshold=0.9, ttl_seconds=0.05))
    cache.put("What is the capital of France?", "Paris")

    time.sleep(0.1)
    result = cache.lookup("What is the capital of France?")

    assert not result.hit


def test_max_entries_evicts_oldest():
    cache = SemanticCache(CacheConfig(similarity_threshold=0.9, max_entries=2))
    cache.put("prompt one", "resp1")
    cache.put("prompt two", "resp2")
    cache.put("prompt three", "resp3")

    assert cache.size() == 2
    assert not cache.lookup("prompt one").hit
    assert cache.lookup("prompt three").hit
