from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from llm_gateway.cache.embeddings import Embedder, HashingEmbedder, SparseVector, cosine_similarity
from llm_gateway.config import CacheConfig


@dataclass
class CacheEntry:
    prompt: str
    response_text: str
    embedding: SparseVector
    scope_key: str
    created_at: float = field(default_factory=time.monotonic)
    hits: int = 0


@dataclass
class CacheLookupResult:
    hit: bool
    entry: CacheEntry | None = None
    similarity: float = 0.0


class SemanticCache:
    """Embedding-similarity cache, scoped to avoid cross-tenant/cross-context leaks.

    `scope_key` must fold in whatever changes the meaning of a cache hit:
    tenant id, system prompt hash, tool-schema hash. Two identical prompts
    under different scopes are different cache entries on purpose.

    Reads and writes are serialized behind a single RLock. At demo/small
    scale (thousands of entries, linear scan) this is not a bottleneck;
    past that, swap the linear scan for an ANN index (FAISS/hnswlib) but
    keep the same lock discipline around mutation.
    """

    def __init__(self, config: CacheConfig, embedder: Embedder | None = None):
        self._config = config
        self._embedder = embedder or HashingEmbedder()
        self._entries: list[CacheEntry] = []
        self._lock = threading.RLock()

    def _evict_expired_locked(self) -> None:
        if self._config.ttl_seconds <= 0:
            return
        cutoff = time.monotonic() - self._config.ttl_seconds
        self._entries = [e for e in self._entries if e.created_at >= cutoff]

    def lookup(self, prompt: str, scope_key: str = "default") -> CacheLookupResult:
        query_vec = self._embedder.embed(prompt)

        with self._lock:
            self._evict_expired_locked()
            best_entry: CacheEntry | None = None
            best_sim = 0.0
            for entry in self._entries:
                if entry.scope_key != scope_key:
                    continue
                sim = cosine_similarity(query_vec, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None and best_sim >= self._config.similarity_threshold:
                best_entry.hits += 1
                return CacheLookupResult(hit=True, entry=best_entry, similarity=best_sim)

        return CacheLookupResult(hit=False, similarity=best_sim)

    def put(self, prompt: str, response_text: str, scope_key: str = "default") -> None:
        embedding = self._embedder.embed(prompt)
        entry = CacheEntry(
            prompt=prompt,
            response_text=response_text,
            embedding=embedding,
            scope_key=scope_key,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._config.max_entries:
                # drop oldest first (simple FIFO eviction)
                overflow = len(self._entries) - self._config.max_entries
                del self._entries[:overflow]

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
