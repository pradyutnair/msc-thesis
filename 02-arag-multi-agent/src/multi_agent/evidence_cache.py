"""Shared evidence cache for cross-agent document reuse."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np

from multi_agent.types import CachedDocument, CacheAnalytics

logger = logging.getLogger(__name__)


class EvidenceCache:
    """In-memory document cache shared across search agents for one question.

    Thread-safe via asyncio.Lock. Created fresh per question, discarded after.

    Operations:
        put       -- store a document; returns False on duplicate
        get_by_id -- retrieve by doc_id
        get_relevant -- cosine similarity over cached embeddings
        get_all_evidence -- all docs sorted by score
        compute_analytics -- hit rate, cross-agent reuse, unique docs
    """

    def __init__(self, enabled: bool = True):
        self._store: dict[str, CachedDocument] = {}
        self._lock = asyncio.Lock()
        self._enabled = enabled

        # Counters
        self._total_puts = 0
        self._duplicate_hits = 0
        self._cross_agent_reuses = 0
        self._total_gets = 0
        self._get_hits = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def put(self, doc: CachedDocument) -> bool:
        """Store a document. Returns True if new, False if duplicate."""
        if not self._enabled:
            return True

        async with self._lock:
            self._total_puts += 1
            if doc.doc_id in self._store:
                existing = self._store[doc.doc_id]
                self._duplicate_hits += 1
                if existing.source_agent != doc.source_agent:
                    self._cross_agent_reuses += 1
                    logger.debug(
                        "Cross-agent reuse: doc %s (agent %d -> agent %d)",
                        doc.doc_id,
                        existing.source_agent,
                        doc.source_agent,
                    )
                # Keep higher-scoring version
                if doc.retrieval_score > existing.retrieval_score:
                    self._store[doc.doc_id] = doc
                return False

            self._store[doc.doc_id] = doc
            return True

    async def get_by_id(self, doc_id: str) -> CachedDocument | None:
        """Get a document by ID."""
        async with self._lock:
            self._total_gets += 1
            doc = self._store.get(doc_id)
            if doc is not None:
                self._get_hits += 1
            return doc

    async def get_relevant(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[CachedDocument]:
        """Find relevant cached docs via cosine similarity over embeddings.

        Only considers docs with non-None embeddings.
        """
        if not self._enabled:
            return []

        async with self._lock:
            self._total_gets += 1
            docs_with_emb = [
                d for d in self._store.values() if d.embedding is not None
            ]
            if not docs_with_emb:
                return []

            embeddings = np.stack([d.embedding for d in docs_with_emb])
            # Normalize for cosine similarity
            query_norm = query_embedding / (
                np.linalg.norm(query_embedding) + 1e-10
            )
            emb_norms = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
            )
            scores = emb_norms @ query_norm

            top_indices = np.argsort(scores)[::-1][:top_k]
            results = [docs_with_emb[i] for i in top_indices if scores[i] > 0.3]

            if results:
                self._get_hits += 1

            return results

    async def get_all_evidence(self) -> list[CachedDocument]:
        """All cached docs sorted by retrieval score (descending)."""
        async with self._lock:
            return sorted(
                self._store.values(),
                key=lambda d: d.retrieval_score,
                reverse=True,
            )

    async def size(self) -> int:
        async with self._lock:
            return len(self._store)

    async def compute_analytics(self) -> CacheAnalytics:
        """Compute cache usage statistics."""
        async with self._lock:
            return CacheAnalytics(
                total_puts=self._total_puts,
                duplicate_hits=self._duplicate_hits,
                cross_agent_reuses=self._cross_agent_reuses,
                unique_docs=len(self._store),
                total_gets=self._total_gets,
                get_hit_rate=(
                    self._get_hits / self._total_gets
                    if self._total_gets > 0
                    else 0.0
                ),
            )

    def compute_analytics_sync(self) -> dict[str, Any]:
        """Synchronous version for serialization."""
        return {
            "total_puts": self._total_puts,
            "duplicate_hits": self._duplicate_hits,
            "cross_agent_reuses": self._cross_agent_reuses,
            "unique_docs": len(self._store),
            "total_gets": self._total_gets,
            "get_hit_rate": (
                self._get_hits / self._total_gets
                if self._total_gets > 0
                else 0.0
            ),
        }
