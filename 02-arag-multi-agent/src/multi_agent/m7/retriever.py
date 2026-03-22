"""M7 Retriever: E5 sentence-level retrieval with neighborhood expansion
and evidence pool search.

The evidence pool search is the core CORAL innovation: before doing any
corpus retrieval, workers search previously retrieved documents using their
own query.  This is retrieval-within-retrieval at zero new retrieval cost.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class M7Retriever:
    """E5 sentence-level retriever with neighborhood expansion and
    evidence pool search."""

    def __init__(
        self,
        index_dir: str | Path,
        embed_model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        neighborhood: int = 2,
    ):
        index_dir = Path(index_dir)
        index_file = index_dir / "sentence_index.pkl"
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")

        with open(index_file, "rb") as f:
            index_data = pickle.load(f)

        self.sentences: list[str] = index_data["sentences"]
        self.embeddings: np.ndarray = index_data["embeddings"]
        self.sentence_to_chunk: list = index_data["sentence_to_chunk"]
        self.chunks: dict = index_data["chunks"]
        self.neighborhood = neighborhood

        # Build sorted list of chunk IDs for neighborhood lookups
        self._chunk_ids_sorted: list = sorted(self.chunks.keys())
        self._chunk_id_to_pos: dict = {
            cid: i for i, cid in enumerate(self._chunk_ids_sorted)
        }

        self.model = SentenceTransformer(embed_model_name, device=device)
        self._lock = Lock()

        # Cache for document embeddings (doc_id -> embedding vector).
        # Populated on first retrieval so evidence pool search can reuse them.
        self._doc_embedding_cache: dict[str, np.ndarray] = {}
        self._cache_lock = Lock()

        logger.info(
            "M7Retriever loaded: %d sentences, %d chunks, neighborhood=%d",
            len(self.sentences), len(self.chunks), self.neighborhood,
        )

    # ------------------------------------------------------------------
    # Evidence Pool Search (CORAL core innovation)
    # ------------------------------------------------------------------

    def search_evidence_pool(
        self,
        query: str,
        evidence_pool: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search previously retrieved documents using a new query.

        Uses the same E5 embedding model to compute similarity between
        the query and documents in the evidence pool.
        Returns top_k most relevant documents from the pool.
        """
        if not evidence_pool:
            return []

        # Encode the query
        with self._lock:
            query_embedding = self.model.encode(
                [f"query: {query}"], normalize_embeddings=True,
            )[0]

        # Get or compute embeddings for each pool document
        doc_embeddings = []
        valid_docs = []
        for doc in evidence_pool:
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            if not content:
                continue

            emb = self._get_or_compute_doc_embedding(doc_id, content)
            doc_embeddings.append(emb)
            valid_docs.append(doc)

        if not doc_embeddings:
            return []

        # Compute similarities via dot product (embeddings are normalized)
        emb_matrix = np.stack(doc_embeddings, axis=0)
        similarities = np.dot(emb_matrix, query_embedding)

        # Rank and return top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            doc = dict(valid_docs[idx])  # shallow copy
            doc["pool_score"] = float(similarities[idx])
            results.append(doc)

        return results

    def _get_or_compute_doc_embedding(
        self, doc_id: str, content: str,
    ) -> np.ndarray:
        """Get cached document embedding or compute and cache it."""
        with self._cache_lock:
            if doc_id in self._doc_embedding_cache:
                return self._doc_embedding_cache[doc_id]

        # Compute outside the cache lock to avoid blocking other threads
        with self._lock:
            embedding = self.model.encode(
                [f"passage: {content}"], normalize_embeddings=True,
            )[0]

        with self._cache_lock:
            self._doc_embedding_cache[doc_id] = embedding

        return embedding

    # ------------------------------------------------------------------
    # Corpus Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand_neighborhood: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k chunks by E5 sentence similarity, then expand
        neighborhoods."""
        with self._lock:
            query_embedding = self.model.encode(
                [f"query: {query}"], normalize_embeddings=True,
            )[0]

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][: top_k * 3]

        # Aggregate sentence scores to chunk level
        chunk_scores: dict[Any, float] = {}
        for idx in top_indices:
            chunk_id = self.sentence_to_chunk[idx]
            score = float(similarities[idx])
            chunk_scores[chunk_id] = max(score, chunk_scores.get(chunk_id, -1.0))

        # Take top-k chunks
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        core_chunk_ids = [cid for cid, _ in ranked]

        # Neighborhood expansion: add +-N adjacent chunks
        if expand_neighborhood and self.neighborhood > 0:
            expanded_ids = set(core_chunk_ids)
            for cid in core_chunk_ids:
                pos = self._chunk_id_to_pos.get(cid)
                if pos is None:
                    continue
                for offset in range(-self.neighborhood, self.neighborhood + 1):
                    neighbor_pos = pos + offset
                    if 0 <= neighbor_pos < len(self._chunk_ids_sorted):
                        expanded_ids.add(self._chunk_ids_sorted[neighbor_pos])
            # Order: core first (by score), then neighbors (by chunk ID)
            neighbor_only = sorted(expanded_ids - set(core_chunk_ids))
            all_ids = core_chunk_ids + neighbor_only
        else:
            all_ids = core_chunk_ids

        docs = []
        for chunk_id in all_ids:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            doc = {
                "doc_id": str(chunk.get("id", chunk_id)),
                "title": chunk.get("title", ""),
                "content": chunk.get("text", chunk.get("content", "")),
                "score": chunk_scores.get(chunk_id, 0.0),
                "is_neighbor": chunk_id not in set(core_chunk_ids),
            }
            docs.append(doc)
            # Cache the document embedding for evidence pool search
            self._get_or_compute_doc_embedding(doc["doc_id"], doc["content"])

        return docs

    def multi_query_retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
        expand_neighborhood: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve from multiple queries and merge results (union, max score)."""
        merged: dict[str, dict[str, Any]] = {}
        for query in queries:
            docs = self.retrieve(query, top_k=top_k, expand_neighborhood=expand_neighborhood)
            for doc in docs:
                did = doc["doc_id"]
                if did not in merged or doc["score"] > merged[did]["score"]:
                    merged[did] = doc
        # Sort by score descending
        return sorted(merged.values(), key=lambda d: d["score"], reverse=True)
