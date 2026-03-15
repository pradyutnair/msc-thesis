"""Semantic search tool - FAISS + E5 for FlashRAG 21M corpus.

Supports two backends:
  1. FAISS index (default): Pre-built FlashRAG E5-base-v2 index over 21M passages.
     Queries encoded with E5-base-v2, searched via FAISS.
  2. Legacy in-memory: Per-dataset sentence_index.pkl files.
"""

import os
import sqlite3
import threading
import numpy as np
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class SemanticSearchTool(BaseTool):
    """Semantic search using embedding similarity."""

    _embedding_lock = threading.Lock()

    def __init__(
        self,
        chunks_file: str | None = None,
        index_dir: str | None = None,
        model_name: str = "intfloat/e5-base-v2",
        device: str = None,
        faiss_index_path: str | None = None,
        sqlite_db: str | None = None,
    ):
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required")
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers required")

        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.model_name = model_name
        self.device = device

        # Determine backend
        self._use_faiss_direct = False
        self._faiss_index_path = faiss_index_path or os.getenv("FLASHRAG_FAISS_INDEX")
        self._sqlite_db = sqlite_db or os.getenv("FLASHRAG_SQLITE_DB")

        if self._faiss_index_path and os.path.exists(self._faiss_index_path):
            self._init_faiss_backend()
        elif index_dir and os.path.exists(os.path.join(index_dir, "sentence_index.pkl")):
            self._init_legacy_backend(chunks_file, index_dir)
        else:
            raise FileNotFoundError(
                f"No index found. Tried FAISS at {self._faiss_index_path} "
                f"and legacy at {index_dir}/sentence_index.pkl"
            )

        # Query prompt handling for E5 models
        raw_use_prompt = os.getenv("ARAG_USE_QUERY_PROMPT", "1").strip().lower()
        self.use_query_prompt = raw_use_prompt not in {"0", "false", "no", "off"}
        self.query_prompt_name = os.getenv("ARAG_QUERY_PROMPT_NAME", "query")

    def _init_faiss_backend(self):
        """Initialize FAISS direct search over FlashRAG corpus."""
        import logging
        logger = logging.getLogger(__name__)

        if not HAS_FAISS:
            raise ImportError("faiss required for FlashRAG backend. Install: pip install faiss-cpu")

        # Find the actual index file
        index_path = self._faiss_index_path
        if os.path.isdir(index_path):
            # Look for index file inside directory
            candidates = []
            for name in os.listdir(index_path):
                if name.endswith((".index", ".faiss")) or name == "index":
                    candidates.append(os.path.join(index_path, name))
            if not candidates:
                # Try the directory itself as containing a single file
                all_files = [
                    os.path.join(index_path, f)
                    for f in os.listdir(index_path)
                    if os.path.isfile(os.path.join(index_path, f))
                ]
                candidates = all_files
            if candidates:
                index_path = candidates[0]
            else:
                raise FileNotFoundError(f"No index file found in {self._faiss_index_path}")

        logger.info("Loading FAISS index from %s ...", index_path)
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(
            "FAISS index: %d vectors, dim=%d",
            self.faiss_index.ntotal, self.faiss_index.d,
        )

        # Load embedding model for query encoding
        # E5 models need "query: " prefix for queries
        logger.info("Loading embedding model: %s", self.model_name)
        self.embedding_model = SentenceTransformer(self.model_name, device=self.device)

        self._use_faiss_direct = True

        # We need SQLite to map FAISS indices back to passage text
        if not self._sqlite_db or not os.path.exists(self._sqlite_db):
            logger.warning(
                "SQLite DB not found at %s. ReadChunk will be needed for full text.",
                self._sqlite_db,
            )

    def _init_legacy_backend(self, chunks_file, index_dir):
        """Initialize legacy sentence-level in-memory search."""
        import pickle
        import logging

        logger = logging.getLogger(__name__)
        logger.info("Loading legacy sentence index from %s", index_dir)

        self.embedding_model = SentenceTransformer(self.model_name, device=self.device)

        index_file = os.path.join(index_dir, "sentence_index.pkl")
        with open(index_file, "rb") as f:
            index_data = pickle.load(f)

        self.sentences = index_data["sentences"]
        self.embeddings = index_data["embeddings"]
        self.sentence_to_chunk = index_data["sentence_to_chunk"]
        self.chunks = index_data["chunks"]

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": """Semantic search using embedding similarity. Matches your query against passages via vector similarity.

WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching

RETURNS: Abbreviated snippets with matched passages. Use read_chunk to get full text for answering.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what information you're looking for",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of most relevant results to return (default: 5, max: 20)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        query: str,
        top_k: int = 5,
    ) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        if self._use_faiss_direct:
            return self._execute_faiss(context, query, top_k)
        else:
            return self._execute_legacy(context, query, top_k)

    # ── FAISS direct backend ─────────────────────────────────────────

    def _execute_faiss(
        self,
        context: "AgentContext",
        query: str,
        top_k: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Search using pre-built FAISS index over 21M passages."""
        # Encode query with E5 prefix
        # E5 models expect "query: " prefix for queries
        query_text = f"query: {query}"

        with self._embedding_lock:
            query_embedding = self.embedding_model.encode(
                [query_text],
                normalize_embeddings=True,
            )[0]

        # Search FAISS
        query_vec = np.array([query_embedding], dtype=np.float32)
        scores, indices = self.faiss_index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]

        # Filter out invalid indices
        valid = [(float(s), int(i)) for s, i in zip(scores, indices) if i >= 0]

        if not valid:
            return f"No results for: {query}", {"retrieved_tokens": 0, "chunks_found": 0}

        # Look up passage text from SQLite
        result_parts = []
        all_snippets = []

        if self._sqlite_db and os.path.exists(self._sqlite_db):
            conn = sqlite3.connect(self._sqlite_db)
            for score, idx in valid:
                # FAISS index position maps to SQLite rowid (1-indexed)
                row = conn.execute(
                    "SELECT passage_id, title, contents FROM passages WHERE id = ?",
                    (idx + 1,),
                ).fetchone()
                if row:
                    passage_id, title, contents = row
                    # Show first ~200 chars as snippet
                    snippet = contents[:300].replace("\n", " ")
                    if len(contents) > 300:
                        snippet += "..."
                    result_parts.append(
                        f"Chunk ID: {passage_id} (Similarity: {score:.3f})\n"
                        f"Title: {title}\nSnippet: {snippet}"
                    )
                    all_snippets.append(contents[:300])
                else:
                    result_parts.append(
                        f"Chunk ID: {idx} (Similarity: {score:.3f})\n"
                        f"(passage text not found in DB)"
                    )
            conn.close()
        else:
            # No SQLite DB - just return indices
            for score, idx in valid:
                result_parts.append(
                    f"Chunk ID: {idx} (Similarity: {score:.3f})\n"
                    f"(use read_chunk to get full text)"
                )

        tool_result = "\n\n".join(result_parts)

        retrieved_tokens = (
            len(self.tokenizer.encode("\n".join(all_snippets)))
            if all_snippets
            else 0
        )

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={
                "query": query,
                "chunks_found": len(valid),
                "backend": "faiss_direct",
            },
        )

        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(valid)}

    # ── Legacy in-memory backend ─────────────────────────────────────

    def _execute_legacy(
        self,
        context: "AgentContext",
        query: str,
        top_k: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Original sentence-level in-memory search for small corpora."""
        with self._embedding_lock:
            if self.use_query_prompt:
                try:
                    query_embedding = self.embedding_model.encode(
                        [query],
                        prompt_name=self.query_prompt_name,
                        normalize_embeddings=True,
                    )[0]
                except TypeError:
                    query_embedding = self.embedding_model.encode(
                        [query], normalize_embeddings=True,
                    )[0]
            else:
                query_embedding = self.embedding_model.encode(
                    [query], normalize_embeddings=True,
                )[0]

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][: top_k * 3]

        chunk_sentences = {}
        for idx in top_indices:
            sentence = self.sentences[idx]
            chunk_id = self.sentence_to_chunk[idx]
            similarity = float(similarities[idx])

            if chunk_id not in chunk_sentences:
                chunk_sentences[chunk_id] = []
            chunk_sentences[chunk_id].append({
                "sentence": sentence,
                "similarity": similarity,
                "position": idx,
            })

        chunk_scores = []
        for chunk_id, sents in chunk_sentences.items():
            max_similarity = max(s["similarity"] for s in sents)
            chunk_scores.append((chunk_id, max_similarity, sents))

        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k]

        if not top_chunks:
            return f"No results for: {query}", {"retrieved_tokens": 0, "chunks_found": 0}

        result_parts = []
        for chunk_id, max_sim, sents in top_chunks:
            chunk_text = self.chunks[chunk_id]["text"]
            sents_sorted = sorted(sents, key=lambda x: chunk_text.find(x["sentence"]))
            matched_text = "... " + " ... ".join([s["sentence"] for s in sents_sorted]) + " ..."
            result_parts.append(
                f"Chunk ID: {chunk_id} (Similarity: {max_sim:.3f})\nMatched: {matched_text}"
            )

        tool_result = "\n\n".join(result_parts)

        all_matched = []
        for _, _, sents in top_chunks:
            all_matched.extend([s["sentence"] for s in sents])

        retrieved_tokens = (
            len(self.tokenizer.encode("\n".join(all_matched)))
            if all_matched
            else 0
        )

        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={
                "query": query,
                "chunks_found": len(top_chunks),
                "use_query_prompt": self.use_query_prompt,
                "query_prompt_name": self.query_prompt_name,
                "backend": "legacy",
            },
        )

        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(top_chunks)}
