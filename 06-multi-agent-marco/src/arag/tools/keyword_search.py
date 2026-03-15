"""Keyword search tool - SQLite FTS5 for FlashRAG 21M corpus.

Supports two backends:
  1. SQLite FTS5 (default): For the FlashRAG 21M-passage corpus.
     Requires a pre-built SQLite DB from scripts/setup_flashrag.py.
  2. Legacy in-memory: For small per-dataset chunks.json files.
"""

import json
import os
import re
import sqlite3
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class KeywordSearchTool(BaseTool):
    """Keyword search using FTS5 or in-memory matching."""

    def __init__(
        self,
        chunks_file: str | None = None,
        sqlite_db: str | None = None,
    ):
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Install: pip install tiktoken")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

        # Determine backend
        self._use_sqlite = False
        self._db_path = sqlite_db or os.getenv("FLASHRAG_SQLITE_DB")

        if self._db_path and os.path.exists(self._db_path):
            self._use_sqlite = True
            # Test connection
            conn = sqlite3.connect(self._db_path)
            count = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            conn.close()
            import logging
            logging.getLogger(__name__).info(
                "KeywordSearch: SQLite FTS5 backend with %d passages", count,
            )
        elif chunks_file:
            self.chunks = self._load_chunks(chunks_file)
            import logging
            logging.getLogger(__name__).info(
                "KeywordSearch: in-memory backend with %d chunks", len(self.chunks),
            )
        else:
            raise ValueError(
                "Either chunks_file or sqlite_db (or FLASHRAG_SQLITE_DB env) required"
            )

    def _load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            return data
        chunks = []
        for item in data:
            if isinstance(item, str):
                parts = item.split(":", 1)
                if len(parts) == 2:
                    chunks.append({"id": parts[0], "text": parts[1]})
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?\n]+", text)
        return [s.strip() for s in sentences if s.strip()]

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": """Search for document chunks using keyword-based exact text matching (case-insensitive). Returns chunk IDs and abbreviated sentence snippets where the keywords appear.

IMPORTANT: This tool matches keywords literally in the text. Use SHORT, SPECIFIC terms (1-3 words maximum). Each keyword is matched independently.

Examples of GOOD keywords:
  - Entity names: "Albert Einstein", "Tesla", "Python", "Argentina"
  - Technical terms: "photosynthesis", "quantum mechanics"
  - Key concepts: "climate change", "GDP growth"

Examples of BAD keywords (DO NOT use):
  - Long phrases: "the person who invented the telephone" → use "Alexander Bell" instead
  - Questions: "when did World War 2 start" → use "World War 2", "1939" instead

RETURNS: Abbreviated snippets marked with "..." showing where keywords appear. These snippets help you identify relevant chunks, but you MUST use read_chunk to get the full text for answering questions.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keywords to search. Each keyword should be 1-3 words maximum.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top-ranked chunks to return (default: 5, max: 20)",
                            "default": 5,
                        },
                    },
                    "required": ["keywords"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        keywords: List[str],
        top_k: int = 5,
    ) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)

        if self._use_sqlite:
            return self._execute_sqlite(context, keywords, top_k)
        else:
            return self._execute_legacy(context, keywords, top_k)

    # ── SQLite FTS5 backend ──────────────────────────────────────────

    def _execute_sqlite(
        self,
        context: "AgentContext",
        keywords: List[str],
        top_k: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Search using SQLite FTS5 full-text search."""
        conn = sqlite3.connect(self._db_path)

        # Build FTS5 query: each keyword as a phrase, combined with OR
        # FTS5 will rank by BM25 automatically
        fts_terms = []
        for kw in keywords:
            # Escape FTS5 special chars and wrap in quotes for phrase match
            escaped = kw.replace('"', '""')
            fts_terms.append(f'"{escaped}"')

        fts_query = " OR ".join(fts_terms)

        try:
            # FTS5 with BM25 ranking
            rows = conn.execute(
                """
                SELECT p.passage_id, p.title, p.contents,
                       rank
                FROM passages_fts
                JOIN passages p ON p.id = passages_fts.rowid
                WHERE passages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            # Fallback: try simpler LIKE query if FTS5 fails
            conn.close()
            return f"Search error: {e}", {"retrieved_tokens": 0, "chunks_found": 0}

        conn.close()

        if not rows:
            tool_result = f"No results found for keywords: {keywords}"
            tool_log = {"retrieved_tokens": 0, "chunks_found": 0}
            return tool_result, tool_log

        result_parts = []
        all_snippets = []

        for passage_id, title, contents, rank in rows:
            # Extract matching sentences for snippet
            sentences = self._split_sentences(contents)
            matched_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(kw.lower() in sentence_lower for kw in keywords):
                    matched_sentences.append(sentence)
                    if len(matched_sentences) >= 3:
                        break

            if matched_sentences:
                snippet = "... " + " ... ".join(matched_sentences) + " ..."
            else:
                # Show first sentence as fallback
                snippet = (sentences[0][:200] + "...") if sentences else "(no text)"

            # Use passage_id as chunk_id for compatibility
            result_parts.append(
                f"Chunk ID: {passage_id}, Title: {title}\n"
                f"Matched keywords in chunk: {snippet}"
            )
            all_snippets.extend(matched_sentences)

        tool_result = "\n\n".join(result_parts)

        retrieved_tokens = (
            len(self.tokenizer.encode("\n".join(all_snippets)))
            if all_snippets
            else 0
        )

        context.add_retrieval_log(
            tool_name="keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "chunks_found": len(rows),
                "chunk_ids": [r[0] for r in rows],
                "backend": "sqlite_fts5",
            },
        )

        tool_log = {"retrieved_tokens": retrieved_tokens, "chunks_found": len(rows)}
        return tool_result, tool_log

    # ── Legacy in-memory backend ─────────────────────────────────────

    def _execute_legacy(
        self,
        context: "AgentContext",
        keywords: List[str],
        top_k: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Original in-memory keyword matching for small corpora."""
        scored_chunks = []
        for chunk in self.chunks:
            text = chunk["text"]
            text_lower = text.lower()
            chunk_id = chunk["id"]

            matches = []
            total_score = 0

            for keyword in keywords:
                keyword_lower = keyword.lower()
                count = text_lower.count(keyword_lower)
                if count > 0:
                    matches.append(keyword)
                    total_score += count * len(keyword)

            if total_score > 0:
                sentences = self._split_sentences(text)
                matched_sentences = []
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(keyword.lower() in sentence_lower for keyword in matches):
                        matched_sentences.append(sentence)

                scored_chunks.append({
                    "chunk_id": chunk_id,
                    "score": total_score,
                    "matched_sentences": matched_sentences[:5],
                    "keywords_found": matches,
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = scored_chunks[:top_k]

        if not top_chunks:
            tool_result = f"No results found for keywords: {keywords}"
            tool_log = {"retrieved_tokens": 0, "chunks_found": 0}
            return tool_result, tool_log

        result_parts = []
        for item in top_chunks:
            if item["matched_sentences"]:
                matched_text = "... " + " ... ".join(item["matched_sentences"]) + " ..."
            else:
                matched_text = "(no exact sentence match)"
            result_parts.append(
                f"Chunk ID: {item['chunk_id']}, Matched keywords in chunk: {matched_text}"
            )

        tool_result = "\n\n".join(result_parts)

        all_matched_sentences = []
        for item in top_chunks:
            all_matched_sentences.extend(item["matched_sentences"])

        retrieved_tokens = (
            len(self.tokenizer.encode("\n".join(all_matched_sentences)))
            if all_matched_sentences
            else 0
        )

        context.add_retrieval_log(
            tool_name="keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "chunks_found": len(top_chunks),
                "chunk_ids": [c["chunk_id"] for c in top_chunks],
                "backend": "legacy",
            },
        )

        tool_log = {"retrieved_tokens": retrieved_tokens, "chunks_found": len(top_chunks)}
        return tool_result, tool_log
