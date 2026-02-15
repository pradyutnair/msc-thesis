#!/usr/bin/env python3
"""
A-RAG tool scaffold for FlashRAG wiki18_100w corpus.

Goal:
- Keep A-RAG tool interface (keyword_search, semantic_search, read_chunk)
- Use large-corpus-safe access (JSONL streaming + id->offset map)
- Use FAISS dense index for semantic_search
- Use SQLite FTS5 for keyword_search

Docs:
- A-RAG tool interface: external/arag/src/arag/tools/base.py
- A-RAG registry usage: external/arag/src/arag/tools/registry.py
- vLLM OpenAI server (for model side): https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- FAISS usage: https://github.com/facebookresearch/faiss/wiki
- SQLite FTS5: https://www.sqlite.org/fts5.html
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken

from arag.tools.base import BaseTool


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?\n]+", text)
    return [p.strip() for p in parts if p.strip()]


class WikiCorpusStore:
    """
    Memory-safe corpus accessor for huge jsonl corpus.

    Expects each line like:
    {"id": "123", "contents": "..."}
    """

    def __init__(self, corpus_jsonl: str, id_offset_json: str):
        self.corpus_jsonl = Path(corpus_jsonl)
        self.id_offset_json = Path(id_offset_json)
        self.id2offset = self._load_id_offset()

    def _load_id_offset(self) -> Dict[str, int]:
        if not self.id_offset_json.exists():
            raise FileNotFoundError(
                f"id->offset map missing: {self.id_offset_json}. "
                "Build it first with 06_build_keyword_fts_index.py."
            )
        data = json.loads(self.id_offset_json.read_text(encoding="utf-8"))
        # JSON stores keys as str already
        return {str(k): int(v) for k, v in data.items()}

    def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        cid = str(chunk_id)
        if cid not in self.id2offset:
            raise KeyError(f"chunk_id not found: {cid}")

        with self.corpus_jsonl.open("rb") as f:
            f.seek(self.id2offset[cid])
            line = f.readline().decode("utf-8")
        row = json.loads(line)
        return {"id": str(row["id"]), "text": row["contents"]}


class FlashragKeywordSearchTool(BaseTool):
    """
    Keyword search via SQLite FTS5 (scales better than in-memory BM25 on 21M docs).
    """

    def __init__(self, sqlite_db: str, corpus_store: WikiCorpusStore):
        self.sqlite_db = Path(sqlite_db)
        self.corpus_store = corpus_store
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": "Keyword search over wiki18_100w corpus using FTS5. Returns chunk IDs with matched snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["keywords"],
                },
            },
        }

    def execute(self, context, keywords: List[str], top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(max(int(top_k), 1), 20)
        query = " OR ".join([f'"{k}"' for k in keywords if k.strip()])
        if not query:
            return "No valid keywords provided.", {"retrieved_tokens": 0, "chunks_found": 0}

        sql = """
        SELECT id, snippet(passages_fts, 1, '[', ']', ' ... ', 24) AS snip
        FROM passages_fts
        WHERE passages_fts MATCH ?
        LIMIT ?;
        """
        rows: List[Tuple[str, str]] = []
        with sqlite3.connect(self.sqlite_db) as conn:
            cur = conn.cursor()
            cur.execute(sql, (query, top_k))
            rows = [(str(r[0]), r[1]) for r in cur.fetchall()]

        if not rows:
            msg = f"No results found for keywords={keywords}"
            context.add_retrieval_log("keyword_search", 0, {"keywords": keywords, "chunks_found": 0})
            return msg, {"retrieved_tokens": 0, "chunks_found": 0}

        lines = [f"Chunk ID: {cid}\nMatched: {snip}" for cid, snip in rows]
        tool_result = "\n\n".join(lines)
        tok = len(self.tokenizer.encode(tool_result))

        context.add_retrieval_log(
            "keyword_search",
            tok,
            {"keywords": keywords, "chunks_found": len(rows), "chunk_ids": [cid for cid, _ in rows]},
        )
        return tool_result, {"retrieved_tokens": tok, "chunks_found": len(rows)}


class FlashragSemanticSearchTool(BaseTool):
    """
    Semantic search via E5 + FAISS.

    Assumption:
    - FAISS row index corresponds to chunk id integer (0..N-1) in wiki18_100w.
    If your index uses a custom ID map, replace `chunk_id = str(int(idx))`
    with your mapping lookup.
    """

    def __init__(self, faiss_index_path: str, embedding_model: str, corpus_store: WikiCorpusStore, device: str = "cuda:0"):
        self.index = faiss.read_index(str(faiss_index_path))
        self.embedder = SentenceTransformer(embedding_model, device=device)
        self.corpus_store = corpus_store
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "semantic_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": "Dense retrieval over wiki18_100w using E5+FAISS. Returns chunk IDs and short sentence snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, context, query: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(max(int(top_k), 1), 20)
        vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, idxs = self.index.search(vec, top_k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        lines = []
        chunk_ids = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            chunk_id = str(int(idx))  # replace if you use custom faiss-id mapping
            row = self.corpus_store.get_chunk(chunk_id)
            sentences = split_sentences(row["text"])
            snippet = sentences[0] if sentences else row["text"][:280]
            lines.append(f"Chunk ID: {chunk_id} (score={score:.4f})\nMatched: {snippet}")
            chunk_ids.append(chunk_id)

        if not lines:
            msg = f"No semantic hits for query={query}"
            context.add_retrieval_log("semantic_search", 0, {"query": query, "chunks_found": 0})
            return msg, {"retrieved_tokens": 0, "chunks_found": 0}

        tool_result = "\n\n".join(lines)
        tok = len(self.tokenizer.encode(tool_result))
        context.add_retrieval_log(
            "semantic_search",
            tok,
            {"query": query, "chunks_found": len(chunk_ids), "chunk_ids": chunk_ids},
        )
        return tool_result, {"retrieved_tokens": tok, "chunks_found": len(chunk_ids)}


class FlashragReadChunkTool(BaseTool):
    def __init__(self, corpus_store: WikiCorpusStore):
        self.corpus_store = corpus_store
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")

    @property
    def name(self) -> str:
        return "read_chunk"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": "Read full content of one or more chunk IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["chunk_ids"],
                },
            },
        }

    def execute(self, context, chunk_ids: List[str] = None, chunk_id: str = None) -> Tuple[str, Dict[str, Any]]:
        ids = chunk_ids if chunk_ids is not None else ([str(chunk_id)] if chunk_id is not None else [])
        if not ids:
            return "No chunk IDs provided.", {"retrieved_tokens": 0}

        blocks = []
        total_tokens = 0
        new_chunks = []
        already_read = []

        for cid in [str(x) for x in ids]:
            if context.is_chunk_read(cid):
                already_read.append(cid)
                blocks.append(f"[Chunk {cid}] (already read)")
                continue

            row = self.corpus_store.get_chunk(cid)
            text = row["text"]
            blocks.append(f"[Chunk {cid}]\n{text}")
            context.mark_chunk_as_read(cid)
            new_chunks.append(cid)
            total_tokens += len(self.tokenizer.encode(text))

        out = "\n\n".join(blocks)
        context.add_retrieval_log(
            "read_chunk",
            total_tokens,
            {"chunk_ids_requested": ids, "new_chunks_read": new_chunks, "already_read": already_read},
        )
        return out, {
            "retrieved_tokens": total_tokens,
            "new_chunks_count": len(new_chunks),
            "already_read_count": len(already_read),
        }




class FinishTool(BaseTool):
    """Tool for the agent to submit its final answer."""

    @property
    def name(self) -> str:
        return "finish"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Submit your final answer. The answer should be concise and directly answer the question. For yes/no questions, answer 'yes' or 'no'. For factual questions, give the specific entity, name, date, or number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your final concise answer to the question."
                        },
                    },
                    "required": ["answer"],
                },
            },
        }

    def execute(self, context, answer: str = "") -> Tuple[str, Dict[str, Any]]:
        return answer, {"finish": True, "retrieved_tokens": 0}


def build_tools_for_arag(
    corpus_jsonl: str,
    id_offset_json: str,
    sqlite_db: str,
    faiss_index_path: str,
    embedding_model: str = "intfloat/e5-base-v2",
    device: str = "cuda:0",
):
    """
    Helper to plug into A-RAG batch runner.
    """
    from arag.tools.registry import ToolRegistry

    store = WikiCorpusStore(corpus_jsonl=corpus_jsonl, id_offset_json=id_offset_json)
    reg = ToolRegistry()
    reg.register(FlashragKeywordSearchTool(sqlite_db=sqlite_db, corpus_store=store))
    reg.register(FlashragSemanticSearchTool(
        faiss_index_path=faiss_index_path,
        embedding_model=embedding_model,
        corpus_store=store,
        device=device,
    ))
    reg.register(FlashragReadChunkTool(corpus_store=store))
    reg.register(FinishTool())
    return reg