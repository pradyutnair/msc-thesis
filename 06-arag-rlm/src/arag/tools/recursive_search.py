"""Recursive context processing tool: retrieve wide, filter with LLM, read relevant.

This is the core RLM-for-RAG contribution. Instead of retrieving top-5 and
hoping the answer is there, we:
1. Retrieve top-20 snippets (wide retrieval)
2. Use a cheap LLM call to filter: "which are relevant to [question]?"
3. Read only the filtered chunks (typically 3-5)

This increases recall without proportional token cost.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext
    from arag.core.llm import LLMClient


FILTER_PROMPT = """You are a relevance filter. Given a question and a list of document snippets, return ONLY the IDs of snippets that contain information useful for answering the question.

Question: {question}

Snippets:
{snippets}

Return a JSON array of relevant chunk IDs, e.g. ["3", "17", "42"]. Return [] if none are relevant. ONLY output the JSON array, nothing else."""


class RecursiveSearchTool(BaseTool):
    """Wide retrieval + LLM filtering + targeted reading."""

    def __init__(self, keyword_tool, semantic_tool, read_tool, llm_client: "LLMClient"):
        self.keyword_tool = keyword_tool
        self.semantic_tool = semantic_tool
        self.read_tool = read_tool
        self.llm = llm_client

    @property
    def name(self) -> str:
        return "recursive_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "recursive_search",
                "description": (
                    "Deep search: retrieves many chunks, uses LLM to filter for "
                    "relevance, then reads only the relevant ones. Use this when "
                    "a normal search might miss the answer — especially for specific "
                    "entity names or multi-hop facts. More thorough than search_and_read."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for.",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question you are trying to answer (used for relevance filtering).",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["keyword", "semantic"],
                            "description": "Search method.",
                            "default": "keyword",
                        },
                    },
                    "required": ["query", "question"],
                },
            },
        }

    @staticmethod
    def _extract_chunk_ids(search_result: str) -> List[str]:
        raw_ids = re.findall(r"Chunk ID:\s*(\S+)", search_result)
        return [cid.rstrip(",.;:") for cid in raw_ids]

    def _filter_with_llm(self, context: "AgentContext", question: str, chunk_ids: List[str]) -> List[str]:
        """Use a cheap LLM call to filter chunks by relevance."""
        if not chunk_ids:
            return []

        # Build snippets from chunk previews (first 200 chars each)
        chunks_dict = getattr(self.read_tool, "chunks_dict", {}) or {}
        snippet_parts = []
        for cid in chunk_ids:
            text = chunks_dict.get(str(cid), "")
            preview = text[:250].replace("\n", " ") if text else "(empty)"
            snippet_parts.append(f"[{cid}] {preview}")

        snippets_text = "\n".join(snippet_parts)
        prompt = FILTER_PROMPT.format(question=question, snippets=snippets_text)

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            # Track tokens
            context.add_llm_usage(
                phase="recursive_filter",
                input_tokens=response.get("input_tokens", 0),
                output_tokens=response.get("output_tokens", 0),
                metadata={"num_candidates": len(chunk_ids)},
            )

            content = response["message"].get("content", "")
            # Strip thinking tags
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Parse JSON array
            match = re.search(r"\[.*?\]", content, re.DOTALL)
            if match:
                filtered = json.loads(match.group())
                return [str(cid) for cid in filtered if str(cid) in set(chunk_ids)]
        except Exception:
            pass

        # Fallback: return top 5
        return chunk_ids[:5]

    def execute(
        self,
        context: "AgentContext",
        query: str,
        question: str,
        method: str = "keyword",
    ) -> Tuple[str, Dict[str, Any]]:
        # Step 1: Wide retrieval (top-20)
        wide_k = 20
        if method == "keyword":
            keywords = [k.strip() for k in query.split(",") if k.strip()]
            if not keywords:
                keywords = [query]
            search_result, search_log = self.keyword_tool.execute(
                context, keywords=keywords, top_k=wide_k,
            )
        elif method == "semantic":
            if self.semantic_tool is None:
                return "Error: semantic search not available", {"retrieved_tokens": 0}
            search_result, search_log = self.semantic_tool.execute(
                context, query=query, top_k=wide_k,
            )
        else:
            return f"Error: unknown method '{method}'", {"retrieved_tokens": 0}

        # Step 2: Extract all candidate chunk IDs
        all_chunk_ids = self._extract_chunk_ids(search_result)
        if not all_chunk_ids:
            return f"No results found for: {query}", {
                "retrieved_tokens": 0, "chunks_found": 0,
                "method": method, "wide_k": wide_k,
            }

        # Step 3: LLM filter
        filtered_ids = self._filter_with_llm(context, question, all_chunk_ids)
        if not filtered_ids:
            # Fallback to top 3
            filtered_ids = all_chunk_ids[:3]

        # Step 4: Read filtered chunks
        read_result, read_log = self.read_tool.execute(context, chunk_ids=filtered_ids)

        total_tokens = search_log.get("retrieved_tokens", 0) + read_log.get("retrieved_tokens", 0)

        combined_log = {
            "retrieved_tokens": total_tokens,
            "method": method,
            "query": query,
            "wide_candidates": len(all_chunk_ids),
            "filtered_to": len(filtered_ids),
            "filtered_ids": filtered_ids,
            "is_recursive": True,
        }

        return read_result, combined_log
