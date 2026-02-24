"""Search agent: wraps BaseAgent with evidence cache integration."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from arag.agent.base import BaseAgent
from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.types import AgentResult, CachedDocument, SubQuestion

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "search_agent.txt"


class SearchAgent:
    """Wraps :class:`BaseAgent` for a single sub-question."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        evidence_cache: EvidenceCache | None = None,
        max_loops: int = 5,
        max_token_budget: int = 64000,
        prompt_path: str | Path | None = None,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.tools = tools
        self.cache = evidence_cache
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    def _build_system_prompt(self, sub_question_text: str, cached_evidence_text: str, original_question: str = "") -> str:
        return self._prompt_template.format(
            sub_question=sub_question_text,
            cached_evidence=cached_evidence_text or "None available.",
            max_loops=self.max_loops,
            original_question=original_question or sub_question_text,
        )

    def _get_semantic_tool(self):
        tool = self.tools.get("semantic_search")
        if tool is None or not hasattr(tool, "embedding_model"):
            return None
        return tool

    def _embed_text(self, text: str, use_query_prompt: bool) -> object | None:
        semantic_tool = self._get_semantic_tool()
        if semantic_tool is None or not text.strip():
            return None
        try:
            with semantic_tool._embedding_lock:
                if use_query_prompt and getattr(semantic_tool, "use_query_prompt", False):
                    try:
                        return semantic_tool.embedding_model.encode(
                            [text],
                            prompt_name=getattr(semantic_tool, "query_prompt_name", "query"),
                            normalize_embeddings=True,
                        )[0]
                    except TypeError:
                        return semantic_tool.embedding_model.encode(
                            [text], normalize_embeddings=True,
                        )[0]
                return semantic_tool.embedding_model.encode(
                    [text], normalize_embeddings=True,
                )[0]
        except Exception as exc:
            logger.debug("Embedding unavailable for cache relevance: %s", exc)
            return None

    async def _gather_cached_evidence(self, sub_question: SubQuestion) -> str:
        if self.cache is None or not self.cache.enabled:
            return ""
        query = f"{sub_question.text} {' '.join(sub_question.search_hints)}".strip()
        relevant_docs = []
        query_embedding = self._embed_text(query, use_query_prompt=True)
        if query_embedding is not None:
            relevant_docs = await self.cache.get_relevant(query_embedding, top_k=8)
        docs = relevant_docs or (await self.cache.get_all_evidence())[:10]
        if not docs:
            return ""
        lines: list[str] = []
        for doc in docs:
            lines.append(f"[Chunk {doc.doc_id}] (agent {doc.source_agent}): {doc.text[:300]}")
        return "\n\n".join(lines)

    async def _write_trajectory_to_cache(self, agent_index: int, trajectory: list[dict]) -> list[str]:
        if self.cache is None or not self.cache.enabled:
            return []
        written_ids: list[str] = []
        read_tool = self.tools.get("read_chunk")
        for entry in trajectory:
            if entry.get("tool_name") != "read_chunk":
                continue
            chunk_ids = entry.get("arguments", {}).get(
                "chunk_ids", entry.get("arguments", {}).get("chunk_id", []),
            )
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            elif isinstance(chunk_ids, int):
                chunk_ids = [str(chunk_ids)]
            for cid in chunk_ids:
                cid = str(cid)
                chunk_text = ""
                if read_tool is not None and hasattr(read_tool, "chunks_dict"):
                    chunk_text = read_tool.chunks_dict.get(cid, "")
                if not chunk_text:
                    chunk_text = str(entry.get("tool_result", ""))[:2000]
                embedding = self._embed_text(chunk_text[:2000], use_query_prompt=False)
                doc = CachedDocument(
                    doc_id=cid, text=chunk_text[:4000], embedding=embedding,
                    source_agent=agent_index, retrieval_score=0.5,
                )
                await self.cache.put(doc)
                written_ids.append(cid)
        seen = set()
        deduped = []
        for cid in written_ids:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        return deduped

    def _resolve_placeholders(self, sub_question: SubQuestion, resolved_answers: dict[int, str]) -> SubQuestion:
        text = sub_question.text
        hints = list(sub_question.search_hints)
        for dep_idx in sub_question.depends_on:
            placeholder = f"[answer_{dep_idx}]"
            answer = resolved_answers.get(dep_idx, "unknown")
            text = text.replace(placeholder, answer)
            hints.append(answer)
        return SubQuestion(
            index=sub_question.index, text=text, search_hints=hints,
            depends_on=sub_question.depends_on, placeholder=sub_question.placeholder,
        )

    def _collect_retrieved_chunks(self, trajectory: list[dict]) -> list[dict]:
        """Extract full chunk texts from read_chunk trajectory entries."""
        chunks: list[dict] = []
        seen_ids: set[str] = set()
        read_tool = self.tools.get("read_chunk")
        for entry in trajectory:
            if entry.get("tool_name") != "read_chunk":
                continue
            for cid in entry.get("arguments", {}).get("chunk_ids", []):
                cid = str(cid)
                if cid in seen_ids:
                    continue
                text = ""
                if read_tool is not None and hasattr(read_tool, "chunks_dict"):
                    text = read_tool.chunks_dict.get(cid, "")
                if text:
                    chunks.append({"id": cid, "text": text})
                    seen_ids.add(cid)
        return chunks

    async def run(
        self,
        sub_question: SubQuestion,
        resolved_answers: dict[int, str] | None = None,
        original_question: str = "",
    ) -> AgentResult:
        t0 = time.monotonic()

        if resolved_answers:
            sub_question = self._resolve_placeholders(sub_question, resolved_answers)

        cached_text = await self._gather_cached_evidence(sub_question)
        system_prompt = self._build_system_prompt(
            sub_question.text, cached_text, original_question=original_question,
        )

        agent = BaseAgent(
            llm_client=self.llm, tools=self.tools, system_prompt=system_prompt,
            max_loops=self.max_loops, max_token_budget=self.max_token_budget,
            verbose=self.verbose,
        )

        context = AgentContext()
        context.source_agent = sub_question.index
        context.evidence_cache = self.cache

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: agent.run(sub_question.text, context=context),
            )
        except Exception as exc:
            logger.error("Agent %d failed: %s", sub_question.index, exc)
            return AgentResult(
                sub_question_index=sub_question.index,
                answer="",
                error=str(exc),
                wall_clock_seconds=time.monotonic() - t0,
            )

        trajectory = result.get("trajectory", [])

        evidence_ids = await self._write_trajectory_to_cache(sub_question.index, trajectory)
        elapsed = time.monotonic() - t0

        answer = result.get("answer", "")
        confidence = result.get("confidence", 1.0)
        supporting = result.get("supporting_chunk_ids", [])
        if supporting:
            seen = set(evidence_ids)
            for cid in supporting:
                if cid not in seen:
                    evidence_ids.append(cid)
                    seen.add(cid)

        # Collect full retrieved chunk texts for the aggregator's unified evidence pool
        retrieved_chunks = self._collect_retrieved_chunks(trajectory)

        logger.info(
            "Agent %d: '%s' → '%s' (%d loops, %.1fs, %d chunks retrieved)",
            sub_question.index, sub_question.text[:40], answer[:60],
            result.get("loops", 0), elapsed, len(retrieved_chunks),
        )

        return AgentResult(
            sub_question_index=sub_question.index,
            answer=answer,
            evidence_doc_ids=evidence_ids,
            trajectory=trajectory,
            loops=result.get("loops", 0),
            total_tokens=result.get("total_retrieved_tokens", 0),
            wall_clock_seconds=elapsed,
            confidence=confidence,
            retrieved_chunks=retrieved_chunks,
        )
