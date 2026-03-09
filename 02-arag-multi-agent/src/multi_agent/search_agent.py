"""Search agent: wraps BaseAgent with evidence cache integration."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from arag.agent.base import BaseAgent
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.types import AgentResult, CachedDocument, SubQuestion

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "search_agent.txt"


class SearchAgent:
    """Wraps :class:`BaseAgent` for a single sub-question.

    Adds:
    - Scoped system prompt focused on the sub-question
    - Evidence cache read-through (inject cached evidence into prompt)
    - Evidence cache write-through (write read_chunk results to cache)
    - Placeholder resolution for bridge [answer_N] tokens
    """

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

    def _build_system_prompt(self, sub_question_text: str, cached_evidence_text: str) -> str:
        return self._prompt_template.format(
            sub_question=sub_question_text,
            cached_evidence=cached_evidence_text or "None available.",
            max_loops=self.max_loops,
        )

    async def _gather_cached_evidence(self) -> str:
        if self.cache is None or not self.cache.enabled:
            return ""
        all_docs = await self.cache.get_all_evidence()
        if not all_docs:
            return ""
        lines: list[str] = []
        for doc in all_docs[:10]:
            snippet = doc.text[:300]
            lines.append(f"[Chunk {doc.doc_id}] (agent {doc.source_agent}): {snippet}")
        return "\n\n".join(lines)

    async def _write_trajectory_to_cache(self, agent_index: int, trajectory: list[dict]) -> list[str]:
        if self.cache is None or not self.cache.enabled:
            return []
        written_ids: list[str] = []
        for entry in trajectory:
            if entry.get("tool_name") != "read_chunk":
                continue
            result_text = entry.get("tool_result", "")
            chunk_ids = entry.get("arguments", {}).get("chunk_ids", entry.get("arguments", {}).get("chunk_id", []))
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            elif isinstance(chunk_ids, int):
                chunk_ids = [str(chunk_ids)]
            for cid in chunk_ids:
                cid = str(cid)
                doc = CachedDocument(
                    doc_id=cid,
                    text=result_text[:2000],
                    source_agent=agent_index,
                    retrieval_score=0.5,
                )
                await self.cache.put(doc)
                written_ids.append(cid)
        return written_ids

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

    async def run(self, sub_question: SubQuestion, resolved_answers: dict[int, str] | None = None) -> AgentResult:
        t0 = time.monotonic()

        if resolved_answers:
            sub_question = self._resolve_placeholders(sub_question, resolved_answers)

        cached_text = await self._gather_cached_evidence()
        system_prompt = self._build_system_prompt(sub_question.text, cached_text)

        agent = BaseAgent(
            llm_client=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            max_loops=self.max_loops,
            max_token_budget=self.max_token_budget,
            verbose=self.verbose,
        )

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, agent.run, sub_question.text)
        except Exception as exc:
            logger.error("Agent %d failed: %s", sub_question.index, exc)
            return AgentResult(
                sub_question_index=sub_question.index, answer="",
                error=str(exc), wall_clock_seconds=time.monotonic() - t0,
            )

        evidence_ids = await self._write_trajectory_to_cache(sub_question.index, result.get("trajectory", []))
        elapsed = time.monotonic() - t0

        answer = result.get("answer", "")
        logger.info(
            "Agent %d: '%s' → '%s' (%d loops, %.1fs)",
            sub_question.index, sub_question.text[:40], answer[:60],
            result.get("loops", 0), elapsed,
        )

        return AgentResult(
            sub_question_index=sub_question.index,
            answer=answer,
            evidence_doc_ids=evidence_ids,
            trajectory=result.get("trajectory", []),
            loops=result.get("loops", 0),
            total_tokens=result.get("total_retrieved_tokens", 0),
            wall_clock_seconds=elapsed,
        )
