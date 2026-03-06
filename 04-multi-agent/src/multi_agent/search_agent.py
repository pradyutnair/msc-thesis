"""Search agent: wraps BaseAgent with evidence cache integration."""

from __future__ import annotations

import logging
import re
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
_EXTRACT_PROMPT_PATH = Path(__file__).parent / "m5" / "prompts" / "extract_evidence.txt"


def is_unsupported_answer(
    answer: str,
    loops: int,
    evidence_count: int,
    retrieved_chunk_count: int,
) -> bool:
    """Heuristic unsupported-answer detector.

    A completion is considered unsupported if it returns a non-empty answer after
    searching but surfaces no supporting evidence IDs and no retrieved chunks.
    """
    return bool(answer.strip()) and loops > 0 and evidence_count <= 0 and retrieved_chunk_count <= 0


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
        dual_retrieval_on_low_evidence: bool = False,
        low_evidence_threshold: int = 2,
        dual_retrieval_top_k: int = 4,
        extract_evidence: bool = False,
        extract_prompt_path: str | Path | None = None,
        extract_max_bullets: int = 3,
    ):
        self.llm = llm_client
        self.tools = tools
        self.cache = evidence_cache
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.verbose = verbose

        self.dual_retrieval_on_low_evidence = dual_retrieval_on_low_evidence
        self.low_evidence_threshold = max(1, int(low_evidence_threshold))
        self.dual_retrieval_top_k = max(1, int(dual_retrieval_top_k))

        self.extract_evidence = extract_evidence
        self.extract_max_bullets = max(1, int(extract_max_bullets))

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

        extract_path = Path(extract_prompt_path) if extract_prompt_path else _EXTRACT_PROMPT_PATH
        self._extract_prompt_template = extract_path.read_text(encoding="utf-8")

    def _build_system_prompt(
        self,
        sub_question_text: str,
        cached_evidence_text: str,
        original_question: str = "",
    ) -> str:
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
            tool_name = entry.get("tool_name")
            if tool_name not in {"read_chunk", "search_and_read"}:
                continue

            if tool_name == "read_chunk":
                chunk_ids = entry.get("arguments", {}).get(
                    "chunk_ids", entry.get("arguments", {}).get("chunk_id", []),
                )
            else:
                chunk_ids = entry.get("chunk_ids_read", [])

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
                    doc_id=cid,
                    text=chunk_text[:4000],
                    embedding=embedding,
                    source_agent=agent_index,
                    retrieval_score=0.5,
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
            index=sub_question.index,
            text=text,
            search_hints=hints,
            depends_on=sub_question.depends_on,
            placeholder=sub_question.placeholder,
        )

    def _collect_retrieved_chunks(self, trajectory: list[dict]) -> list[dict]:
        """Extract full chunk texts from read_chunk and search_and_read entries."""
        chunks: list[dict] = []
        seen_ids: set[str] = set()
        read_tool = self.tools.get("read_chunk")
        for entry in trajectory:
            tool_name = entry.get("tool_name", "")
            chunk_ids_to_collect: list = []

            if tool_name == "read_chunk":
                chunk_ids_to_collect = entry.get("arguments", {}).get("chunk_ids", [])
            elif tool_name == "search_and_read":
                chunk_ids_to_collect = entry.get("chunk_ids_read", [])

            for cid in chunk_ids_to_collect:
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

    @staticmethod
    def _parse_method_from_task_text(task_text: str) -> str:
        m = re.search(r"method:\s*(keyword|semantic)", task_text, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return "keyword"

    def _primary_query_for_task(self, sub_question: SubQuestion) -> str:
        if sub_question.search_hints:
            return sub_question.search_hints[0]
        m = re.search(r"Suggested search:\s*(.+?)\s*\(method:", sub_question.text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return sub_question.text

    def _merge_chunks(self, base: list[dict], extra: list[dict]) -> list[dict]:
        seen = {str(c.get("id", "")) for c in base}
        merged = list(base)
        for chunk in extra:
            cid = str(chunk.get("id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                merged.append(chunk)
        return merged

    def _recover_low_evidence(
        self,
        context: AgentContext,
        sub_question: SubQuestion,
        retrieved_chunks: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """Optional semantic/keyword follow-up when evidence is sparse."""
        if not self.dual_retrieval_on_low_evidence:
            return retrieved_chunks, []

        if len(retrieved_chunks) >= self.low_evidence_threshold:
            return retrieved_chunks, []

        search_tool = self.tools.get("search_and_read")
        read_tool = self.tools.get("read_chunk")
        if search_tool is None or read_tool is None:
            return retrieved_chunks, []

        primary_method = self._parse_method_from_task_text(sub_question.text)
        alt_method = "semantic" if primary_method == "keyword" else "keyword"
        query = self._primary_query_for_task(sub_question)

        extra_chunk_ids: list[str] = []
        extra_chunks: list[dict] = []
        top_k = min(5, self.dual_retrieval_top_k + (1 if sub_question.depends_on else 0))

        try:
            _, log = search_tool.execute(
                context=context,
                query=query,
                method=alt_method,
                top_k=top_k,
            )
            for cid in log.get("chunk_ids_read", []) or []:
                cid_s = str(cid)
                text = read_tool.chunks_dict.get(cid_s, "") if hasattr(read_tool, "chunks_dict") else ""
                if text:
                    extra_chunks.append({"id": cid_s, "text": text})
                    extra_chunk_ids.append(cid_s)
        except Exception as exc:
            logger.debug("Dual retrieval fallback failed for task %s: %s", sub_question.index, exc)

        return self._merge_chunks(retrieved_chunks, extra_chunks), list(dict.fromkeys(extra_chunk_ids))

    async def _extract_evidence_bullets(
        self,
        sub_question: SubQuestion,
        chunks: list[dict],
    ) -> list[str]:
        """Extract concise evidence bullets from retrieved chunks."""
        if not self.extract_evidence or not chunks:
            return []

        bullets: list[str] = []
        focus = sub_question.text.strip()

        for chunk in chunks[:5]:
            text = str(chunk.get("text", ""))[:12000]
            if not text.strip():
                continue
            prompt = self._extract_prompt_template.format(
                focus=focus,
                text=text,
            )
            try:
                response = await self.llm.async_chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    temperature=0.0,
                )
                raw = (response.get("message", {}) or {}).get("content", "")
            except Exception as exc:
                logger.debug("Evidence extraction failed for task %s: %s", sub_question.index, exc)
                continue

            for line in raw.splitlines():
                cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
                if not cleaned:
                    continue
                if cleaned.upper() == "NO_RELEVANT_EVIDENCE":
                    continue
                bullets.append(cleaned)

        deduped: list[str] = []
        seen = set()
        for bullet in bullets:
            norm = re.sub(r"\s+", " ", bullet.lower())
            if norm in seen:
                continue
            seen.add(norm)
            deduped.append(bullet)
            if len(deduped) >= self.extract_max_bullets:
                break
        return deduped

    async def run(
        self,
        sub_question: SubQuestion,
        resolved_answers: dict[int, str] | None = None,
        original_question: str = "",
        chain_evidence: str = "",
    ) -> AgentResult:
        t0 = time.monotonic()

        if resolved_answers:
            sub_question = self._resolve_placeholders(sub_question, resolved_answers)

        cached_text = chain_evidence or await self._gather_cached_evidence(sub_question)
        system_prompt = self._build_system_prompt(
            sub_question.text,
            cached_text,
            original_question=original_question,
        )

        agent = BaseAgent(
            llm_client=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            max_loops=self.max_loops,
            max_token_budget=self.max_token_budget,
            verbose=self.verbose,
        )

        context = AgentContext()
        context.source_agent = sub_question.index
        context.evidence_cache = self.cache

        try:
            result = await agent.async_run(sub_question.text, context=context)
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
                cid_s = str(cid)
                if cid_s not in seen:
                    evidence_ids.append(cid_s)
                    seen.add(cid_s)

        retrieved_chunks = self._collect_retrieved_chunks(trajectory)
        retrieved_chunks, extra_ids = self._recover_low_evidence(
            context=context,
            sub_question=sub_question,
            retrieved_chunks=retrieved_chunks,
        )
        if extra_ids:
            seen = set(evidence_ids)
            for cid in extra_ids:
                if cid not in seen:
                    evidence_ids.append(cid)
                    seen.add(cid)

        extracted_evidence = await self._extract_evidence_bullets(sub_question, retrieved_chunks)

        evidence_count = max(len(evidence_ids), len(retrieved_chunks))
        unsupported = is_unsupported_answer(
            answer=answer,
            loops=result.get("loops", 0),
            evidence_count=evidence_count,
            retrieved_chunk_count=len(retrieved_chunks),
        )
        if unsupported:
            confidence = min(float(confidence or 0.0), 0.2)

        logger.info(
            "Agent %d: '%s' -> '%s' (%d loops, %.1fs, %d chunks retrieved, unsupported=%s)",
            sub_question.index,
            sub_question.text[:40],
            answer[:60],
            result.get("loops", 0),
            elapsed,
            len(retrieved_chunks),
            unsupported,
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
            unsupported_answer=unsupported,
            extracted_evidence=extracted_evidence,
            evidence_count=evidence_count,
        )
