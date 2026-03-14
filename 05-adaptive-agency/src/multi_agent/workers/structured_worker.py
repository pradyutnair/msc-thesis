"""StructuredWorker: entity-extraction + systematic retrieval + summarization.

Fixed 3-step pipeline per sub-question (no ReAct loop):
  1. LLM extracts entities and search queries
  2. Systematic keyword + semantic search per entity/query
  3. LLM summarizes retrieved chunks into an answer
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from arag.tools.search_and_read import SearchAndReadTool
from multi_agent.autonomous_agent import AutonomousAgent
from multi_agent.blackboard import Blackboard
from multi_agent.types import EvidenceEntry, RetrievalMode, SubQuestionStatus
from multi_agent.utils import (
    clean_answer,
    dedupe_keep_order,
    parse_json_robust,
    resolve_placeholders,
    strip_llm_wrappers,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class StructuredWorker(AutonomousAgent):
    """Deterministic structured retrieval worker for a single sub-question."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        base_tools: ToolRegistry,
        assigned_sq_id: int,
        retrieval_top_k: int = 10,
        max_queries_per_entity: int = 6,
        max_doc_chars: int = 2000,
        retry_low_confidence: bool = True,
        confidence_threshold: float = 0.6,
    ):
        super().__init__(agent_id=agent_id, agent_type="structured_worker")
        self.llm = llm_client
        self.assigned_sq_id = assigned_sq_id
        self.retrieval_top_k = retrieval_top_k
        self.max_queries_per_entity = max_queries_per_entity
        self.max_doc_chars = max_doc_chars
        self.retry_low_confidence = retry_low_confidence
        self.confidence_threshold = confidence_threshold
        self._done = False
        self._last_epoch = 0

        self._entity_prompt = (_PROMPTS_DIR / "entity_id.txt").read_text(encoding="utf-8")
        self._summarizer_prompt = (_PROMPTS_DIR / "summarizer.txt").read_text(encoding="utf-8")

        self.read_tool = base_tools.get("read_chunk")
        self.search_tool = SearchAndReadTool(
            keyword_tool=base_tools.get("keyword_search"),
            semantic_tool=base_tools.get("semantic_search"),
            read_tool=self.read_tool,
        )

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_worker(self.agent_id)

    def should_act(self, observation: dict[str, Any]) -> bool:
        epoch = observation.get("redecomposition_epoch", 0)
        if epoch > self._last_epoch:
            self._done = False
            self._last_epoch = epoch
        if self._done:
            return False
        if observation["claimed_sub_question"] is not None:
            return True
        return any(sq["id"] == self.assigned_sq_id for sq in observation["available_sub_questions"])

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        sq_dict = observation["claimed_sub_question"]
        if sq_dict is None:
            if not await blackboard.claim_sub_question(self.assigned_sq_id, self.agent_id):
                return 0
            observation = await self.observe(blackboard)
            sq_dict = observation["claimed_sub_question"]
            if sq_dict is None:
                return 0

        sq_id = sq_dict["id"]
        entity_registry = observation["entity_registry"]
        resolved_text = resolve_placeholders(sq_dict["text"], entity_registry)
        original_question = observation.get("question", resolved_text)
        bb_context = observation.get("blackboard_context", "")
        warm_start = observation.get("warm_start_context", "")
        planner_queries = [
            resolve_placeholders(q, entity_registry)
            for q in (observation.get("search_queries") or [])
        ]

        combined_context = bb_context
        if warm_start:
            combined_context = f"{warm_start}\n\n{bb_context}" if bb_context else warm_start

        total_tokens = 0

        # Step 1: entity extraction via LLM
        entity_requests, tok = await self._extract_entities(resolved_text, original_question, combined_context)
        total_tokens += tok

        # Merge planner-generated queries into entity requests
        if planner_queries:
            if entity_requests:
                existing = set(q for er in entity_requests for q in er.get("queries", []))
                extra = [q for q in planner_queries if q not in existing]
                entity_requests[0]["queries"] = entity_requests[0]["queries"] + extra
            else:
                entity_requests = [{"entity": resolved_text[:80], "goal": resolved_text, "queries": planner_queries}]

        # Step 2+3: retrieve and summarize per entity
        best_answer, best_confidence, all_evidence, tok = await self._retrieve_and_summarize(
            entity_requests, resolved_text, original_question, sq_id,
        )
        total_tokens += tok

        # Retry: use first-round answer to generate targeted follow-up queries
        if self.retry_low_confidence and best_confidence < self.confidence_threshold:
            retry_queries = self._build_retry_queries(resolved_text, best_answer, entity_requests)
            retry_requests = [{"entity": resolved_text[:80], "goal": resolved_text, "queries": retry_queries}]
            retry_answer, retry_conf, retry_evidence, tok = await self._retrieve_and_summarize(
                retry_requests, resolved_text, original_question, sq_id,
            )
            total_tokens += tok
            all_evidence.extend(retry_evidence)
            if retry_conf > best_confidence and retry_answer:
                best_answer, best_confidence = retry_answer, retry_conf

        best_answer = self._clean_structured_answer(best_answer)

        await blackboard.post_evidence(all_evidence, sq_id, best_answer, self.agent_id)
        is_usable = bool(best_answer) and best_answer.lower() not in ("unknown", "error", "")
        await blackboard.verify_sub_question(sq_id, verified=is_usable)
        await blackboard.record_mode_tokens(RetrievalMode.STRUCTURED, total_tokens)

        self._done = is_usable
        logger.info(
            "%s: SQ-%d -> '%s' (conf=%.2f, %d evidence, %d tokens, done=%s)",
            self.agent_id, sq_id, best_answer[:60],
            best_confidence, len(all_evidence), total_tokens, self._done,
        )
        return total_tokens

    # ── Pipeline steps ───────────────────────────────────────────────

    async def _call_llm(self, prompt: str) -> tuple[str, int]:
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        raw = strip_llm_wrappers((response.get("message") or {}).get("content", ""))
        tokens = int(response.get("cost", 0.0) * 1_000_000)
        return raw, tokens

    async def _extract_entities(
        self, hop_question: str, original_question: str, blackboard_context: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Step 1: LLM extracts entities and search queries from the sub-question."""
        prompt = self._entity_prompt.format(
            hop_question=hop_question,
            original_question=original_question,
            blackboard_context=blackboard_context or "No context yet.",
        )
        raw, tokens = await self._call_llm(prompt)

        try:
            data = parse_json_robust(raw)
        except ValueError:
            return [{"entity": hop_question[:80], "goal": hop_question, "queries": [hop_question[:50]]}], tokens

        entities_raw = data.get("entities", [])
        if not isinstance(entities_raw, list) or not entities_raw:
            return [{"entity": hop_question[:80], "goal": hop_question, "queries": [hop_question[:50]]}], tokens

        normalized = []
        for item in entities_raw[:3]:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            goal = str(item.get("goal", "")).strip()
            queries_raw = item.get("queries", [])
            if not isinstance(queries_raw, list):
                queries_raw = []
            queries = dedupe_keep_order([str(q).strip() for q in queries_raw if str(q).strip()])
            if not queries and entity:
                queries = [entity]
            if entity:
                normalized.append({
                    "entity": entity,
                    "goal": goal or f"Find facts about {entity}",
                    "queries": queries[:self.max_queries_per_entity],
                })

        if not normalized:
            return [{"entity": hop_question[:80], "goal": hop_question, "queries": [hop_question[:50]]}], tokens
        return normalized, tokens

    async def _systematic_retrieve(self, queries: list[str]) -> dict[str, dict[str, str]]:
        """Step 2: keyword + semantic search, ranked by multi-query hit count."""
        context = AgentContext()
        hit_count: dict[str, int] = {}
        best_rank: dict[str, int] = {}
        chunk_texts: dict[str, str] = {}

        for method in ("keyword", "semantic"):
            for query in queries:
                _, log = await asyncio.to_thread(
                    self.search_tool.execute,
                    context, query, method, self.retrieval_top_k,
                )
                for rank, chunk_id in enumerate(log.get("chunk_ids_read", []) or []):
                    cid = str(chunk_id)
                    hit_count[cid] = hit_count.get(cid, 0) + 1
                    best_rank[cid] = min(best_rank.get(cid, 999), rank)
                    if cid not in chunk_texts:
                        if self.read_tool is not None and hasattr(self.read_tool, "chunks_dict"):
                            text = str(self.read_tool.chunks_dict.get(cid, ""))
                            if text:
                                chunk_texts[cid] = text

        ranked_ids = sorted(
            chunk_texts.keys(),
            key=lambda cid: (-hit_count[cid], best_rank[cid]),
        )

        result: dict[str, dict[str, str]] = {}
        for cid in ranked_ids:
            result[cid] = {"id": cid, "text": chunk_texts[cid]}
        return result

    async def _summarize(
        self,
        entity_request: dict[str, Any],
        chunks: dict[str, dict[str, str]],
        hop_question: str,
        original_question: str,
    ) -> tuple[dict[str, Any], int]:
        """Step 3: LLM summarizes retrieved chunks into answer + confidence."""
        doc_blocks = []
        chunk_ids = []
        total_chars = 0
        for cid, chunk in chunks.items():
            text = chunk["text"][:self.max_doc_chars]
            if total_chars + len(text) > 20_000:
                break
            chunk_ids.append(cid)
            doc_blocks.append(f"[Chunk {cid}]\n{text}")
            total_chars += len(text)

        prompt = self._summarizer_prompt.format(
            hop_question=hop_question,
            original_question=original_question,
            entity=entity_request["entity"],
            goal=entity_request["goal"],
            documents="\n\n".join(doc_blocks) if doc_blocks else "No retrieved documents.",
        )
        raw, tokens = await self._call_llm(prompt)

        try:
            data = parse_json_robust(raw)
            answer = str(data.get("answer", "")).strip()
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            supporting = data.get("supporting_chunk_ids", [])
            if not isinstance(supporting, list):
                supporting = []
            supporting = dedupe_keep_order([str(x) for x in supporting])
            if not supporting:
                supporting = chunk_ids[:5]
            return {"answer": answer, "confidence": confidence, "supporting_chunk_ids": supporting}, tokens
        except (ValueError, TypeError):
            return {"answer": "", "confidence": 0.0, "supporting_chunk_ids": chunk_ids[:5]}, tokens

    async def _retrieve_and_summarize(
        self,
        entity_requests: list[dict[str, Any]],
        hop_question: str,
        original_question: str,
        sq_id: int,
    ) -> tuple[str, float, list[EvidenceEntry], int]:
        """Run retrieve + summarize for a list of entity requests, return best answer."""
        best_answer = ""
        best_confidence = 0.0
        all_evidence: list[EvidenceEntry] = []
        total_tokens = 0

        for entity_req in entity_requests:
            chunks = await self._systematic_retrieve(entity_req.get("queries", []))
            if not chunks:
                continue

            summary, tok = await self._summarize(entity_req, chunks, hop_question, original_question)
            total_tokens += tok

            answer = summary.get("answer", "")
            confidence = summary.get("confidence", 0.0)
            supporting = summary.get("supporting_chunk_ids", [])

            for cid in supporting[:5]:
                chunk = chunks.get(cid)
                if chunk:
                    all_evidence.append(EvidenceEntry(
                        id="",
                        sub_question_id=sq_id,
                        content=chunk["text"][:2000],
                        source_chunk_id=cid,
                        relevance_score=confidence,
                        retriever_id=self.agent_id,
                    ))

            if confidence > best_confidence and answer:
                best_confidence = confidence
                best_answer = answer

        return best_answer, best_confidence, all_evidence, total_tokens

    def _build_retry_queries(
        self,
        hop_question: str,
        best_answer: str,
        entity_requests: list[dict[str, Any]],
    ) -> list[str]:
        """Build targeted retry queries using partial answer + goal."""
        queries = [hop_question[:80]]
        for er in entity_requests:
            entity = er.get("entity", "")
            goal = er.get("goal", "")
            if goal:
                queries.append(goal[:60])
            if entity:
                queries.append(entity)
        if best_answer:
            for er in entity_requests:
                entity = er.get("entity", "")
                if entity:
                    queries.append(f"{entity} {best_answer}"[:60])

        return dedupe_keep_order(queries)

    # ── Helpers ──────────────────────────────────────────────────────

    def _clean_structured_answer(self, answer: str) -> str:
        return clean_answer(answer)
