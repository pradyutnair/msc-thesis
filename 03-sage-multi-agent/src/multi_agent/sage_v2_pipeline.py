"""SAGE v2 pipeline: iterative Reason -> Retrieve -> Summarize -> Answer."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from arag.core.config import Config
from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from arag.tools.search_and_read import SearchAndReadTool
from multi_agent.types import AgentResult, DecompositionPlan, PipelineResult, SubQuestion

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_REASONER_PROMPT = _PROMPTS_DIR / "sage_v2_reasoner.txt"
_DEFAULT_SUMMARIZER_PROMPT = _PROMPTS_DIR / "sage_v2_summarizer.txt"
_DEFAULT_ANSWER_PROMPT = _PROMPTS_DIR / "sage_v2_answer.txt"

_REFUSAL_PATTERNS = [
    "cannot be determined",
    "insufficient information",
    "not mentioned",
    "no evidence",
    "unable to determine",
    "not enough information",
    "unknown",
]


def _strip_llm_wrappers(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _is_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(p in lowered for p in _REFUSAL_PATTERNS)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _infer_expected_answer_type(question: str) -> str:
    q = (question or "").strip().lower()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had)\b", q):
        return "yes_no"
    if q.startswith("when ") or "what year" in q or "what date" in q or "birthdate" in q:
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith("who "):
        return "person"
    if q.startswith("how many ") or q.startswith("how much "):
        return "number"
    return "entity"


def _normalize_final_answer(answer: str, expected_answer_type: str) -> str:
    text = (answer or "").strip()
    text = re.sub(r"^\s*(final answer\s*:|answer\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[\.,;:!?]+$", "", text)
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"^(the answer is|answer is)\s+", "", text, flags=re.IGNORECASE).strip()

    if expected_answer_type == "yes_no":
        lowered = text.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8]).strip()
    return text


def _loads_json_with_repair(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = (raw or "").replace("\\'", "'")
        repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", repaired)
        return json.loads(repaired)


class SageV2Pipeline:
    """Iterative dual-process QA pipeline.

    Flow:
      1) Reasoner decides if answer is ready or what to retrieve next.
      2) Retriever executes entity-focused search_and_read.
      3) Summarizer condenses per-entity evidence into Knowledge Outline.
      4) Repeat until sufficient, then generate final concise answer.
    """

    def __init__(self, llm_client: LLMClient, tools: ToolRegistry, config: Config | None = None):
        self.llm = llm_client
        self.base_tools = tools
        self.config = config or Config()

        ma_cfg = self.config.get("multi_agent", {}) or {}
        self.max_iterations = int(ma_cfg.get("sage_v2_max_iterations", 5))
        self.max_entities_per_iteration = int(ma_cfg.get("sage_v2_max_entities_per_iteration", 3))
        self.max_queries_per_entity = int(ma_cfg.get("sage_v2_max_queries_per_entity", 3))
        self.retrieval_top_k = int(ma_cfg.get("sage_v2_retrieval_top_k", 3))
        self.max_doc_chars = int(ma_cfg.get("sage_v2_max_doc_chars", 3000))
        self.verbose = bool(ma_cfg.get("verbose", False))

        self.reasoner_prompt = Path(
            ma_cfg.get("sage_v2_reasoner_prompt", str(_DEFAULT_REASONER_PROMPT)),
        ).read_text(encoding="utf-8")
        self.summarizer_prompt = Path(
            ma_cfg.get("sage_v2_summarizer_prompt", str(_DEFAULT_SUMMARIZER_PROMPT)),
        ).read_text(encoding="utf-8")
        self.answer_prompt = Path(
            ma_cfg.get("sage_v2_answer_prompt", str(_DEFAULT_ANSWER_PROMPT)),
        ).read_text(encoding="utf-8")

        self.read_tool = self.base_tools.get("read_chunk")
        self.search_tool = SearchAndReadTool(
            keyword_tool=self.base_tools.get("keyword_search"),
            semantic_tool=self.base_tools.get("semantic_search"),
            read_tool=self.read_tool,
        )

    @staticmethod
    def _render_prompt(template: str, **kwargs: Any) -> str:
        rendered = template
        for key, value in kwargs.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered

    async def _reason(
        self,
        question: str,
        knowledge_outline: dict[str, Any],
        reasoning_history: list[str],
        iteration: int,
    ) -> dict[str, Any]:
        prompt = self._render_prompt(
            self.reasoner_prompt,
            question=question,
            knowledge_outline=json.dumps(knowledge_outline, ensure_ascii=False, indent=2),
            reasoning_history=json.dumps(reasoning_history, ensure_ascii=False, indent=2),
            iteration=iteration,
            max_iterations=self.max_iterations,
        )
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        raw = _strip_llm_wrappers((response.get("message") or {}).get("content", ""))
        data = _loads_json_with_repair(raw)

        mode = data.get("mode", "retrieve")
        if mode not in {"answer", "retrieve"}:
            mode = "retrieve"
        reasoning = str(data.get("reasoning", "")).strip()
        answer = str(data.get("answer", "")).strip()
        expected_type = str(data.get("expected_answer_type", "entity")).strip() or "entity"
        missing = data.get("missing", [])
        if not isinstance(missing, list):
            missing = []

        normalized_missing = []
        for item in missing[: self.max_entities_per_iteration]:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            goal = str(item.get("goal", "")).strip()
            method = str(item.get("method", "keyword")).strip().lower()
            if method not in {"keyword", "semantic"}:
                method = "keyword"
            queries_raw = item.get("queries", [])
            if not isinstance(queries_raw, list):
                queries_raw = []
            queries = _dedupe_keep_order([str(q).strip() for q in queries_raw if str(q).strip()])
            if not queries and entity:
                queries = [entity]
            if entity:
                normalized_missing.append(
                    {
                        "entity": entity,
                        "goal": goal or f"Find facts about {entity}",
                        "queries": queries[: self.max_queries_per_entity],
                        "method": method,
                    },
                )

        if mode == "answer" and not answer:
            mode = "retrieve"
        if mode == "retrieve" and not normalized_missing:
            normalized_missing = [
                {
                    "entity": "question_focus",
                    "goal": "Retrieve missing evidence for the question",
                    "queries": [question],
                    "method": "semantic",
                },
            ]

        return {
            "mode": mode,
            "reasoning": reasoning,
            "answer": answer,
            "expected_answer_type": expected_type,
            "missing": normalized_missing,
        }

    async def _retrieve_one_entity(self, entity_request: dict[str, Any]) -> tuple[list[dict], AgentContext]:
        context = AgentContext()
        chunks_by_id: dict[str, dict[str, str]] = {}
        method = entity_request["method"]
        for query in entity_request["queries"][: self.max_queries_per_entity]:
            _, log = await asyncio.to_thread(
                self.search_tool.execute,
                context,
                query,
                method,
                self.retrieval_top_k,
            )
            for chunk_id in log.get("chunk_ids_read", []) or []:
                cid = str(chunk_id)
                text = ""
                if self.read_tool is not None and hasattr(self.read_tool, "chunks_dict"):
                    text = str(self.read_tool.chunks_dict.get(cid, ""))
                if text:
                    chunks_by_id[cid] = {"id": cid, "text": text}

        if not chunks_by_id:
            alt_method = "semantic" if method == "keyword" else "keyword"
            fallback_queries = [entity_request["entity"]] + entity_request["queries"][:1]
            for query in _dedupe_keep_order([q for q in fallback_queries if q]):
                _, log = await asyncio.to_thread(
                    self.search_tool.execute,
                    context,
                    query,
                    alt_method,
                    self.retrieval_top_k,
                )
                for chunk_id in log.get("chunk_ids_read", []) or []:
                    cid = str(chunk_id)
                    text = ""
                    if self.read_tool is not None and hasattr(self.read_tool, "chunks_dict"):
                        text = str(self.read_tool.chunks_dict.get(cid, ""))
                    if text:
                        chunks_by_id[cid] = {"id": cid, "text": text}

        return list(chunks_by_id.values()), context

    async def _summarize_entity(
        self,
        question: str,
        reasoning: str,
        entity_request: dict[str, Any],
        chunks: list[dict],
    ) -> dict[str, Any]:
        doc_blocks = []
        chunk_ids = []
        for chunk in chunks:
            cid = str(chunk.get("id", ""))
            text = str(chunk.get("text", ""))[: self.max_doc_chars]
            if not cid or not text:
                continue
            chunk_ids.append(cid)
            doc_blocks.append(f"[Chunk {cid}]\n{text}")

        prompt = self._render_prompt(
            self.summarizer_prompt,
            question=question,
            reasoning=reasoning or "No prior reasoning yet.",
            entity=entity_request["entity"],
            goal=entity_request["goal"],
            queries=json.dumps(entity_request["queries"], ensure_ascii=False),
            documents="\n\n".join(doc_blocks) if doc_blocks else "No retrieved documents.",
        )
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        raw = _strip_llm_wrappers((response.get("message") or {}).get("content", ""))

        try:
            data = _loads_json_with_repair(raw)
            facts_raw = data.get("facts", [])
            if not isinstance(facts_raw, list):
                facts_raw = []
            facts = _dedupe_keep_order([str(x).strip() for x in facts_raw if str(x).strip()])
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            supporting_chunk_ids = data.get("supporting_chunk_ids", [])
            if not isinstance(supporting_chunk_ids, list):
                supporting_chunk_ids = []
            supporting_chunk_ids = _dedupe_keep_order([str(x) for x in supporting_chunk_ids])
            if not supporting_chunk_ids:
                supporting_chunk_ids = _dedupe_keep_order(chunk_ids)
            return {
                "entity": str(data.get("entity", entity_request["entity"])).strip() or entity_request["entity"],
                "facts": facts,
                "confidence": confidence,
                "supporting_chunk_ids": supporting_chunk_ids,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            fallback_facts = []
            if chunks:
                first = chunks[0].get("text", "")
                sentence = re.split(r"[.!?\n]+", str(first))[0].strip()
                if sentence:
                    fallback_facts.append(sentence)
            return {
                "entity": entity_request["entity"],
                "facts": fallback_facts,
                "confidence": 0.2 if fallback_facts else 0.0,
                "supporting_chunk_ids": _dedupe_keep_order(chunk_ids),
            }

    async def _generate_final_answer(
        self,
        question: str,
        expected_answer_type: str,
        reasoning: str,
        knowledge_outline: dict[str, Any],
    ) -> tuple[str, float]:
        prompt = self._render_prompt(
            self.answer_prompt,
            question=question,
            expected_answer_type=expected_answer_type,
            reasoning=reasoning or "",
            knowledge_outline=json.dumps(knowledge_outline, ensure_ascii=False, indent=2),
        )
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        raw = ((response.get("message") or {}).get("content", "") or "").strip()
        cleaned = _strip_llm_wrappers(raw)
        match = re.search(r"FINAL ANSWER:\s*(.+)$", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        answer = (match.group(1).strip() if match else cleaned).strip()
        if "\n" in answer:
            answer = answer.splitlines()[0].strip()
        return answer, float(response.get("cost", 0.0) or 0.0)

    @staticmethod
    def _build_decomposition(question: str, missing_items: list[dict[str, Any]]) -> DecompositionPlan:
        sub_questions = []
        for idx, item in enumerate(missing_items):
            sub_questions.append(
                SubQuestion(
                    index=idx,
                    text=item.get("goal", "") or item.get("entity", ""),
                    search_hints=list(item.get("queries", [])),
                    depends_on=[],
                ),
            )
        question_type = "comparison" if len(sub_questions) > 1 else "single_hop"
        return DecompositionPlan(
            question_type=question_type,
            sub_questions=sub_questions,
            dependency_edges=[],
            raw_llm_output=question,
            parse_retries=0,
        )

    async def run(self, question: str) -> PipelineResult:
        start = time.monotonic()
        result = PipelineResult(question=question)
        reasoning_history: list[str] = []
        knowledge_outline: dict[str, Any] = {}
        expected_answer_type = "entity"
        total_retrieval_tokens = 0
        total_iterations = 0
        task_idx = 0
        last_missing: list[dict[str, Any]] = []
        final_answer = ""
        final_reasoning = ""
        synthesis_cost = 0.0

        try:
            for iteration in range(1, self.max_iterations + 1):
                total_iterations = iteration
                decision = await self._reason(
                    question=question,
                    knowledge_outline=knowledge_outline,
                    reasoning_history=reasoning_history,
                    iteration=iteration,
                )
                expected_answer_type = decision["expected_answer_type"] or expected_answer_type
                if expected_answer_type == "entity":
                    expected_answer_type = _infer_expected_answer_type(question)
                reasoning = decision.get("reasoning", "")
                if reasoning:
                    reasoning_history.append(reasoning)
                final_reasoning = reasoning

                if decision["mode"] == "answer":
                    if _is_refusal(decision["answer"]):
                        decision["mode"] = "retrieve"
                        decision["missing"] = [
                            {
                                "entity": "question_focus",
                                "goal": "Retrieve missing evidence for final answer",
                                "queries": [question],
                                "method": "semantic",
                            },
                        ]
                    else:
                        final_answer = _normalize_final_answer(
                            decision["answer"].strip(),
                            expected_answer_type,
                        )
                        break

                missing = decision["missing"]
                last_missing = missing
                if not missing:
                    break

                coros = [self._retrieve_one_entity(item) for item in missing]
                retrieval_results = await asyncio.gather(*coros, return_exceptions=True)
                added_facts = 0

                for item, retrieved in zip(missing, retrieval_results):
                    if isinstance(retrieved, Exception):
                        logger.warning("Entity retrieval failed for %s: %s", item.get("entity"), retrieved)
                        continue
                    chunks, context = retrieved
                    total_retrieval_tokens += int(context.total_retrieved_tokens)
                    summary = await self._summarize_entity(
                        question=question,
                        reasoning=reasoning,
                        entity_request=item,
                        chunks=chunks,
                    )
                    entity_name = summary["entity"]
                    entry = knowledge_outline.setdefault(
                        entity_name,
                        {"facts": [], "supporting_chunk_ids": [], "confidence": 0.0},
                    )
                    before = len(entry["facts"])
                    entry["facts"] = _dedupe_keep_order(entry["facts"] + summary["facts"])
                    added_facts += max(0, len(entry["facts"]) - before)
                    entry["supporting_chunk_ids"] = _dedupe_keep_order(
                        entry["supporting_chunk_ids"] + summary["supporting_chunk_ids"],
                    )
                    entry["confidence"] = max(float(entry["confidence"]), float(summary["confidence"]))

                    result.agent_results[task_idx] = AgentResult(
                        sub_question_index=task_idx,
                        answer=summary["facts"][0] if summary["facts"] else "",
                        evidence_doc_ids=summary["supporting_chunk_ids"],
                        loops=len(item.get("queries", [])),
                        total_tokens=int(context.total_retrieved_tokens),
                        confidence=float(summary["confidence"]),
                        retrieved_chunks=chunks,
                        extracted_evidence=summary["facts"],
                        evidence_count=len(summary["supporting_chunk_ids"]),
                    )
                    task_idx += 1

                # Avoid spinning when retrieval adds no new knowledge.
                if added_facts == 0:
                    break

            if not final_answer or _is_refusal(final_answer):
                final_answer, synthesis_cost = await self._generate_final_answer(
                    question=question,
                    expected_answer_type=expected_answer_type,
                    reasoning=final_reasoning or " ".join(reasoning_history[-2:]),
                    knowledge_outline=knowledge_outline,
                )
                final_answer = _normalize_final_answer(final_answer, expected_answer_type)

            if not final_answer or _is_refusal(final_answer):
                # Last-resort fallback: first fact of most confident entity.
                best_entity = None
                best_conf = -1.0
                for entity, payload in knowledge_outline.items():
                    conf = float(payload.get("confidence", 0.0))
                    facts = [f for f in payload.get("facts", []) if not _is_refusal(str(f))]
                    if conf > best_conf and facts:
                        best_conf = conf
                        best_entity = entity
                if best_entity:
                    final_answer = _normalize_final_answer(
                        knowledge_outline[best_entity]["facts"][0],
                        expected_answer_type,
                    )

            result.final_answer = final_answer.strip()
            result.total_tokens = total_retrieval_tokens
            result.aggregator_tokens = int(synthesis_cost * 1_000_000) if synthesis_cost > 0 else 0
            result.question_type = "comparison" if len(last_missing) > 1 else "bridge"
            result.num_sub_questions = len(result.agent_results)
            result.num_waves = total_iterations
            result.decomposition = self._build_decomposition(question, last_missing)
            result.verifier_parse_ok = True
            result.cache_analytics = {
                "knowledge_outline_entities": len(knowledge_outline),
                "knowledge_outline": knowledge_outline,
            }

        except Exception as exc:
            logger.error("SAGE v2 pipeline error for '%s': %s", question[:80], exc)
            result.error = str(exc)

        result.wall_clock_seconds = time.monotonic() - start
        return result
