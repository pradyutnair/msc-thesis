"""SAGE v2 pipeline: Reason -> Entity Identify -> Retrieve -> Summarize -> Answer."""

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
_DEFAULT_ENTITY_IDENTIFIER_PROMPT = _PROMPTS_DIR / "sage_v2_entity_identifier.txt"
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

# Attribute keywords for automatic query augmentation
_BIRTH_KEYWORDS = {"birthplace", "born", "birth", "native", "hometown"}
_DEATH_KEYWORDS = {"death", "died", "death place", "buried", "grave"}
_FOUNDING_KEYWORDS = {"founded", "established", "created", "formed", "inception", "origin"}
_BATTLE_KEYWORDS = {"battle", "war", "conflict", "siege", "fought"}
_ABOLISH_KEYWORDS = {"abolished", "dissolved", "ended", "terminated", "ceased"}
_LOCATION_KEYWORDS = {"located", "headquarters", "based", "situated", "capital", "city"}


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
    text = re.sub(r"^(the answer is|answer is)\s+", "", text, flags=re.IGNORECASE).strip()
    if expected_answer_type == "yes_no":
        lowered = text.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
    return text


def _loads_json_with_repair(raw: str) -> dict[str, Any]:
    text = _strip_llm_wrappers(raw or "")
    if not text:
        raise ValueError("empty JSON payload")
    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        cleaned = re.sub(r"[\x00-\x1f\x7f]", "", candidate).strip()
        if not cleaned:
            continue
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = cleaned.replace("\\'", "'")
            repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", repaired)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    raise ValueError("unable to parse JSON")


def _augment_queries_for_goal(entity: str, goal: str, base_queries: list[str]) -> list[str]:
    """Add attribute-specific query variants based on the retrieval goal.

    This is the KEY fix for multi-hop chain propagation: when searching for
    "Stanton Moore birthplace", BM25 won't match documents that say "born in
    New Orleans". So we auto-add variants like "Stanton Moore born",
    "Stanton Moore early life", etc.
    """
    augmented = list(base_queries)
    name = entity.strip()
    goal_lower = (goal or "").lower()
    goal_words = set(goal_lower.split())

    # Always add the bare entity name — surfaces its Wikipedia article
    if name and name.lower() not in {q.lower() for q in augmented}:
        augmented.append(name)

    # Birthplace / born
    if goal_words & _BIRTH_KEYWORDS:
        augmented.extend([
            f"{name} born",
            f"{name} early life biography",
        ])

    # Death / died
    if goal_words & _DEATH_KEYWORDS:
        augmented.extend([
            f"{name} died death",
            f"{name} obituary",
        ])

    # Founded / established / created
    if goal_words & _FOUNDING_KEYWORDS:
        augmented.extend([
            f"{name} history founded",
            f"{name} established origin",
        ])

    # Battle / war
    if goal_words & _BATTLE_KEYWORDS:
        augmented.extend([
            f"battle of {name}",
            f"{name} military history war",
        ])

    # Abolished / dissolved
    if goal_words & _ABOLISH_KEYWORDS:
        augmented.extend([
            f"{name} abolished dissolved history",
            f"{name} ended",
        ])

    # Location / headquarters
    if goal_words & _LOCATION_KEYWORDS:
        augmented.extend([
            f"{name} location headquarters",
            f"{name} based city",
        ])

    return _dedupe_keep_order(augmented)


def _extract_candidate_fact_from_chunks(
    chunks: list[dict[str, Any]],
    entity: str,
    goal: str = "",
) -> str:
    """Extract a candidate fact from chunks with multi-level fallback.

    Tries:
    1. Exact entity name match in sentence
    2. Individual words of entity name match
    3. Goal keyword match
    4. First meaningful sentence of any chunk
    """
    entity_lower = (entity or "").strip().lower()
    entity_words = [w for w in entity_lower.split() if len(w) > 2]
    goal_lower = (goal or "").strip().lower()
    goal_words = [w for w in goal_lower.split() if len(w) > 3]

    all_sentences: list[str] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        for sentence in re.split(r"[.!?\n]+", text):
            sent = sentence.strip()
            if len(sent) < 15 or len(sent) > 300:
                continue
            all_sentences.append(sent)

    # Level 1: exact entity match
    if entity_lower:
        for sent in all_sentences:
            if entity_lower in sent.lower():
                return sent

    # Level 2: entity word match (at least 2 words or all words if entity is 1 word)
    if entity_words:
        min_matches = min(2, len(entity_words))
        for sent in all_sentences:
            sent_lower = sent.lower()
            matches = sum(1 for w in entity_words if w in sent_lower)
            if matches >= min_matches:
                return sent

    # Level 3: goal keyword match
    if goal_words:
        for sent in all_sentences:
            sent_lower = sent.lower()
            matches = sum(1 for w in goal_words if w in sent_lower)
            if matches >= 2:
                return sent

    # Level 4: first meaningful sentence from any chunk
    if all_sentences:
        return all_sentences[0]

    return ""


class SageV2Pipeline:
    """Iterative dual-process QA pipeline with explicit agent coordination."""

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
        self.entity_identifier_prompt = Path(
            ma_cfg.get("sage_v2_entity_identifier_prompt", str(_DEFAULT_ENTITY_IDENTIFIER_PROMPT)),
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
        try:
            data = _loads_json_with_repair(raw)
        except ValueError:
            return {
                "mode": "retrieve",
                "reasoning": "",
                "answer": "",
                "expected_answer_type": _infer_expected_answer_type(question),
                "knowledge_gaps": [f"Missing evidence to answer: {question}"],
            }

        mode = str(data.get("mode", "retrieve")).strip().lower()
        if mode not in {"answer", "retrieve"}:
            mode = "retrieve"
        reasoning = str(data.get("reasoning", "")).strip()
        answer = str(data.get("answer", "")).strip()
        expected_type = str(data.get("expected_answer_type", "entity")).strip() or "entity"
        knowledge_gaps_raw = data.get("knowledge_gaps", [])
        if not isinstance(knowledge_gaps_raw, list):
            knowledge_gaps_raw = []
        knowledge_gaps = _dedupe_keep_order([str(g).strip() for g in knowledge_gaps_raw if str(g).strip()])
        if mode == "answer" and not answer:
            mode = "retrieve"
        if mode == "retrieve" and not knowledge_gaps:
            knowledge_gaps = [f"Missing evidence to answer: {question}"]
        return {
            "mode": mode,
            "reasoning": reasoning,
            "answer": answer,
            "expected_answer_type": expected_type,
            "knowledge_gaps": knowledge_gaps,
        }

    async def _identify_entities(
        self,
        question: str,
        reasoning: str,
        knowledge_outline: dict[str, Any],
        knowledge_gaps: list[str],
    ) -> list[dict[str, Any]]:
        prompt = self._render_prompt(
            self.entity_identifier_prompt,
            question=question,
            reasoning=reasoning or "No prior reasoning yet.",
            knowledge_outline=json.dumps(knowledge_outline, ensure_ascii=False, indent=2),
            knowledge_gaps=json.dumps(knowledge_gaps, ensure_ascii=False, indent=2),
        )
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        raw = _strip_llm_wrappers((response.get("message") or {}).get("content", ""))

        try:
            data = _loads_json_with_repair(raw)
        except ValueError:
            return [
                {
                    "entity": "question_focus",
                    "goal": knowledge_gaps[0] if knowledge_gaps else "Retrieve missing evidence",
                    "queries": [question],
                    "link_to": None,
                },
            ]

        entities_raw = data.get("entities", [])
        if not isinstance(entities_raw, list):
            entities_raw = data.get("missing", [])
        if not isinstance(entities_raw, list):
            entities_raw = []

        normalized: list[dict[str, Any]] = []
        for item in entities_raw[: self.max_entities_per_iteration]:
            if not isinstance(item, dict):
                continue
            entity = str(item.get("entity", "")).strip()
            goal = str(item.get("goal", "")).strip()
            queries_raw = item.get("queries", [])
            if not isinstance(queries_raw, list):
                queries_raw = []
            queries = _dedupe_keep_order([str(q).strip() for q in queries_raw if str(q).strip()])
            if not queries and entity:
                queries = [entity]
            link_to = item.get("link_to", None)
            if entity:
                # Augment queries with attribute-specific variants
                augmented = _augment_queries_for_goal(entity, goal, queries)
                normalized.append(
                    {
                        "entity": entity,
                        "goal": goal or f"Find missing facts for {entity}",
                        "queries": augmented[: self.max_queries_per_entity],
                        "link_to": str(link_to).strip() if link_to else None,
                    },
                )

        if not normalized:
            fallback_query = knowledge_gaps[0] if knowledge_gaps else question
            normalized = [
                {
                    "entity": "question_focus",
                    "goal": fallback_query,
                    "queries": [question, fallback_query][: self.max_queries_per_entity],
                    "link_to": None,
                },
            ]
        return normalized

    async def _retrieve_one_entity(self, entity_request: dict[str, Any]) -> tuple[list[dict], AgentContext]:
        context = AgentContext()
        chunks_by_id: dict[str, dict[str, str]] = {}
        base_queries = entity_request.get("queries", [])
        if not isinstance(base_queries, list):
            base_queries = []
        fallback = [str(entity_request.get("entity", "")).strip()]
        queries = _dedupe_keep_order([str(q).strip() for q in base_queries if str(q).strip()] + fallback)
        queries = queries[: self.max_queries_per_entity]

        for method in ("keyword", "semantic"):
            for query in queries:
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
            # Force at least 1 fact when chunks were retrieved but LLM returned 0
            if not facts and chunks:
                fallback_fact = _extract_candidate_fact_from_chunks(
                    chunks, entity_request["entity"], entity_request.get("goal", ""),
                )
                facts = [fallback_fact] if fallback_fact else []
                if facts and confidence < 0.2:
                    confidence = 0.2
            return {
                "entity": str(data.get("entity", entity_request["entity"])).strip() or entity_request["entity"],
                "facts": facts,
                "confidence": confidence,
                "supporting_chunk_ids": supporting_chunk_ids,
            }
        except (ValueError, TypeError):
            fallback_fact = _extract_candidate_fact_from_chunks(
                chunks, entity_request["entity"], entity_request.get("goal", ""),
            )
            fallback_facts = [fallback_fact] if fallback_fact else []
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

    @staticmethod
    def _find_last_hop_entity(
        question: str,
        knowledge_outline: dict[str, Any],
    ) -> str | None:
        """Try to identify the entity most relevant to the last hop of the question."""
        q_lower = question.lower()
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for",
                       "and", "or", "that", "this", "what", "which", "who", "where", "when",
                       "how", "did", "do", "does", "has", "have", "had"}
        q_words = [w for w in re.findall(r"\w+", q_lower) if w not in stop_words and len(w) > 2]

        best_entity = None
        best_score = -1.0

        for entity, payload in knowledge_outline.items():
            facts = [f for f in payload.get("facts", []) if not _is_refusal(str(f))]
            if not facts:
                continue
            conf = float(payload.get("confidence", 0.0))
            entity_text = (entity + " " + " ".join(facts)).lower()
            word_overlap = sum(1 for w in q_words if w in entity_text)
            score = conf * 0.3 + word_overlap * 0.7
            if score > best_score:
                best_score = score
                best_entity = entity

        return best_entity

    async def _retry_low_confidence_entities(
        self,
        question: str,
        reasoning: str,
        entities: list[dict[str, Any]],
        summaries: list[dict[str, Any]],
    ) -> list[tuple[dict[str, Any], list[dict], AgentContext]]:
        """Retry retrieval for entities where summarizer returned low confidence.

        Generates alternative query formulations to find facts that the initial
        queries missed (e.g., "born in" instead of "birthplace").
        """
        retries = []
        for item, summary in zip(entities, summaries):
            if summary["confidence"] > 0.3:
                continue
            if not summary["facts"] or all(_is_refusal(f) for f in summary["facts"]):
                entity = item["entity"]
                goal = item.get("goal", "")
                original_queries = set(q.lower() for q in item.get("queries", []))
                # Generate alternative queries NOT in the original set
                alt_queries = []
                for q in [
                    f"{entity} wikipedia",
                    f"who is {entity}",
                    f"what is {entity}",
                    f"{entity} information",
                    f"{entity} facts",
                ]:
                    if q.lower() not in original_queries:
                        alt_queries.append(q)
                if not alt_queries:
                    continue
                retry_request = {
                    "entity": entity,
                    "goal": goal,
                    "queries": alt_queries[:3],
                    "link_to": item.get("link_to"),
                }
                chunks, ctx = await self._retrieve_one_entity(retry_request)
                if chunks:
                    retry_summary = await self._summarize_entity(
                        question=question,
                        reasoning=reasoning,
                        entity_request=retry_request,
                        chunks=chunks,
                    )
                    retries.append((item, retry_summary, chunks, ctx))
        return retries

    async def run(self, question: str) -> PipelineResult:
        start = time.monotonic()
        result = PipelineResult(question=question)
        reasoning_history: list[str] = []
        knowledge_outline: dict[str, Any] = {}
        expected_answer_type = _infer_expected_answer_type(question)
        total_retrieval_tokens = 0
        total_iterations = 0
        task_idx = 0
        last_missing: list[dict[str, Any]] = []
        final_reasoning = ""
        synthesis_cost = 0.0
        consecutive_empty = 0

        try:
            for iteration in range(1, self.max_iterations + 1):
                total_iterations = iteration
                decision = await self._reason(
                    question=question,
                    knowledge_outline=knowledge_outline,
                    reasoning_history=reasoning_history,
                    iteration=iteration,
                )
                expected_answer_type = decision.get("expected_answer_type", expected_answer_type) or expected_answer_type
                if expected_answer_type == "entity":
                    expected_answer_type = _infer_expected_answer_type(question)
                reasoning = str(decision.get("reasoning", "")).strip()
                if reasoning:
                    reasoning_history.append(reasoning)
                final_reasoning = reasoning or final_reasoning

                if decision.get("mode") == "answer" and not _is_refusal(decision.get("answer", "")):
                    break

                entities_to_retrieve = await self._identify_entities(
                    question=question,
                    reasoning=reasoning,
                    knowledge_outline=knowledge_outline,
                    knowledge_gaps=list(decision.get("knowledge_gaps", [])),
                )
                last_missing = entities_to_retrieve
                if not entities_to_retrieve:
                    break

                coros = [self._retrieve_one_entity(item) for item in entities_to_retrieve]
                retrieval_results = await asyncio.gather(*coros, return_exceptions=True)
                added_facts = 0
                iteration_summaries = []

                for item, retrieved in zip(entities_to_retrieve, retrieval_results):
                    if isinstance(retrieved, Exception):
                        logger.warning("Entity retrieval failed for %s: %s", item.get("entity"), retrieved)
                        iteration_summaries.append({"entity": item["entity"], "facts": [], "confidence": 0.0, "supporting_chunk_ids": []})
                        continue
                    chunks, context = retrieved
                    total_retrieval_tokens += int(context.total_retrieved_tokens)
                    summary = await self._summarize_entity(
                        question=question,
                        reasoning=reasoning,
                        entity_request=item,
                        chunks=chunks,
                    )
                    iteration_summaries.append(summary)
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

                # Retry low-confidence entities with alternative queries
                retry_results = await self._retry_low_confidence_entities(
                    question=question,
                    reasoning=reasoning,
                    entities=entities_to_retrieve,
                    summaries=iteration_summaries,
                )
                for item, retry_summary, chunks, ctx in retry_results:
                    total_retrieval_tokens += int(ctx.total_retrieved_tokens)
                    entity_name = retry_summary["entity"]
                    entry = knowledge_outline.setdefault(
                        entity_name,
                        {"facts": [], "supporting_chunk_ids": [], "confidence": 0.0},
                    )
                    before = len(entry["facts"])
                    entry["facts"] = _dedupe_keep_order(entry["facts"] + retry_summary["facts"])
                    added_facts += max(0, len(entry["facts"]) - before)
                    entry["supporting_chunk_ids"] = _dedupe_keep_order(
                        entry["supporting_chunk_ids"] + retry_summary["supporting_chunk_ids"],
                    )
                    entry["confidence"] = max(float(entry["confidence"]), float(retry_summary["confidence"]))

                    result.agent_results[task_idx] = AgentResult(
                        sub_question_index=task_idx,
                        answer=retry_summary["facts"][0] if retry_summary["facts"] else "",
                        evidence_doc_ids=retry_summary["supporting_chunk_ids"],
                        loops=len(item.get("queries", [])),
                        total_tokens=int(ctx.total_retrieved_tokens),
                        confidence=float(retry_summary["confidence"]),
                        retrieved_chunks=chunks,
                        extracted_evidence=retry_summary["facts"],
                        evidence_count=len(retry_summary["supporting_chunk_ids"]),
                    )
                    task_idx += 1

                # Replace premature termination with consecutive-empty counter
                if added_facts == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        break
                else:
                    consecutive_empty = 0

            final_answer, synthesis_cost = await self._generate_final_answer(
                question=question,
                expected_answer_type=expected_answer_type,
                reasoning=final_reasoning or " ".join(reasoning_history[-2:]),
                knowledge_outline=knowledge_outline,
            )
            final_answer = _normalize_final_answer(final_answer, expected_answer_type)

            if not final_answer or _is_refusal(final_answer):
                # Try to find the entity most relevant to the last hop
                best_entity = self._find_last_hop_entity(question, knowledge_outline)
                if not best_entity:
                    # Fall back to highest-confidence entity
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