"""Investigator agent: programmatic structured retrieval for single-hop investigation.

Uses SAGE v3r2's proven retrieval pattern (systematic multi-query keyword+semantic
search with LLM summarization) instead of unreliable ReAct tool-calling loops.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from arag.core.context import AgentContext
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from arag.tools.search_and_read import SearchAndReadTool
from multi_agent.blackboard import Blackboard, Hop

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_REFUSAL_PATTERNS = [
    "cannot be determined", "insufficient information", "not mentioned",
    "no evidence", "unable to determine", "not enough information", "unknown",
]

_ENTITY_ID_PROMPT = """Extract search entities and queries for this retrieval task.

Task: {hop_question}
Original question: {original_question}
Context from other agents: {blackboard_context}

Generate entities to search for. Each entity should have 3-5 SHORT search queries.
- Use resolved names from context rather than pronouns or descriptions when possible.
- Queries should be evidence-seeking and faithful to the task.
- Prefer generic reformulations, aliases, and concise property hints over dataset-specific templates.

Output JSON only:
{{"entities": [{{"entity": "name", "goal": "what to find", "queries": ["q1", "q2", "q3"]}}]}}"""

_SUMMARIZER_PROMPT = """Extract facts from these documents for the given task.

Task: {hop_question}
Original question: {original_question}
Entity: {entity}
Goal: {goal}

Documents:
{documents}

Instructions:
- Extract only facts supported by the documents.
- Preserve exact names, dates, numbers VERBATIM from the documents.
- Include the most relevant facts for the goal.
- The "answer" field MUST directly answer the GOAL, not just name the entity.
- The "answer" field MUST be a concise entity/value (1-6 words), NOT a full sentence.
- Use the FULL CANONICAL NAME as it appears in the documents (e.g., "James Thomas Harrison" not "Jim Harrison").
- If the documents do not support a confident answer, set "answer" to "" and "confidence" to 0.0.



Output JSON only:
{{"entity": "entity name", "answer": "concise answer to the goal (1-6 words)", "facts": ["fact1", "fact2"], "confidence": 0.0, "supporting_chunk_ids": ["id1"]}}"""


def _strip_llm_wrappers(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _loads_json_with_repair(raw: str) -> dict[str, Any]:
    text = _strip_llm_wrappers(raw or "")
    if not text:
        raise ValueError("empty JSON payload")
    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

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


def _augment_queries_for_goal(entity: str, goal: str, base_queries: list[str]) -> list[str]:
    augmented = list(base_queries)
    name = entity.strip()
    goal_words = [
        word for word in re.findall(r"[A-Za-z0-9]+", (goal or "").lower())
        if len(word) > 2
    ]
    stop_words = {
        "the", "and", "for", "with", "from", "that", "this", "what", "which",
        "when", "where", "who", "whose", "how", "many", "much", "into", "onto",
        "about", "after", "before", "during", "between", "under", "over",
    }
    content_words = [word for word in goal_words if word not in stop_words]
    goal_hint = " ".join(content_words[:4]).strip()

    if name and name.lower() not in {q.lower() for q in augmented}:
        augmented.append(name)

    if name and goal_hint:
        augmented.append(f"{name} {goal_hint}")

    if goal_hint:
        augmented.append(goal_hint)

    return _dedupe_keep_order(augmented)


def _extract_candidate_fact_from_chunks(
    chunks: list[dict[str, Any]], entity: str, goal: str = "",
) -> str:
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
            if 15 <= len(sent) <= 300:
                all_sentences.append(sent)

    if entity_lower:
        for sent in all_sentences:
            if entity_lower in sent.lower():
                return sent
    if entity_words:
        min_matches = min(2, len(entity_words))
        for sent in all_sentences:
            sent_lower = sent.lower()
            if sum(1 for w in entity_words if w in sent_lower) >= min_matches:
                return sent
    if goal_words:
        for sent in all_sentences:
            sent_lower = sent.lower()
            if sum(1 for w in goal_words if w in sent_lower) >= 2:
                return sent
    return all_sentences[0] if all_sentences else ""


class Investigator:
    """Programmatic structured investigator for a single hop.

    Uses SAGE v3r2's proven retrieval approach:
    1. LLM extracts entities + queries from hop question
    2. Systematic keyword + semantic search with multiple queries
    3. LLM summarizes retrieved chunks to extract answer + facts
    4. Writes results to blackboard
    """

    def __init__(
        self,
        llm_client: LLMClient,
        base_tools: ToolRegistry,
        blackboard: Blackboard,
        hop: Hop,
        agent_id: str,
        retrieval_top_k: int = 10,
        max_queries_per_entity: int = 6,
        max_doc_chars: int = 2000,
        retry_low_confidence: bool = False,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.blackboard = blackboard
        self.hop = hop
        self.agent_id = agent_id
        self.retrieval_top_k = retrieval_top_k
        self.max_queries_per_entity = max_queries_per_entity
        self.max_doc_chars = max_doc_chars
        self.retry_low_confidence = retry_low_confidence
        self.verbose = verbose

        # Build search-and-read tool from base tools
        self.read_tool = base_tools.get("read_chunk")
        self.search_tool = SearchAndReadTool(
            keyword_tool=base_tools.get("keyword_search"),
            semantic_tool=base_tools.get("semantic_search"),
            read_tool=self.read_tool,
        )

    async def _call_llm(self, prompt: str) -> str:
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        return _strip_llm_wrappers((response.get("message") or {}).get("content", ""))

    async def _extract_entities_and_queries(self) -> list[dict[str, Any]]:
        """Step 1: Use LLM to extract entities and queries from the hop question."""
        hop_q = self.hop.resolved_question or self.hop.question
        bb_context = self.blackboard.get_context_for_investigator(self.hop)

        prompt = _ENTITY_ID_PROMPT.replace("{hop_question}", hop_q).replace(
            "{original_question}", self.blackboard.question
        ).replace("{blackboard_context}", bb_context)

        raw = await self._call_llm(prompt)

        try:
            data = _loads_json_with_repair(raw)
        except ValueError:
            # Fallback: extract entity names from the question
            return self._fallback_entities()

        entities_raw = data.get("entities", [])
        if not isinstance(entities_raw, list) or not entities_raw:
            return self._fallback_entities()

        normalized = []
        for item in entities_raw[:3]:
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
            if entity:
                augmented = _augment_queries_for_goal(entity, goal, queries)
                normalized.append({
                    "entity": entity,
                    "goal": goal or f"Find facts about {entity}",
                    "queries": augmented[:self.max_queries_per_entity],
                })

        return normalized if normalized else self._fallback_entities()

    def _fallback_entities(self) -> list[dict[str, Any]]:
        """Generate fallback entity queries from the hop question."""
        hop_q = self.hop.resolved_question or self.hop.question

        # Extract capitalized phrases as entity names
        words = hop_q.replace("?", "").split()
        entities = []
        current = []
        stop_words = {"who", "what", "where", "when", "which", "how", "is", "are",
                       "was", "were", "the", "a", "an", "of", "in", "at", "by",
                       "for", "to", "from", "with", "on", "and", "or", "did", "do",
                       "does", "has", "have", "had"}
        for word in words:
            clean = word.strip("?,!.\"'()")
            if clean and clean[0].isupper() and clean.lower() not in stop_words:
                current.append(clean)
            else:
                if current:
                    entities.append(" ".join(current))
                    current = []
        if current:
            entities.append(" ".join(current))

        if not entities:
            entities = [hop_q[:50]]

        result = []
        for entity in entities[:2]:
            queries = _augment_queries_for_goal(entity, hop_q, [entity])
            result.append({
                "entity": entity,
                "goal": hop_q,
                "queries": queries[:self.max_queries_per_entity],
            })
        return result

    async def _systematic_retrieve(self, entity_request: dict[str, Any]) -> list[dict]:
        """Step 2: Systematic multi-query retrieval (keyword + semantic)."""
        context = AgentContext()
        chunks_by_id: dict[str, dict[str, str]] = {}
        queries = entity_request.get("queries", [])
        if not queries:
            queries = [entity_request.get("entity", "")]

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

        return list(chunks_by_id.values())

    async def _summarize(
        self, entity_request: dict[str, Any], chunks: list[dict],
    ) -> dict[str, Any]:
        """Step 3: LLM summarization of retrieved chunks."""
        hop_q = self.hop.resolved_question or self.hop.question

        doc_blocks = []
        chunk_ids = []
        total_chars = 0
        max_total_chars = 20000  # Cap total to ~5K tokens to stay within context
        for chunk in chunks:
            cid = str(chunk.get("id", ""))
            text = str(chunk.get("text", ""))[:self.max_doc_chars]
            if cid and text:
                if total_chars + len(text) > max_total_chars:
                    break
                chunk_ids.append(cid)
                doc_blocks.append(f"[Chunk {cid}]\n{text}")
                total_chars += len(text)

        prompt = _SUMMARIZER_PROMPT.replace("{hop_question}", hop_q).replace(
            "{original_question}", self.blackboard.question
        ).replace("{entity}", entity_request["entity"]).replace(
            "{goal}", entity_request["goal"]
        ).replace("{documents}", "\n\n".join(doc_blocks) if doc_blocks else "No retrieved documents.")

        raw = await self._call_llm(prompt)

        try:
            data = _loads_json_with_repair(raw)
            answer = str(data.get("answer", "")).strip()
            facts_raw = data.get("facts", [])
            if not isinstance(facts_raw, list):
                facts_raw = []
            facts = _dedupe_keep_order([str(x).strip() for x in facts_raw if str(x).strip()])
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            supporting = data.get("supporting_chunk_ids", [])
            if not isinstance(supporting, list):
                supporting = []
            supporting = _dedupe_keep_order([str(x) for x in supporting])
            if not supporting:
                supporting = chunk_ids[:5]

            # Preserve some evidence text if parsing succeeded but fact extraction came back empty.
            if not facts and chunks:
                fallback = _extract_candidate_fact_from_chunks(
                    chunks, entity_request["entity"], entity_request.get("goal", ""),
                )
                if fallback:
                    facts = [fallback]

            return {
                "answer": answer,
                "facts": facts,
                "confidence": confidence,
                "supporting_chunk_ids": supporting,
            }
        except (ValueError, TypeError):
            # Preserve a candidate fact for debugging/retry, but do not invent an answer.
            fallback = _extract_candidate_fact_from_chunks(
                chunks, entity_request["entity"], entity_request.get("goal", ""),
            )
            return {
                "answer": "",
                "facts": [fallback] if fallback else [],
                "confidence": 0.0,
                "supporting_chunk_ids": chunk_ids[:5],
            }

    async def run(self) -> dict[str, Any]:
        """Run the full investigation for this hop."""
        self.hop.status = "investigating"
        self.hop.assigned_to = self.agent_id
        hop_q = self.hop.resolved_question or self.hop.question

        if self.verbose:
            logger.info("[%s] Investigating hop %d: %s", self.agent_id, self.hop.id, hop_q)

        # Step 1: Extract entities and queries
        entity_requests = await self._extract_entities_and_queries()

        # Step 2+3: Retrieve and summarize for each entity
        best_answer = ""
        best_confidence = 0.0
        all_facts: list[str] = []
        all_evidence: list[dict] = []
        all_chunk_ids: list[str] = []
        total_chunks_read = 0

        for entity_req in entity_requests:
            chunks = await self._systematic_retrieve(entity_req)
            total_chunks_read += len(chunks)

            if not chunks:
                continue

            summary = await self._summarize(entity_req, chunks)

            answer = summary.get("answer", "")
            facts = summary.get("facts", [])
            confidence = summary.get("confidence", 0.0)
            supporting = summary.get("supporting_chunk_ids", [])

            all_facts.extend(facts)
            all_chunk_ids.extend(supporting)

            for cid in supporting[:3]:
                text = ""
                if self.read_tool and hasattr(self.read_tool, "chunks_dict"):
                    text = self.read_tool.chunks_dict.get(str(cid), "")
                all_evidence.append({
                    "id": cid, "text": text[:500], "source_agent": self.agent_id,
                })

            # Add to entity KB
            if facts:
                self.blackboard.add_entity(
                    name=entity_req["entity"],
                    facts=facts,
                    confidence=confidence,
                    chunks=supporting,
                    agent_id=self.agent_id,
                )

            if confidence > best_confidence and answer:
                best_confidence = confidence
                best_answer = answer

        # Retry low-confidence entities with alternative query strategies
        if self.retry_low_confidence and best_confidence < 0.6 and entity_requests:
            for entity_req in entity_requests:
                entity = entity_req.get("entity", "")
                goal = entity_req.get("goal", "")
                # Generate alternative queries
                alt_queries = []
                if entity:
                    alt_queries.append(entity)
                    alt_queries.append(hop_q[:80])
                    goal_words = [w for w in re.findall(r"[A-Za-z0-9]+", goal) if len(w) > 3]
                    if goal_words:
                        alt_queries.append(f"{entity} {' '.join(goal_words[:3])}")
                        alt_queries.append(" ".join(goal_words[:4]))

                if not alt_queries:
                    continue

                alt_request = {
                    "entity": entity,
                    "goal": goal,
                    "queries": alt_queries[:4],
                }
                retry_chunks = await self._systematic_retrieve(alt_request)
                if not retry_chunks:
                    continue

                # Only use chunks not already seen
                seen_ids = {ev.get("id") for ev in all_evidence}
                new_chunks = [c for c in retry_chunks if c.get("id") not in seen_ids]
                if not new_chunks:
                    continue

                retry_summary = await self._summarize(alt_request, new_chunks)
                retry_answer = retry_summary.get("answer", "")
                retry_confidence = retry_summary.get("confidence", 0.0)
                retry_facts = retry_summary.get("facts", [])
                retry_supporting = retry_summary.get("supporting_chunk_ids", [])

                all_facts.extend(retry_facts)
                all_chunk_ids.extend(retry_supporting)

                for cid in retry_supporting[:3]:
                    text = ""
                    if self.read_tool and hasattr(self.read_tool, "chunks_dict"):
                        text = self.read_tool.chunks_dict.get(str(cid), "")
                    all_evidence.append({
                        "id": cid, "text": text[:500], "source_agent": self.agent_id,
                    })

                if retry_facts:
                    self.blackboard.add_entity(
                        name=entity,
                        facts=retry_facts,
                        confidence=retry_confidence,
                        chunks=retry_supporting,
                        agent_id=self.agent_id,
                    )

                if retry_confidence > best_confidence and retry_answer:
                    best_confidence = retry_confidence
                    best_answer = retry_answer

        # Retry with alternative queries if nothing was retrieved at all.
        if not best_answer and total_chunks_read == 0:
            alt_queries = [hop_q]
            for q in alt_queries:
                context = AgentContext()
                _, log = await asyncio.to_thread(
                    self.search_tool.execute, context, q, "semantic", self.retrieval_top_k,
                )
                for cid in log.get("chunk_ids_read", []) or []:
                    text = ""
                    if self.read_tool and hasattr(self.read_tool, "chunks_dict"):
                        text = self.read_tool.chunks_dict.get(str(cid), "")
                    if text:
                        all_evidence.append({
                            "id": cid, "text": text[:500], "source_agent": self.agent_id,
                        })
                if all_evidence:
                    break

        # Write to blackboard
        self.blackboard.resolve_hop(
            self.hop.id,
            answer=best_answer,
            evidence=all_evidence,
            confidence=best_confidence,
        )

        if self.verbose:
            logger.info(
                "[%s] Hop %d result: %s (conf=%.2f, chunks=%d)",
                self.agent_id, self.hop.id,
                best_answer[:60], best_confidence, total_chunks_read,
            )

        return {
            "answer": best_answer,
            "trajectory": [],
            "total_cost": 0.0,
            "loops": len(entity_requests),
            "confidence": best_confidence,
            "supporting_chunk_ids": _dedupe_keep_order(all_chunk_ids),
            "total_retrieved_tokens": 0,
        }
