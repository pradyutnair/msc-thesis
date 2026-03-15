"""Escalation Pipeline: structured-first with learned escalation to agentic.

For each sub-question in dependency order:
  1. Run StructuredWorker (cheap)
  2. Escalation policy decides ACCEPT or ESCALATE based on structured result
  3. If ESCALATE, run AgenticWorker (expensive)
  4. Use best available answer for downstream sub-questions

The escalation policy can be:
  - 'learned': LoRA-finetuned LLM (trained with GRPO on counterfactual data)
  - 'heuristic': rule-based (escalate if structured answer is empty/unknown)
  - 'always_structured': never escalate (ablation baseline)
  - 'always_agentic': always escalate (ablation baseline / upper bound)
  - 'oracle': use counterfactual data to make perfect decisions (upper bound)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.blackboard import Blackboard
from multi_agent.planner_agent import PlannerAgent
from multi_agent.synthesizer_agent import SynthesizerAgent
from multi_agent.types import (
    EntityEntry,
    M6PipelineResult,
    M6SubQuestion,
    RetrievalMode,
    SubQuestionStatus,
)
from multi_agent.utils import resolve_placeholders
from multi_agent.workers.agentic_worker import AgenticWorker
from multi_agent.workers.aggregate_worker import AggregateWorker
from multi_agent.workers.structured_worker import StructuredWorker

logger = logging.getLogger(__name__)

# Same prompt template used in training — must match exactly
ESCALATION_PROMPT_TEMPLATE = """\
You are an escalation policy for a multi-hop question answering system.
A structured retrieval worker has attempted to answer a sub-question.
Decide whether to ACCEPT the structured result or ESCALATE to a more expensive agentic retrieval worker.

Original question: {question}
Sub-question: {sq_text}

Structured retrieval result:
- Answer: "{structured_answer}"
- Evidence passages found: {evidence_count}

Previously resolved answers:
{entity_context}

Decision (ACCEPT or ESCALATE with brief reasoning):
"""


class EscalationPipeline:
    """Structured-first pipeline with learned escalation to agentic retrieval."""

    def __init__(
        self,
        llm_client: LLMClient,
        worker_llm_client: LLMClient,
        tools: ToolRegistry,
        escalation_mode: str = "learned",
        escalation_llm_client: LLMClient | None = None,
        decomposer_prompt: str | None = None,
        synthesizer_prompt: str | None = None,
        synthesizer_consistency_prompt: str | None = None,
        worker_max_steps: int = 16,
        token_budget: int = 300_000,
        structured_retrieval_top_k: int = 10,
        structured_max_queries: int = 6,
        structured_retry_low_confidence: bool = True,
        structured_confidence_threshold: float = 0.6,
        structured_timeout: float = 180.0,
        agentic_timeout: float = 300.0,
        decompose_timeout: float = 120.0,
    ):
        self.llm = llm_client
        self.worker_llm = worker_llm_client
        self.escalation_llm = escalation_llm_client or llm_client
        self.tools = tools
        self.escalation_mode = escalation_mode

        self.decomposer_prompt = decomposer_prompt
        self.synthesizer_prompt = synthesizer_prompt
        self.synthesizer_consistency_prompt = synthesizer_consistency_prompt

        self.worker_max_steps = worker_max_steps
        self.token_budget = token_budget
        self.structured_retrieval_top_k = structured_retrieval_top_k
        self.structured_max_queries = structured_max_queries
        self.structured_retry_low_confidence = structured_retry_low_confidence
        self.structured_confidence_threshold = structured_confidence_threshold

        self.structured_timeout = structured_timeout
        self.agentic_timeout = agentic_timeout
        self.decompose_timeout = decompose_timeout

    async def run(self, question: str) -> M6PipelineResult:
        t0 = time.monotonic()

        # Step 1: Decompose
        blackboard = Blackboard(question=question, token_budget=self.token_budget)
        planner = PlannerAgent(
            llm_client=self.llm,
            decompose_prompt_path=self.decomposer_prompt,
            max_redecompositions=0,
            decompose_temperature=0.0,
        )

        try:
            await asyncio.wait_for(
                planner.decompose_first(blackboard), timeout=self.decompose_timeout,
            )
        except (asyncio.TimeoutError, Exception) as exc:
            logger.error("Decomposition failed: %s", exc)
            return M6PipelineResult(
                question=question,
                pred_answer=f"Error: decomposition failed: {exc}",
                error=str(exc),
                wall_clock_seconds=time.monotonic() - t0,
            )

        sub_questions = blackboard.search_plan
        if not sub_questions:
            return M6PipelineResult(
                question=question,
                pred_answer="Error: empty decomposition",
                error="empty decomposition",
                wall_clock_seconds=time.monotonic() - t0,
            )

        # Step 2: Process SQs in dependency order with escalation
        entity_registry: dict[str, str] = {}
        sq_details: list[dict[str, Any]] = []
        total_tokens = 0
        escalation_decisions: list[str] = []
        mode_tokens: dict[str, int] = {"structured": 0, "agentic": 0, "aggregate": 0}

        ordered_sqs = self._topological_sort(sub_questions)

        for sq in ordered_sqs:
            resolved_text = resolve_placeholders(sq.text, entity_registry)

            # Aggregate SQs don't retrieve — handle separately
            if sq.mode == RetrievalMode.AGGREGATE:
                answer, tokens = await self._run_aggregate(
                    sq, question, entity_registry, blackboard,
                )
                total_tokens += tokens
                mode_tokens["aggregate"] += tokens
                if answer:
                    entity_registry[f"answer_{sq.id}"] = answer
                sq_details.append({
                    "id": sq.id, "text": sq.text, "mode": "aggregate",
                    "answer": answer, "tokens": tokens,
                    "escalation_decision": "N/A",
                    "status": "verified" if answer else "failed",
                })
                continue

            # Run structured first
            s_answer, s_tokens, s_evidence = await self._run_structured(
                sq, question, entity_registry,
            )
            total_tokens += s_tokens
            mode_tokens["structured"] += s_tokens

            # Escalation decision
            decision = await self._escalation_decision(
                question, resolved_text, s_answer, s_evidence, entity_registry,
            )
            escalation_decisions.append(decision)

            final_answer = s_answer
            a_tokens = 0

            if decision == "ESCALATE":
                a_answer, a_tokens, a_evidence = await self._run_agentic(
                    sq, question, entity_registry,
                )
                total_tokens += a_tokens
                mode_tokens["agentic"] += a_tokens
                # Use agentic answer if it produced something usable
                if self._is_usable(a_answer):
                    final_answer = a_answer
                elif self._is_usable(s_answer):
                    final_answer = s_answer  # fall back to structured

            if final_answer:
                entity_registry[f"answer_{sq.id}"] = final_answer

            sq_details.append({
                "id": sq.id, "text": sq.text,
                "resolved_text": resolved_text,
                "mode": "escalated" if decision == "ESCALATE" else "structured",
                "answer": final_answer,
                "structured_tokens": s_tokens,
                "agentic_tokens": a_tokens,
                "total_tokens": s_tokens + a_tokens,
                "escalation_decision": decision,
                "status": "verified" if self._is_usable(final_answer) else "failed",
            })

        # Step 3: Synthesize final answer
        final_answer = await self._synthesize(
            question, sub_questions, sq_details, entity_registry, blackboard,
        )

        elapsed = time.monotonic() - t0

        # Compute mode distribution
        n_accept = escalation_decisions.count("ACCEPT")
        n_escalate = escalation_decisions.count("ESCALATE")

        return M6PipelineResult(
            question=question,
            pred_answer=final_answer,
            question_type=blackboard.question_type,
            expected_answer=blackboard.expected_answer,
            num_sub_questions=len(sub_questions),
            num_workers=len(sub_questions),
            total_tokens=total_tokens,
            wall_clock_seconds=elapsed,
            sub_question_details=sq_details,
            entity_registry=entity_registry,
            mode_distribution={"accept": n_accept, "escalate": n_escalate},
            mode_tokens=mode_tokens,
            decomposition_text=planner.last_decomposition_text,
        )

    # ── Worker execution ─────────────────────────────────────────────

    async def _run_structured(
        self,
        sq: M6SubQuestion,
        question: str,
        entity_registry: dict[str, str],
    ) -> tuple[str, int, int]:
        """Run structured worker. Returns (answer, tokens, evidence_count)."""
        bb = Blackboard(question=question, token_budget=self.token_budget)
        sq_copy = M6SubQuestion(
            id=sq.id, text=sq.text, dependencies=[],
            known_entities=list(sq.known_entities),
            unknown_entities=list(sq.unknown_entities),
            search_hints=list(sq.search_hints),
            search_queries=list(sq.search_queries),
            mode=RetrievalMode.STRUCTURED,
            status=SubQuestionStatus.READY,
        )
        await bb.set_search_plan([sq_copy])
        for name, value in entity_registry.items():
            await bb.post_entity(EntityEntry(
                name=name, value=value, source_evidence_id="prior",
            ))

        worker = StructuredWorker(
            agent_id=f"structured_{sq.id}",
            llm_client=self.worker_llm,
            base_tools=self.tools,
            assigned_sq_id=sq.id,
            retrieval_top_k=self.structured_retrieval_top_k,
            max_queries_per_entity=self.structured_max_queries,
            retry_low_confidence=self.structured_retry_low_confidence,
            confidence_threshold=self.structured_confidence_threshold,
        )

        try:
            await asyncio.wait_for(worker.tick(bb), timeout=self.structured_timeout)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Structured worker failed on SQ-%d: %s", sq.id, exc)
            return "", 0, 0

        snapshot = await bb.get_snapshot()
        sq_data = snapshot["sub_questions"][0] if snapshot["sub_questions"] else {}
        answer = sq_data.get("answer", "") or ""
        evidence_count = len([
            e for e in snapshot["evidence"] if e["sub_question_id"] == sq.id
        ])
        return answer, snapshot["tokens_used"], evidence_count

    async def _run_agentic(
        self,
        sq: M6SubQuestion,
        question: str,
        entity_registry: dict[str, str],
    ) -> tuple[str, int, int]:
        """Run agentic worker. Returns (answer, tokens, evidence_count)."""
        bb = Blackboard(question=question, token_budget=self.token_budget)
        sq_copy = M6SubQuestion(
            id=sq.id, text=sq.text, dependencies=[],
            known_entities=list(sq.known_entities),
            unknown_entities=list(sq.unknown_entities),
            search_hints=list(sq.search_hints),
            search_queries=list(sq.search_queries),
            mode=RetrievalMode.AGENTIC,
            status=SubQuestionStatus.READY,
        )
        await bb.set_search_plan([sq_copy])
        for name, value in entity_registry.items():
            await bb.post_entity(EntityEntry(
                name=name, value=value, source_evidence_id="prior",
            ))

        worker = AgenticWorker(
            agent_id=f"agentic_{sq.id}",
            llm_client=self.worker_llm,
            tools=self.tools,
            assigned_sq_id=sq.id,
            max_steps=self.worker_max_steps,
        )

        try:
            await asyncio.wait_for(worker.tick(bb), timeout=self.agentic_timeout)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Agentic worker failed on SQ-%d: %s", sq.id, exc)
            return "", 0, 0

        snapshot = await bb.get_snapshot()
        sq_data = snapshot["sub_questions"][0] if snapshot["sub_questions"] else {}
        answer = sq_data.get("answer", "") or ""
        evidence_count = len([
            e for e in snapshot["evidence"] if e["sub_question_id"] == sq.id
        ])
        return answer, snapshot["tokens_used"], evidence_count

    async def _run_aggregate(
        self,
        sq: M6SubQuestion,
        question: str,
        entity_registry: dict[str, str],
        main_blackboard: Blackboard,
    ) -> tuple[str, int]:
        """Run aggregate worker (no retrieval, synthesize from context)."""
        bb = Blackboard(question=question, token_budget=self.token_budget)
        sq_copy = M6SubQuestion(
            id=sq.id, text=sq.text, dependencies=[],
            mode=RetrievalMode.AGGREGATE,
            status=SubQuestionStatus.READY,
        )
        await bb.set_search_plan([sq_copy])
        for name, value in entity_registry.items():
            await bb.post_entity(EntityEntry(
                name=name, value=value, source_evidence_id="prior",
            ))

        worker = AggregateWorker(
            agent_id=f"aggregate_{sq.id}",
            llm_client=self.worker_llm,
            assigned_sq_id=sq.id,
        )

        try:
            await asyncio.wait_for(worker.tick(bb), timeout=60.0)
        except (asyncio.TimeoutError, Exception) as exc:
            logger.warning("Aggregate worker failed on SQ-%d: %s", sq.id, exc)
            return "", 0

        snapshot = await bb.get_snapshot()
        sq_data = snapshot["sub_questions"][0] if snapshot["sub_questions"] else {}
        answer = sq_data.get("answer", "") or ""
        return answer, snapshot["tokens_used"]

    # ── Escalation decision ──────────────────────────────────────────

    async def _escalation_decision(
        self,
        question: str,
        sq_text: str,
        structured_answer: str,
        evidence_count: int,
        entity_registry: dict[str, str],
    ) -> str:
        """Decide ACCEPT or ESCALATE based on the configured mode."""

        if self.escalation_mode == "always_structured":
            return "ACCEPT"

        if self.escalation_mode == "always_agentic":
            return "ESCALATE"

        if self.escalation_mode == "heuristic":
            # Simple rule: escalate if structured didn't find a usable answer
            if not self._is_usable(structured_answer):
                return "ESCALATE"
            return "ACCEPT"

        if self.escalation_mode == "learned":
            return await self._learned_escalation(
                question, sq_text, structured_answer, evidence_count, entity_registry,
            )

        logger.warning("Unknown escalation mode '%s', defaulting to heuristic", self.escalation_mode)
        return "ESCALATE" if not self._is_usable(structured_answer) else "ACCEPT"

    async def _learned_escalation(
        self,
        question: str,
        sq_text: str,
        structured_answer: str,
        evidence_count: int,
        entity_registry: dict[str, str],
    ) -> str:
        """Call the RL-trained escalation policy LLM."""
        entity_lines = []
        for name, value in entity_registry.items():
            entity_lines.append(f"- {name}: {value}")
        entity_context = "\n".join(entity_lines) if entity_lines else "(none yet)"

        prompt = ESCALATION_PROMPT_TEMPLATE.format(
            question=question,
            sq_text=sq_text,
            structured_answer=structured_answer if structured_answer else "(no answer)",
            evidence_count=evidence_count,
            entity_context=entity_context,
        )

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.escalation_llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    tools=None,
                    temperature=0.0,
                    max_tokens=100,
                ),
            )
        except Exception as exc:
            logger.warning("Escalation LLM call failed: %s, defaulting to ESCALATE", exc)
            return "ESCALATE"

        raw = (response.get("message") or {}).get("content", "")

        # Strip thinking tags if present
        raw_clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        if "ESCALATE" in raw_clean.upper():
            return "ESCALATE"
        if "ACCEPT" in raw_clean.upper():
            return "ACCEPT"

        # Default: escalate if structured answer is empty (safer)
        logger.warning("Could not parse escalation decision from: %s", raw_clean[:100])
        return "ESCALATE" if not self._is_usable(structured_answer) else "ACCEPT"

    # ── Synthesis ────────────────────────────────────────────────────

    async def _synthesize(
        self,
        question: str,
        sub_questions: list[M6SubQuestion],
        sq_details: list[dict],
        entity_registry: dict[str, str],
        blackboard: Blackboard,
    ) -> str:
        """Synthesize final answer from all SQ results."""
        # Build context for synthesizer
        context_lines = []
        for detail in sq_details:
            answer = detail.get("answer", "unknown")
            context_lines.append(
                f"Sub-question {detail['id']}: \"{detail['text']}\" -> {answer}"
            )

        # If only one SQ with a usable answer, return it directly
        usable_answers = [
            d["answer"] for d in sq_details
            if self._is_usable(d.get("answer", ""))
        ]
        if len(usable_answers) == 1 and len(sq_details) == 1:
            return usable_answers[0]

        # Use synthesizer agent via blackboard
        # Populate blackboard with collected results
        for detail in sq_details:
            sq_obj = None
            for sq in sub_questions:
                if sq.id == detail["id"]:
                    sq_obj = sq
                    break
            if sq_obj is not None:
                sq_obj.answer = detail.get("answer")
                sq_obj.status = (
                    SubQuestionStatus.VERIFIED
                    if self._is_usable(detail.get("answer", ""))
                    else SubQuestionStatus.FAILED
                )

        blackboard.allow_synthesis = True
        for name, value in entity_registry.items():
            await blackboard.post_entity(EntityEntry(
                name=name, value=value, source_evidence_id="escalation",
            ))

        synthesizer = SynthesizerAgent(
            llm_client=self.llm,
            prompt_path=self.synthesizer_prompt,
            consistency_prompt_path=self.synthesizer_consistency_prompt,
            enable_consistency_check=False,
        )

        obs = await synthesizer.observe(blackboard)
        if synthesizer.should_act(obs):
            await synthesizer.act(obs, blackboard)

        if blackboard.final_answer:
            return blackboard.final_answer

        # Fallback: return the last usable answer
        for detail in reversed(sq_details):
            if self._is_usable(detail.get("answer", "")):
                return detail["answer"]
        return "unknown"

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_usable(answer: str) -> bool:
        if not answer:
            return False
        return answer.strip().lower() not in ("unknown", "error", "", "none", "n/a")

    @staticmethod
    def _topological_sort(sub_questions: list[M6SubQuestion]) -> list[M6SubQuestion]:
        """Sort sub-questions in dependency order."""
        resolved: set[int] = set()
        remaining = list(sub_questions)
        ordered: list[M6SubQuestion] = []

        while remaining:
            batch = [
                sq for sq in remaining
                if all(dep in resolved for dep in sq.dependencies)
            ]
            if not batch:
                # Unresolvable dependencies — append remaining as-is
                ordered.extend(remaining)
                break
            ordered.extend(batch)
            for sq in batch:
                resolved.add(sq.id)
                remaining.remove(sq)

        return ordered
