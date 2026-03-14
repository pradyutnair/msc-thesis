"""Adaptive Agency Pipeline: mode-aware multi-agent RAG.

Creates per question: Blackboard + PlannerAgent (with mode assignment)
+ mode-specific workers + SynthesizerAgent + Coordinator -> run -> result.

Key difference from M6Pipeline: workers are created based on the planner's
per-sub-question mode assignment (structured, agentic, aggregate).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.blackboard import Blackboard
from multi_agent.coordinator import Coordinator
from multi_agent.planner_agent import PlannerAgent
from multi_agent.synthesizer_agent import SynthesizerAgent
from multi_agent.types import M6PipelineResult, RetrievalMode
from multi_agent.workers.agentic_worker import AgenticWorker
from multi_agent.workers.aggregate_worker import AggregateWorker
from multi_agent.workers.structured_worker import StructuredWorker

logger = logging.getLogger(__name__)


class AdaptiveAgencyPipeline:
    """Blackboard-coordinated multi-agent RAG with per-SQ mode selection."""

    def __init__(
        self,
        llm_client: LLMClient,
        worker_llm_client: LLMClient,
        tools: ToolRegistry,
        decomposer_prompt: str | None = None,
        synthesizer_prompt: str | None = None,
        synthesizer_consistency_prompt: str | None = None,
        worker_plan_prompt: str | None = None,
        worker_max_steps: int = 8,
        token_budget: int = 300_000,
        wall_clock_timeout: float = 900.0,
        idle_timeout: float = 300.0,
        max_actions: int = 100,
        enable_consistency_check: bool = False,
        max_redecompositions: int = 1,
        structured_retrieval_top_k: int = 10,
        structured_max_queries: int = 6,
        structured_retry_low_confidence: bool = True,
        structured_confidence_threshold: float = 0.6,
        decompose_temperature: float = 0.0,
        force_mode: str | None = None,
    ):
        self.llm = llm_client
        self.worker_llm = worker_llm_client
        self.decompose_temperature = decompose_temperature
        self.force_mode = force_mode
        self.tools = tools

        self.decomposer_prompt = decomposer_prompt
        self.synthesizer_prompt = synthesizer_prompt
        self.synthesizer_consistency_prompt = synthesizer_consistency_prompt
        self.worker_plan_prompt = worker_plan_prompt

        self.worker_max_steps = worker_max_steps
        self.token_budget = token_budget
        self.wall_clock_timeout = wall_clock_timeout
        self.idle_timeout = idle_timeout
        self.max_actions = max_actions
        self.enable_consistency_check = enable_consistency_check
        self.max_redecompositions = max_redecompositions

        self.structured_retrieval_top_k = structured_retrieval_top_k
        self.structured_max_queries = structured_max_queries
        self.structured_retry_low_confidence = structured_retry_low_confidence
        self.structured_confidence_threshold = structured_confidence_threshold

    async def run(self, question: str) -> M6PipelineResult:
        t0 = time.monotonic()

        blackboard = Blackboard(question=question, token_budget=self.token_budget)

        planner = PlannerAgent(
            llm_client=self.llm,
            decompose_prompt_path=self.decomposer_prompt,
            max_redecompositions=self.max_redecompositions,
            decompose_temperature=self.decompose_temperature,
        )
        num_sqs = await planner.decompose_first(blackboard)

        if self.force_mode:
            self._override_modes(blackboard)

        agents = self._create_agents(planner, blackboard)

        coordinator = Coordinator(
            agents=agents,
            token_budget=self.token_budget,
            wall_clock_timeout=self.wall_clock_timeout,
            idle_timeout=self.idle_timeout,
            max_actions=self.max_actions,
        )

        try:
            answer = await coordinator.run(blackboard)
        except Exception as exc:
            logger.error("Adaptive pipeline error: %s", exc)
            return M6PipelineResult(
                question=question,
                pred_answer=f"Error: {exc}",
                error=str(exc),
                wall_clock_seconds=time.monotonic() - t0,
            )

        elapsed = time.monotonic() - t0
        snapshot = await blackboard.get_snapshot()
        return self._build_result(
            question, answer, snapshot, elapsed,
            decomposition_text=planner.last_decomposition_text,
        )

    def _override_modes(self, blackboard: Blackboard) -> None:
        """Override all sub-question modes for ablation studies."""
        modes = [RetrievalMode.STRUCTURED, RetrievalMode.AGENTIC, RetrievalMode.AGGREGATE]
        for sq in blackboard.search_plan:
            if self.force_mode == "random":
                sq.mode = random.choice(modes)
            else:
                sq.mode = RetrievalMode(self.force_mode)
        logger.info("Force-mode override: all SQs set to %s", self.force_mode)

    def _create_agents(self, planner: PlannerAgent, blackboard: Blackboard) -> list:
        agents: list[Any] = [planner]

        agents.append(SynthesizerAgent(
            llm_client=self.llm,
            prompt_path=self.synthesizer_prompt,
            consistency_prompt_path=self.synthesizer_consistency_prompt,
            enable_consistency_check=self.enable_consistency_check,
        ))

        mode_counts: dict[str, int] = {}
        for sq in blackboard.search_plan:
            mode_counts[sq.mode.value] = mode_counts.get(sq.mode.value, 0) + 1

            if sq.mode == RetrievalMode.STRUCTURED:
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
            elif sq.mode == RetrievalMode.AGENTIC:
                worker = AgenticWorker(
                    agent_id=f"agentic_{sq.id}",
                    llm_client=self.worker_llm,
                    tools=self.tools,
                    assigned_sq_id=sq.id,
                    plan_prompt_path=self.worker_plan_prompt,
                    max_steps=self.worker_max_steps,
                )
            elif sq.mode == RetrievalMode.AGGREGATE:
                worker = AggregateWorker(
                    agent_id=f"aggregate_{sq.id}",
                    llm_client=self.worker_llm,
                    assigned_sq_id=sq.id,
                )
            else:
                worker = StructuredWorker(
                    agent_id=f"structured_{sq.id}",
                    llm_client=self.worker_llm,
                    base_tools=self.tools,
                    assigned_sq_id=sq.id,
                )

            agents.append(worker)

        logger.info(
            "Adaptive workers: %d sub-questions -> modes %s",
            len(blackboard.search_plan), mode_counts,
        )
        return agents

    @staticmethod
    def _build_result(
        question: str,
        answer: str,
        snapshot: dict[str, Any],
        elapsed: float,
        decomposition_text: str = "",
    ) -> M6PipelineResult:
        sqs = snapshot.get("sub_questions", [])
        status_counts: dict[str, int] = {}
        mode_distribution: dict[str, int] = {}
        for sq in sqs:
            s = sq["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
            m = sq.get("mode", "structured")
            mode_distribution[m] = mode_distribution.get(m, 0) + 1

        return M6PipelineResult(
            question=question,
            pred_answer=answer,
            question_type=snapshot.get("question_type", "unknown"),
            expected_answer=snapshot.get("expected_answer", ""),
            num_sub_questions=len(sqs),
            num_workers=len(sqs),
            total_ticks=snapshot.get("current_tick", 0),
            total_tokens=snapshot.get("tokens_used", 0),
            wall_clock_seconds=elapsed,
            backtrack_count=snapshot.get("backtrack_count", 0),
            sub_question_details=sqs,
            entity_registry=snapshot.get("entity_registry", {}),
            evidence=snapshot.get("evidence", []),
            evidence_count=snapshot.get("evidence_count", 0),
            verified_count=status_counts.get("verified", 0),
            failed_count=status_counts.get("failed", 0),
            execution_log=snapshot.get("execution_log", []),
            termination_reason=snapshot.get("termination_reason", ""),
            mode_distribution=mode_distribution,
            mode_tokens=snapshot.get("mode_tokens", {}),
            decomposition_text=decomposition_text,
        )