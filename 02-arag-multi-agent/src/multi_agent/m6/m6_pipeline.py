"""M6 Pipeline: top-level entry point for blackboard-coordinated multi-agent RAG.

Creates per question: Blackboard + PlannerAgent + SynthesizerAgent + WorkerAgents
+ Coordinator -> run -> M6PipelineResult.

Architecture:
  - PlannerAgent: decompose -> monitor -> signal synthesis
  - WorkerAgents: autonomous plan -> execute loops per sub-question
  - SynthesizerAgent: aggregate evidence into final answer
  - Coordinator: concurrent async loops + watchdog
  - Blackboard: shared state for emergent coordination
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.coordinator import Coordinator
from multi_agent.m6.planner_agent import PlannerAgent
from multi_agent.m6.synthesizer_agent import SynthesizerAgent
from multi_agent.m6.worker_agent import WorkerAgent
from multi_agent.m6.types import M6PipelineResult

logger = logging.getLogger(__name__)


class M6Pipeline:
    """Blackboard-coordinated multi-agent RAG pipeline."""

    def __init__(
        self,
        llm_client: LLMClient,
        worker_llm_client: LLMClient,
        tools: ToolRegistry,
        # Prompt paths
        decomposer_prompt: str | None = None,
        synthesizer_prompt: str | None = None,
        synthesizer_consistency_prompt: str | None = None,
        worker_plan_prompt: str | None = None,
        # Agent config
        num_workers: int = 2,
        worker_max_steps: int = 8,
        # Coordinator config
        token_budget: int = 200_000,
        wall_clock_timeout: float = 300.0,
        idle_timeout: float = 30.0,
        max_actions: int = 100,
        # Feature flags
        enable_consistency_check: bool = True,
        max_redecompositions: int = 1,
        enable_extraction_pass: bool = False,
        enable_answer_validation: bool = False,
        enable_bridge_guard: bool = False,
    ):
        self.llm = llm_client
        self.worker_llm = worker_llm_client
        self.tools = tools

        self.decomposer_prompt = decomposer_prompt
        self.synthesizer_prompt = synthesizer_prompt
        self.synthesizer_consistency_prompt = synthesizer_consistency_prompt
        self.worker_plan_prompt = worker_plan_prompt

        self.num_workers = num_workers
        self.worker_max_steps = worker_max_steps

        self.token_budget = token_budget
        self.wall_clock_timeout = wall_clock_timeout
        self.idle_timeout = idle_timeout
        self.max_actions = max_actions

        self.enable_consistency_check = enable_consistency_check
        self.max_redecompositions = max_redecompositions
        self.enable_extraction_pass = enable_extraction_pass
        self.enable_answer_validation = enable_answer_validation
        self.enable_bridge_guard = enable_bridge_guard
        self.enable_semantic_warmstart = False

    async def run(self, question: str) -> M6PipelineResult:
        """Run the full M6 pipeline on a single question."""
        t0 = time.monotonic()

        blackboard = Blackboard(question=question, token_budget=self.token_budget)

        # Warm-start: keyword search on full question for initial context
        warm_ctx = await self._warm_start_search(question)
        if warm_ctx:
            blackboard.warm_start_context = warm_ctx

        # Phase 1: decompose upfront so we know how many workers to spawn
        planner = PlannerAgent(
            llm_client=self.llm,
            decompose_prompt_path=self.decomposer_prompt,
            max_redecompositions=self.max_redecompositions,
        )
        num_sqs = await planner.decompose_first(blackboard)

        # Phase 2: create workers dynamically
        num_workers = 1 if num_sqs < 2 else num_sqs
        logger.info(
            "Dynamic workers: %d sub-questions -> %d workers",
            num_sqs, num_workers,
        )

        agents = self._create_agents(planner=planner, num_workers=num_workers)

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
            logger.error("M6 pipeline error: %s", exc)
            return M6PipelineResult(
                question=question,
                pred_answer=f"Error: {exc}",
                error=str(exc),
                wall_clock_seconds=time.monotonic() - t0,
            )

        elapsed = time.monotonic() - t0

        snapshot = await blackboard.get_snapshot()
        result = self._build_result(question, answer, snapshot, elapsed,
                                    num_workers=num_workers)
        return result

    async def _warm_start_search(self, question: str) -> str:
        """Run keyword + semantic search on the full question for initial context."""
        lines = []

        # Keyword warm-start
        try:
            keyword_tool = self.tools.get("keyword_search")
            if keyword_tool is not None and hasattr(keyword_tool, "chunks"):
                keywords = [w for w in question.replace("?", "").split() if len(w) > 2]
                if keywords:
                    def _kw_search():
                        scored = []
                        for chunk in keyword_tool.chunks:
                            text_lower = chunk["text"].lower()
                            score = sum(
                                text_lower.count(kw.lower()) * len(kw)
                                for kw in keywords
                            )
                            if score > 0:
                                scored.append((score, chunk))
                        scored.sort(key=lambda x: -x[0])
                        return scored[:3]

                    loop = asyncio.get_running_loop()
                    top = await loop.run_in_executor(None, _kw_search)
                    if top:
                        lines.append("Keyword search on full question:")
                        for _, chunk in top:
                            cid = chunk.get("id", "?")
                            text = chunk.get("text", "")[:400]
                            lines.append(f"[{cid}] {text}")
        except Exception as exc:
            logger.warning("Keyword warm-start failed: %s", exc)

        # Semantic warm-start (v22)
        if self.enable_semantic_warmstart:
            try:
                semantic_tool = self.tools.get("semantic_search")
                if semantic_tool is not None:
                    from arag.core.context import AgentContext
                    ctx = AgentContext()

                    def _sem_search():
                        return semantic_tool.execute(ctx, query=question, top_k=3)

                    loop = asyncio.get_running_loop()
                    result, _ = await loop.run_in_executor(None, _sem_search)
                    if result and "No results" not in result:
                        lines.append("\nSemantic search on full question:")
                        lines.append(result[:1500])
            except Exception as exc:
                logger.warning("Semantic warm-start failed: %s", exc)

        return "\n".join(lines) if lines else ""

    def _create_agents(self, planner: PlannerAgent, num_workers: int) -> list:
        """Create the agent list: PlannerAgent + SynthesizerAgent + N WorkerAgents."""
        agents = [planner]

        agents.append(SynthesizerAgent(
            llm_client=self.llm,
            prompt_path=self.synthesizer_prompt,
            consistency_prompt_path=self.synthesizer_consistency_prompt,
            enable_consistency_check=self.enable_consistency_check,
        ))

        for i in range(num_workers):
            agents.append(WorkerAgent(
                agent_id=f"worker_{i}",
                llm_client=self.worker_llm,
                tools=self.tools,
                plan_prompt_path=self.worker_plan_prompt,
                max_steps=self.worker_max_steps,
                enable_extraction_pass=self.enable_extraction_pass,
                enable_answer_validation=self.enable_answer_validation,
                enable_bridge_guard=self.enable_bridge_guard,
            ))

        return agents

    @staticmethod
    def _build_result(
        question: str,
        answer: str,
        snapshot: dict[str, Any],
        elapsed: float,
        num_workers: int = 0,
    ) -> M6PipelineResult:
        """Build M6PipelineResult from blackboard snapshot."""
        sqs = snapshot.get("sub_questions", [])
        status_counts: dict[str, int] = {}
        for sq in sqs:
            s = sq["status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        return M6PipelineResult(
            question=question,
            pred_answer=answer,
            num_sub_questions=len(sqs),
            num_workers=num_workers,
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
        )
