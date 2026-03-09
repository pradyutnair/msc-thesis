"""M6 Pipeline: top-level entry point for blackboard-coordinated multi-agent RAG.

Creates per question: Blackboard + PlannerAgent + WorkerAgents + Coordinator →
run → M6PipelineResult.

Architecture (AgentFlow-inspired):
  - PlannerAgent: decompose → monitor → synthesize lifecycle
  - WorkerAgents: autonomous plan → execute → verify loops per sub-question
  - Coordinator: concurrent async loops + watchdog
  - Blackboard: shared state for emergent coordination
"""

from __future__ import annotations

import logging
import time
from typing import Any

from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.coordinator import Coordinator
from multi_agent.m6.planner_agent import PlannerAgent
from multi_agent.m6.worker_agent import WorkerAgent
from multi_agent.m6.types import M6PipelineResult

logger = logging.getLogger(__name__)


class M6Pipeline:
    """Blackboard-coordinated multi-agent RAG pipeline.

    Two agent types: PlannerAgent (decompose/synthesize) and WorkerAgents
    (AgentFlow-style plan/execute/verify per sub-question).
    """

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
        worker_verify_prompt: str | None = None,
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
    ):
        self.llm = llm_client
        self.worker_llm = worker_llm_client
        self.tools = tools

        self.decomposer_prompt = decomposer_prompt
        self.synthesizer_prompt = synthesizer_prompt
        self.synthesizer_consistency_prompt = synthesizer_consistency_prompt
        self.worker_plan_prompt = worker_plan_prompt
        self.worker_verify_prompt = worker_verify_prompt

        self.num_workers = num_workers
        self.worker_max_steps = worker_max_steps

        self.token_budget = token_budget
        self.wall_clock_timeout = wall_clock_timeout
        self.idle_timeout = idle_timeout
        self.max_actions = max_actions

        self.enable_consistency_check = enable_consistency_check
        self.max_redecompositions = max_redecompositions

    async def run(self, question: str) -> M6PipelineResult:
        """Run the full M6 pipeline on a single question."""
        t0 = time.monotonic()

        # Create blackboard
        blackboard = Blackboard(question=question, token_budget=self.token_budget)

        # Create agents
        agents = self._create_agents()

        coordinator = Coordinator(
            agents=agents,
            token_budget=self.token_budget,
            wall_clock_timeout=self.wall_clock_timeout,
            idle_timeout=self.idle_timeout,
            max_actions=self.max_actions,
        )

        # Run
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

        # Build result from blackboard snapshot
        snapshot = await blackboard.get_snapshot()
        result = self._build_result(question, answer, snapshot, elapsed)
        return result

    def _create_agents(self) -> list:
        """Create the agent list: 1 PlannerAgent + N WorkerAgents."""
        agents = []

        # PlannerAgent (thinking ON — decompose + monitor + synthesize)
        agents.append(PlannerAgent(
            llm_client=self.llm,
            decompose_prompt_path=self.decomposer_prompt,
            synthesize_prompt_path=self.synthesizer_prompt,
            consistency_prompt_path=self.synthesizer_consistency_prompt,
            max_redecompositions=self.max_redecompositions,
            enable_consistency_check=self.enable_consistency_check,
        ))

        # WorkerAgents (thinking OFF — plan/execute/verify per sub-question)
        for i in range(self.num_workers):
            agents.append(WorkerAgent(
                agent_id=f"worker_{i}",
                llm_client=self.worker_llm,
                tools=self.tools,
                plan_prompt_path=self.worker_plan_prompt,
                verify_prompt_path=self.worker_verify_prompt,
                max_steps=self.worker_max_steps,
            ))

        return agents

    @staticmethod
    def _build_result(
        question: str,
        answer: str,
        snapshot: dict[str, Any],
        elapsed: float,
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
            total_ticks=snapshot.get("current_tick", 0),
            total_tokens=snapshot.get("tokens_used", 0),
            wall_clock_seconds=elapsed,
            backtrack_count=snapshot.get("backtrack_count", 0),
            sub_question_details=sqs,
            entity_registry=snapshot.get("entity_registry", {}),
            evidence_count=snapshot.get("evidence_count", 0),
            verified_count=status_counts.get("verified", 0),
            failed_count=status_counts.get("failed", 0),
            contradictions=snapshot.get("contradictions", []),
            knowledge_gaps=snapshot.get("knowledge_gaps", []),
            execution_log=snapshot.get("execution_log", []),
            termination_reason=snapshot.get("termination_reason", ""),
        )
