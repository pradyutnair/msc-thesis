"""M6 Pipeline: top-level entry point for blackboard-coordinated multi-agent RAG.

Creates per question: Blackboard + agents + Coordinator → run → M6PipelineResult.
Feature flags for ablations: enable_critic, enable_backtracking, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.coordinator import Coordinator
from multi_agent.m6.decomposer_agent import DecomposerAgent
from multi_agent.m6.retriever_agent import RetrieverAgent
from multi_agent.m6.critic_agent import CriticAgent
from multi_agent.m6.synthesizer_agent import SynthesizerAgent
from multi_agent.m6.types import M6PipelineResult, SubQuestionStatus

logger = logging.getLogger(__name__)


class M6Pipeline:
    """Blackboard-coordinated multi-agent RAG pipeline.

    Creates all agents + coordinator per question for stateless operation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        retriever_llm_client: LLMClient,
        tools: ToolRegistry,
        # Prompt paths
        decomposer_prompt: str | None = None,
        retriever_prompt: str | None = None,
        critic_prompt: str | None = None,
        synthesizer_prompt: str | None = None,
        synthesizer_consistency_prompt: str | None = None,
        # Agent config
        num_retrievers: int = 2,
        retriever_max_loops: int = 5,
        retriever_max_budget: int = 32_000,
        # Coordinator config
        max_ticks: int = 30,
        token_budget: int = 200_000,
        wall_clock_timeout: float = 300.0,
        # Feature flags
        enable_critic: bool = True,
        enable_backtracking: bool = True,
        enable_consistency_check: bool = True,
    ):
        self.llm = llm_client
        self.retriever_llm = retriever_llm_client
        self.tools = tools

        # Prompt paths
        self.decomposer_prompt = decomposer_prompt
        self.retriever_prompt = retriever_prompt
        self.critic_prompt = critic_prompt
        self.synthesizer_prompt = synthesizer_prompt
        self.synthesizer_consistency_prompt = synthesizer_consistency_prompt

        # Agent config
        self.num_retrievers = num_retrievers
        self.retriever_max_loops = retriever_max_loops
        self.retriever_max_budget = retriever_max_budget

        # Coordinator config
        self.max_ticks = max_ticks
        self.token_budget = token_budget
        self.wall_clock_timeout = wall_clock_timeout

        # Feature flags
        self.enable_critic = enable_critic
        self.enable_backtracking = enable_backtracking
        self.enable_consistency_check = enable_consistency_check

    async def run(self, question: str) -> M6PipelineResult:
        """Run the full M6 pipeline on a single question."""
        t0 = time.monotonic()

        # Create blackboard
        blackboard = Blackboard(question=question, token_budget=self.token_budget)

        # Create agents
        agents = self._create_agents()

        # Create coordinator
        coordinator = Coordinator(
            agents=agents,
            max_ticks=self.max_ticks,
            token_budget=self.token_budget,
            wall_clock_timeout=self.wall_clock_timeout,
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
        """Create the ordered agent list for the coordinator."""
        agents = []

        # 1. Decomposer (thinking ON — uses main llm_client)
        agents.append(DecomposerAgent(
            llm_client=self.llm,
            prompt_path=self.decomposer_prompt,
        ))

        # 2. Retrievers (thinking OFF — use retriever_llm_client)
        auto_verify = not self.enable_critic
        for i in range(self.num_retrievers):
            agents.append(RetrieverAgent(
                agent_id=f"retriever_{i}",
                llm_client=self.retriever_llm,
                tools=self.tools,
                prompt_path=self.retriever_prompt,
                max_loops=self.retriever_max_loops,
                max_token_budget=self.retriever_max_budget,
                auto_verify=auto_verify,
            ))

        # 3. Critic (thinking ON)
        if self.enable_critic:
            agents.append(CriticAgent(
                llm_client=self.llm,
                prompt_path=self.critic_prompt,
                enable_backtracking=self.enable_backtracking,
            ))

        # 4. Synthesizer (thinking ON)
        agents.append(SynthesizerAgent(
            llm_client=self.llm,
            prompt_path=self.synthesizer_prompt,
            consistency_prompt_path=self.synthesizer_consistency_prompt,
            enable_consistency_check=self.enable_consistency_check,
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
        status_counts = {}
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
            verified_count=status_counts.get(SubQuestionStatus.VERIFIED.value, 0),
            failed_count=status_counts.get(SubQuestionStatus.FAILED.value, 0),
            contradictions=snapshot.get("contradictions", []),
            knowledge_gaps=snapshot.get("knowledge_gaps", []),
            execution_log=snapshot.get("execution_log", []),
            termination_reason=snapshot.get("termination_reason", ""),
        )
