"""End-to-end multi-agent pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.aggregator import Aggregator
from multi_agent.decomposer import Decomposer
from multi_agent.dispatcher import Dispatcher
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.types import PipelineResult

logger = logging.getLogger(__name__)


class MultiAgentPipeline:
    """Orchestrate: question → decompose → dispatch → aggregate → result.

    Parameters
    ----------
    llm_client:
        Shared LLM client (vLLM-compatible OpenAI API).
    tools:
        Pre-initialized tool registry (keyword_search, semantic_search,
        read_chunk, finish).
    config:
        Experiment configuration (overrides defaults).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        config: Config | None = None,
    ):
        self.llm = llm_client
        self.tools = tools

        # Extract config values with defaults
        cfg = config or Config()
        ma_cfg = cfg.get("multi_agent", {}) or {}

        self.enable_doc_cache = ma_cfg.get("enable_doc_cache", True)
        self.enable_self_verify = ma_cfg.get("enable_self_verify", True)
        self.enable_decomposer = ma_cfg.get("enable_decomposer", True)
        self.skip_aggregator = ma_cfg.get("skip_aggregator", False)
        self.max_loops_per_agent = ma_cfg.get("max_loops_per_agent", 5)
        self.max_token_budget = ma_cfg.get("max_token_budget", 64000)
        self.max_concurrent = ma_cfg.get("max_concurrent", 3)
        self.verbose = ma_cfg.get("verbose", False)
        self.use_nothink = ma_cfg.get("use_nothink", False)
        # Prompt paths (optional overrides)
        self.decomposer_prompt = ma_cfg.get("decomposer_prompt")
        self.search_agent_prompt = ma_cfg.get("search_agent_prompt")
        self.aggregator_prompt = ma_cfg.get("aggregator_prompt")

    def _create_components(self) -> tuple[Decomposer, Dispatcher, Aggregator, EvidenceCache]:
        """Instantiate pipeline components for one question."""
        cache = EvidenceCache(enabled=self.enable_doc_cache)

        decomposer = Decomposer(
            llm_client=self.llm,
            prompt_path=self.decomposer_prompt,
            use_nothink=self.use_nothink,
        )

        dispatcher = Dispatcher(
            llm_client=self.llm,
            tools=self.tools,
            evidence_cache=cache,
            max_loops_per_agent=self.max_loops_per_agent,
            max_token_budget=self.max_token_budget,
            max_concurrent=self.max_concurrent,
            verbose=self.verbose,
            search_agent_prompt_path=self.search_agent_prompt,
        )

        aggregator = Aggregator(
            llm_client=self.llm,
            evidence_cache=cache,
            enable_self_verify=self.enable_self_verify,
            prompt_path=self.aggregator_prompt,
        )

        return decomposer, dispatcher, aggregator, cache

    async def run(self, question: str) -> PipelineResult:
        """Run the full multi-agent pipeline for one question.

        Steps:
        1. Decompose question into sub-questions + dependency DAG
        2. Dispatch search agents in topological waves
        3. Aggregate sub-answers into final answer
        """
        t0 = time.monotonic()
        result = PipelineResult(question=question)

        decomposer, dispatcher, aggregator, cache = self._create_components()

        try:
            # --- Phase 1: Decomposition ---
            if self.enable_decomposer:
                plan = await decomposer.decompose(question)
            else:
                # Ablation A1: skip decomposer, send full question to one agent
                from multi_agent.types import DecompositionPlan, SubQuestion

                plan = DecompositionPlan(
                    question_type="single_hop",
                    sub_questions=[
                        SubQuestion(index=0, text=question, search_hints=[])
                    ],
                    dependency_edges=[],
                )

            result.decomposition = plan
            result.question_type = plan.question_type
            result.num_sub_questions = len(plan.sub_questions)

            if self.verbose:
                print(
                    f"\nDecomposition: {plan.question_type}, "
                    f"{len(plan.sub_questions)} sub-Qs"
                )
                for sq in plan.sub_questions:
                    deps = f" (depends on {sq.depends_on})" if sq.depends_on else ""
                    print(f"  SQ-{sq.index}: {sq.text}{deps}")

            # --- Phase 2: Dispatch ---
            agent_results, num_waves = await dispatcher.dispatch(plan)
            result.agent_results = agent_results
            result.num_waves = num_waves

            # Sum agent tokens
            agent_tokens = sum(
                ar.total_tokens for ar in agent_results.values()
            )

            # --- Phase 3: Aggregation ---
            if self.skip_aggregator:
                # Ablation A2: use last agent's answer directly
                last_idx = max(agent_results.keys())
                result.final_answer = agent_results[last_idx].answer
                agg_tokens = 0
            else:
                final_answer, agg_tokens = await aggregator.aggregate(
                    question, plan, agent_results
                )
                result.final_answer = final_answer
            result.aggregator_tokens = agg_tokens

            # --- Cache analytics ---
            result.cache_analytics = cache.compute_analytics_sync()

            # --- Token totals ---
            result.total_tokens = agent_tokens + agg_tokens

        except Exception as exc:
            logger.error("Pipeline error for '%s': %s", question[:60], exc)
            result.error = str(exc)
            # Try to salvage an answer from agent results
            if result.agent_results:
                best = max(
                    result.agent_results.values(),
                    key=lambda r: len(r.answer),
                )
                result.final_answer = best.answer

        result.wall_clock_seconds = time.monotonic() - t0
        return result

    def run_sync(self, question: str) -> PipelineResult:
        """Synchronous wrapper for environments without an event loop."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — use nest_asyncio or thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(
                    asyncio.run, self.run(question)
                ).result()
        else:
            return asyncio.run(self.run(question))
