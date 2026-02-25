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
from multi_agent.types import DecompositionPlan, PipelineResult, ScoutResult, SubQuestion

logger = logging.getLogger(__name__)


class MultiAgentPipeline:
    """Orchestrate: question → decompose → dispatch → aggregate → result.

    Supports two execution modes controlled by ``enable_osprey``:

    **Standard mode** (M1/M2/M3):
      decompose → dispatch → aggregate

    **OSPREY mode** (M4):
      Phase 1 Scout → ConfidenceGate → EvidenceAwareDecompose →
      dispatch (with global scout evidence) → aggregate (with scout pool)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        config: Config | None = None,
    ):
        self.llm = llm_client
        self.tools = tools

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

        # Prompt paths
        self.decomposer_prompt = ma_cfg.get("decomposer_prompt")
        self.search_agent_prompt = ma_cfg.get("search_agent_prompt")
        self.enable_cep = ma_cfg.get("enable_cep", False)
        self.aggregator_prompt = ma_cfg.get("aggregator_prompt")

        # OSPREY-specific config
        self.enable_osprey = ma_cfg.get("enable_osprey", False)
        self.scout_max_loops = ma_cfg.get("scout_max_loops", 3)
        self.scout_confidence_threshold = ma_cfg.get("scout_confidence_threshold", 0.65)
        self.scout_prompt = ma_cfg.get("scout_prompt")
        self.osprey_decomposer_prompt = ma_cfg.get("osprey_decomposer_prompt")
        self.scout_max_chunks_evidence = ma_cfg.get("scout_max_chunks_evidence", 5)
        self.scout_max_chars_per_chunk = ma_cfg.get("scout_max_chars_per_chunk", 700)

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
            enable_cep=self.enable_cep,
        )

        aggregator = Aggregator(
            llm_client=self.llm,
            evidence_cache=cache,
            enable_self_verify=self.enable_self_verify,
            prompt_path=self.aggregator_prompt,
        )

        return decomposer, dispatcher, aggregator, cache

    # ------------------------------------------------------------------
    # OSPREY helpers
    # ------------------------------------------------------------------

    def _build_scout_evidence_str(
        self,
        scout_result: ScoutResult,
    ) -> str:
        """Format Scout chunks as global chain evidence for Phase 2 agents."""
        if not scout_result.chunks:
            return ""
        lines: list[str] = []
        for chunk in scout_result.chunks[:self.scout_max_chunks_evidence]:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")[:self.scout_max_chars_per_chunk]
            lines.append(f"[Scout Chunk {cid}]:\n{text}")
        header = "[Phase 1 Scout Evidence — broad context for gap-filling]\n\n"
        return header + "\n\n---\n\n".join(lines)

    # ------------------------------------------------------------------
    # OSPREY execution path
    # ------------------------------------------------------------------

    async def _run_osprey(self, question: str) -> PipelineResult:
        """Full OSPREY (M4) execution: Scout → Gate → EvidDecomp → Dispatch → Aggregate."""
        from multi_agent.scout import Scout

        t0 = time.monotonic()
        result = PipelineResult(question=question)

        decomposer, dispatcher, aggregator, cache = self._create_components()

        # Phase 1: Scout (3 loops, full question)
        scout = Scout(
            llm_client=self.llm,
            tools=self.tools,
            evidence_cache=cache,
            max_loops=self.scout_max_loops,
            max_token_budget=self.max_token_budget,
            confidence_threshold=self.scout_confidence_threshold,
            prompt_path=self.scout_prompt,
            verbose=self.verbose,
        )

        scout_result = await scout.scout(question)
        result.scout_answer = scout_result.answer
        result.scout_confidence = scout_result.confidence
        result.scout_chunks = scout_result.chunks

        # Record scout as agent at sentinel index -1
        if scout_result.agent_result is not None:
            result.agent_results[-1] = scout_result.agent_result

        # Phase 2: Confidence Gate
        if scout_result.is_confident:
            result.final_answer = scout_result.answer
            result.osprey_fast_exit = True
            result.question_type = "single_hop"
            result.num_sub_questions = 1
            result.num_waves = 1
            result.total_tokens = (
                scout_result.agent_result.total_tokens
                if scout_result.agent_result else 0
            )
            result.wall_clock_seconds = time.monotonic() - t0
            logger.info(
                "OSPREY fast-exit (conf=%.2f): '%s' → '%s'",
                scout_result.confidence, question[:60], result.final_answer[:60],
            )
            return result

        # Phase 3: Evidence-Aware Decomposition
        if self.osprey_decomposer_prompt:
            plan = await decomposer.decompose_with_evidence(
                question,
                scout_result.chunks,
                scout_result.answer,
                osprey_prompt_path=self.osprey_decomposer_prompt,
            )
        else:
            plan = await decomposer.decompose(question)

        result.decomposition = plan
        result.question_type = plan.question_type
        result.num_sub_questions = len(plan.sub_questions)

        if self.verbose:
            print(
                f"\nOSPREY Decomposition: {plan.question_type}, "
                f"{len(plan.sub_questions)} sub-Qs "
                f"(scout conf={scout_result.confidence:.2f})"
            )
            for sq in plan.sub_questions:
                deps = f" (depends on {sq.depends_on})" if sq.depends_on else ""
                print(f"  SQ-{sq.index}: {sq.text}{deps}")

        # Phase 4: Dispatch with global scout evidence injected into ALL agents
        global_chain_evidence = self._build_scout_evidence_str(scout_result)
        agent_results, num_waves = await dispatcher.dispatch(
            plan,
            original_question=question,
            global_chain_evidence=global_chain_evidence,
        )
        result.agent_results.update(agent_results)
        result.num_waves = num_waves

        # Token counting (scout + agents)
        scout_tokens = (
            scout_result.agent_result.total_tokens
            if scout_result.agent_result else 0
        )
        agent_tokens = sum(ar.total_tokens for ar in agent_results.values())

        # Phase 5: Aggregation (with scout chunks as pool prefix)
        if self.skip_aggregator:
            last_idx = max(agent_results.keys())
            result.final_answer = agent_results[last_idx].answer
            agg_tokens = 0
        else:
            final_answer, agg_tokens = await aggregator.aggregate(
                question,
                plan,
                agent_results,
                scout_chunks=scout_result.chunks,
            )
            result.final_answer = final_answer
        result.aggregator_tokens = agg_tokens

        result.cache_analytics = cache.compute_analytics_sync()
        result.total_tokens = scout_tokens + agent_tokens + agg_tokens
        result.wall_clock_seconds = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # Standard execution path (M1/M2/M3)
    # ------------------------------------------------------------------

    async def run(self, question: str) -> PipelineResult:
        """Run the full multi-agent pipeline for one question.

        Routes to OSPREY (_run_osprey) if enable_osprey=True, otherwise
        runs the standard decompose → dispatch → aggregate pipeline.
        """
        if self.enable_osprey:
            return await self._run_osprey(question)

        t0 = time.monotonic()
        result = PipelineResult(question=question)

        decomposer, dispatcher, aggregator, cache = self._create_components()

        try:
            # --- Phase 1: Decomposition ---
            if self.enable_decomposer:
                plan = await decomposer.decompose(question)
            else:
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
            agent_results, num_waves = await dispatcher.dispatch(
                plan, original_question=question
            )
            result.agent_results = agent_results
            result.num_waves = num_waves

            agent_tokens = sum(ar.total_tokens for ar in agent_results.values())

            # --- Phase 3: Aggregation ---
            if self.skip_aggregator:
                last_idx = max(agent_results.keys())
                result.final_answer = agent_results[last_idx].answer
                agg_tokens = 0
            else:
                final_answer, agg_tokens = await aggregator.aggregate(
                    question, plan, agent_results
                )
                result.final_answer = final_answer
            result.aggregator_tokens = agg_tokens

            result.cache_analytics = cache.compute_analytics_sync()
            result.total_tokens = agent_tokens + agg_tokens

        except Exception as exc:
            logger.error("Pipeline error for '%s': %s", question[:60], exc)
            result.error = str(exc)
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
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, self.run(question)).result()
        else:
            return asyncio.run(self.run(question))
