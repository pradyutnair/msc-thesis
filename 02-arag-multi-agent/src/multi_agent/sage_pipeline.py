"""SAGE pipeline: Planner -> Retriever(s) -> Verifier -> [loop] -> Synthesizer.

Architecture:
  1. Planner: analyzes question, generates retrieval tasks with specific queries
  2. Retriever(s): A-RAG agents with search_and_read tool, dispatched per plan
  3. Verifier: checks evidence sufficiency, filters noise, identifies gaps
  4. Synthesizer: clean synthesis from verified evidence only

Key differences from OSPREY/M4:
  - No Scout phase (Planner replaces it)
  - search_and_read tool eliminates reads=0 failure mode
  - Verifier feedback loop for adaptive retrieval depth
  - Synthesizer receives clean evidence (no search history)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.planner import Planner, SagePlan, RetrievalTask
from multi_agent.verifier import Verifier
from multi_agent.synthesizer import Synthesizer
from multi_agent.search_agent import SearchAgent
from multi_agent.types import (
    AgentResult,
    DecompositionPlan,
    PipelineResult,
    SubQuestion,
)

logger = logging.getLogger(__name__)


class SagePipeline:
    """SAGE: Planner -> Retriever(s) -> Verifier -> [gap loop] -> Synthesizer."""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: ToolRegistry,
        config: Config | None = None,
    ):
        self.llm = llm_client
        self.base_tools = tools
        self.config = config or Config()

        ma_cfg = self.config.get("multi_agent", {}) or {}
        self.retriever_max_loops = ma_cfg.get("sage_retriever_max_loops", 2)
        self.max_concurrent = ma_cfg.get("sage_max_concurrent", 3)
        self.max_verification_rounds = ma_cfg.get("sage_max_verification_rounds", 1)
        self.max_token_budget = ma_cfg.get("max_token_budget", 64000)
        self.verbose = ma_cfg.get("verbose", False)

        self.planner = Planner(
            llm_client=self.llm,
            prompt_path=ma_cfg.get("sage_planner_prompt"),
        )
        self.verifier = Verifier(
            llm_client=self.llm,
            prompt_path=ma_cfg.get("sage_verifier_prompt"),
        )
        self.synthesizer = Synthesizer(
            llm_client=self.llm,
            prompt_path=ma_cfg.get("sage_synth_prompt"),
        )

        self.retriever_prompt = ma_cfg.get("sage_retriever_prompt")
        self._sage_tools = self._build_sage_tools()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    # ------------------------------------------------------------------
    # Tool registry for SAGE retrievers
    # ------------------------------------------------------------------

    def _build_sage_tools(self) -> ToolRegistry:
        """Build retriever tool registry with search_and_read.

        Replaces standalone keyword_search/semantic_search with the combined
        search_and_read tool, ensuring every search reads chunk text.
        """
        from arag.tools.search_and_read import SearchAndReadTool
        from arag.tools.finish import FinishTool

        keyword_tool = self.base_tools.get("keyword_search")
        semantic_tool = self.base_tools.get("semantic_search")
        read_tool = self.base_tools.get("read_chunk")

        sage_reg = ToolRegistry()
        sage_reg.register(
            SearchAndReadTool(keyword_tool, semantic_tool, read_tool)
        )
        sage_reg.register(read_tool)  # Keep read_chunk for targeted follow-up
        sage_reg.register(FinishTool())
        return sage_reg

    # ------------------------------------------------------------------
    # Retriever dispatch
    # ------------------------------------------------------------------

    async def _run_retriever(
        self,
        task: RetrievalTask,
        resolved_answers: dict[int, str],
        question: str,
    ) -> AgentResult:
        """Run a single retriever agent for one task."""
        async with self._semaphore:
            # Resolve placeholders in query and goal
            query = task.query
            goal = task.goal
            for dep_id in task.depends_on:
                placeholder = f"[answer_{dep_id}]"
                answer = resolved_answers.get(dep_id, "unknown")
                query = query.replace(placeholder, answer)
                goal = goal.replace(placeholder, answer)

            # Encode task info in sub_question text for the retriever prompt
            task_text = (
                f"Goal: {goal}\n"
                f"Suggested search: {query} (method: {task.search_method})"
            )

            sq = SubQuestion(
                index=task.id,
                text=task_text,
                search_hints=[query],
                depends_on=task.depends_on,
            )

            agent = SearchAgent(
                llm_client=self.llm,
                tools=self._sage_tools,
                max_loops=self.retriever_max_loops,
                max_token_budget=self.max_token_budget,
                prompt_path=self.retriever_prompt,
                verbose=self.verbose,
            )

            result = await agent.run(
                sub_question=sq,
                original_question=question,
            )

            return result

    async def _dispatch_tasks(
        self,
        plan: SagePlan,
        question: str,
    ) -> tuple[dict[int, AgentResult], list[dict]]:
        """Dispatch retriever agents according to plan structure."""
        agent_results: dict[int, AgentResult] = {}
        all_chunks: list[dict] = []
        resolved_answers: dict[int, str] = {}

        if plan.question_type == "comparison":
            # All tasks in parallel
            coros = [
                self._run_retriever(task, {}, question)
                for task in plan.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    logger.error("Retriever for task %d failed: %s", task.id, result)
                    result = AgentResult(
                        sub_question_index=task.id, answer="", error=str(result),
                    )
                agent_results[task.id] = result
                resolved_answers[task.id] = result.answer
                for chunk in result.retrieved_chunks:
                    chunk["task_id"] = task.id
                all_chunks.extend(result.retrieved_chunks)

        elif plan.question_type == "bridge":
            # Sequential dispatch respecting dependencies
            remaining = {t.id: t for t in plan.tasks}
            completed: set[int] = set()

            while remaining:
                ready = [
                    t for t in remaining.values()
                    if all(d in completed for d in t.depends_on)
                ]
                if not ready:
                    # Deadlock safety — run remaining anyway
                    ready = list(remaining.values())

                # Run ready tasks (parallel if multiple at same level)
                if len(ready) == 1:
                    result = await self._run_retriever(
                        ready[0], resolved_answers, question,
                    )
                    if isinstance(result, Exception):
                        result = AgentResult(
                            sub_question_index=ready[0].id,
                            answer="",
                            error=str(result),
                        )
                    results_batch = [(ready[0], result)]
                else:
                    coros = [
                        self._run_retriever(t, dict(resolved_answers), question)
                        for t in ready
                    ]
                    raw_results = await asyncio.gather(
                        *coros, return_exceptions=True,
                    )
                    results_batch = list(zip(ready, raw_results))

                for task, result in results_batch:
                    if isinstance(result, Exception):
                        result = AgentResult(
                            sub_question_index=task.id,
                            answer="",
                            error=str(result),
                        )
                    agent_results[task.id] = result
                    resolved_answers[task.id] = result.answer
                    completed.add(task.id)
                    del remaining[task.id]
                    for chunk in result.retrieved_chunks:
                        chunk["task_id"] = task.id
                    all_chunks.extend(result.retrieved_chunks)

        else:  # single
            task = plan.tasks[0]
            result = await self._run_retriever(task, {}, question)
            if isinstance(result, Exception):
                result = AgentResult(
                    sub_question_index=task.id, answer="", error=str(result),
                )
            agent_results[task.id] = result
            for chunk in result.retrieved_chunks:
                chunk["task_id"] = task.id
            all_chunks.extend(result.retrieved_chunks)

        return agent_results, all_chunks

    # ------------------------------------------------------------------
    # Gap retrieval
    # ------------------------------------------------------------------

    async def _run_gap_retrieval(
        self,
        gaps: list[dict],
        question: str,
        gap_round: int,
    ) -> list[dict]:
        """Run gap-filling retrievers for identified evidence gaps."""
        gap_chunks: list[dict] = []

        for i, gap in enumerate(gaps[:3]):  # Max 3 gap queries
            task = RetrievalTask(
                id=100 + gap_round * 10 + i,
                query=gap.get("query", ""),
                search_method=gap.get("method", "keyword"),
                goal=gap.get("description", gap.get("query", "")),
            )

            try:
                result = await self._run_retriever(task, {}, question)
                for chunk in result.retrieved_chunks:
                    chunk["task_id"] = task.id
                gap_chunks.extend(result.retrieved_chunks)
            except Exception as exc:
                logger.error("Gap retriever %d failed: %s", task.id, exc)

        return gap_chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plan_to_decomposition(plan: SagePlan) -> DecompositionPlan:
        """Convert SagePlan to DecompositionPlan for PipelineResult compat."""
        q_type_map = {
            "single": "single_hop",
            "bridge": "bridge",
            "comparison": "comparison",
        }
        sub_questions = [
            SubQuestion(
                index=t.id,
                text=t.goal,
                search_hints=[t.query],
                depends_on=t.depends_on,
            )
            for t in plan.tasks
        ]
        edges = []
        for t in plan.tasks:
            for dep in t.depends_on:
                edges.append((dep, t.id))

        return DecompositionPlan(
            question_type=q_type_map.get(plan.question_type, "single_hop"),
            sub_questions=sub_questions,
            dependency_edges=edges,
            raw_llm_output=plan.raw_llm_output,
            parse_retries=plan.parse_retries,
        )

    @staticmethod
    def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
        """Remove duplicate chunks by ID."""
        seen: set[str] = set()
        unique: list[dict] = []
        for chunk in chunks:
            cid = str(chunk.get("id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(chunk)
        return unique

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def run(self, question: str) -> PipelineResult:
        """Run the full SAGE pipeline for one question."""
        t0 = time.monotonic()
        result = PipelineResult(question=question)

        try:
            # Phase 1: Plan
            plan = await self.planner.plan(question)
            result.decomposition = self._plan_to_decomposition(plan)
            q_type_map = {
                "single": "single_hop",
                "bridge": "bridge",
                "comparison": "comparison",
            }
            result.question_type = q_type_map.get(
                plan.question_type, "single_hop"
            )
            result.num_sub_questions = len(plan.tasks)

            if self.verbose:
                print(
                    f"\nSAGE Plan: {plan.question_type}, "
                    f"{len(plan.tasks)} tasks"
                )
                for t in plan.tasks:
                    deps = f" (depends on {t.depends_on})" if t.depends_on else ""
                    print(
                        f"  Task {t.id}: {t.goal} "
                        f"[{t.search_method}] "
                        f"query='{t.query}'{deps}"
                    )

            # Phase 2: Dispatch retrievers
            agent_results, all_chunks = await self._dispatch_tasks(
                plan, question,
            )
            result.agent_results = agent_results
            all_chunks = self._deduplicate_chunks(all_chunks)

            if self.verbose:
                print(f"  Retrieved {len(all_chunks)} unique chunks")

            # Compute waves for logging
            if plan.question_type == "comparison":
                result.num_waves = 1
            elif plan.question_type == "bridge":
                # Count sequential dependency levels
                max_depth = 0
                for t in plan.tasks:
                    depth = 0
                    visiting = t.depends_on[:]
                    visited: set[int] = set()
                    while visiting:
                        depth += 1
                        next_visit: list[int] = []
                        for d in visiting:
                            if d not in visited:
                                visited.add(d)
                                for tt in plan.tasks:
                                    if tt.id == d:
                                        next_visit.extend(tt.depends_on)
                        visiting = next_visit
                    max_depth = max(max_depth, depth)
                result.num_waves = max_depth + 1
            else:
                result.num_waves = 1

            # Phase 3: Verify
            verification = await self.verifier.verify(
                question, plan, all_chunks,
            )

            if self.verbose:
                status = "SUFFICIENT" if verification.sufficient else "INSUFFICIENT"
                print(f"  Verification: {status}")
                if verification.gaps:
                    print(f"  Gaps: {len(verification.gaps)}")

            # Phase 3b: Gap retrieval loop
            for gap_round in range(self.max_verification_rounds):
                if verification.sufficient or not verification.gaps:
                    break

                if self.verbose:
                    print(f"  Gap retrieval round {gap_round + 1}...")

                gap_chunks = await self._run_gap_retrieval(
                    verification.gaps, question, gap_round,
                )
                all_chunks.extend(gap_chunks)
                all_chunks = self._deduplicate_chunks(all_chunks)

                verification = await self.verifier.verify(
                    question, plan, all_chunks,
                )

                if self.verbose:
                    status = "SUFFICIENT" if verification.sufficient else "INSUFFICIENT"
                    print(f"  Re-verification: {status}")

            # Phase 4: Synthesize from verified evidence
            answer, synth_cost = await self.synthesizer.synthesize(
                question, verification.verified_chunks,
                agent_results=agent_results,
                expected_answer_type=getattr(plan, "expected_answer_type", None),
            )
            result.final_answer = answer

            # Token counting
            agent_tokens = sum(
                ar.total_tokens for ar in agent_results.values()
            )
            result.total_tokens = agent_tokens
            result.aggregator_tokens = (
                int(synth_cost * 1_000_000) if synth_cost > 0 else 0
            )

        except Exception as exc:
            logger.error(
                "SAGE pipeline error for '%s': %s", question[:60], exc,
            )
            result.error = str(exc)
            if result.agent_results:
                best = max(
                    result.agent_results.values(),
                    key=lambda r: len(r.answer) if r.answer else 0,
                )
                result.final_answer = best.answer

        result.wall_clock_seconds = time.monotonic() - t0
        return result

    def run_sync(self, question: str) -> PipelineResult:
        """Synchronous wrapper."""
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
