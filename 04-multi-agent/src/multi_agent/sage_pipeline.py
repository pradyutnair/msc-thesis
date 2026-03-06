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
import re
import time
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.planner import Planner, SagePlan, RetrievalTask
from multi_agent.search_agent import SearchAgent
from multi_agent.synthesizer import Synthesizer, find_best_evidence_backed_answer
from multi_agent.types import (
    AgentResult,
    DecompositionPlan,
    PipelineResult,
    SubQuestion,
)
from multi_agent.verifier import VerificationResult, Verifier

logger = logging.getLogger(__name__)

_REFUSAL_PATTERNS = [
    "cannot be determined",
    "insufficient information",
    "not mentioned",
    "no evidence",
    "unable to determine",
    "no information",
    "cannot determine",
    "not enough information",
    "information is not available",
    "not provided in",
    "no relevant information",
    "not available",
]


def _is_empty_or_refusal(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    lower = answer.lower().strip()
    return any(pattern in lower for pattern in _REFUSAL_PATTERNS)


def _is_invalid_answer_format(answer: str) -> bool:
    ans = (answer or "").strip()
    if not ans:
        return True
    if re.search(r"<[^>]+>", ans):
        return True
    if ans.endswith(":"):
        return True
    return False


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
        self.retriever_max_loops = int(ma_cfg.get("sage_retriever_max_loops", 2))
        self.max_concurrent = int(ma_cfg.get("sage_max_concurrent", 3))
        self.max_verification_rounds = int(ma_cfg.get("sage_max_verification_rounds", 1))
        self.max_token_budget = int(ma_cfg.get("max_token_budget", 64000))
        self.verbose = bool(ma_cfg.get("verbose", False))

        self.fail_open_on_verifier_error = bool(
            ma_cfg.get("sage_fail_open_on_verifier_error", False)
        )
        self.dual_retrieval_on_low_evidence = bool(
            ma_cfg.get("sage_dual_retrieval_on_low_evidence", False)
        )
        self.low_evidence_threshold = int(ma_cfg.get("sage_low_evidence_threshold", 2))
        self.dual_retrieval_top_k = int(ma_cfg.get("sage_dual_retrieval_top_k", 4))

        self.extract_evidence = bool(ma_cfg.get("sage_extract_evidence", False))
        self.extract_max_bullets = int(ma_cfg.get("sage_extract_max_bullets", 3))
        self.extract_prompt = ma_cfg.get(
            "sage_extract_prompt",
            "src/multi_agent/m5/prompts/extract_evidence.txt",
        )

        self.adaptive_dependency_loops = bool(
            ma_cfg.get("sage_adaptive_dependency_loops", True)
        )

        self.retry_pass_enabled = bool(ma_cfg.get("sage_retry_pass_enabled", False))
        self.retry_trigger_policy = str(
            ma_cfg.get("sage_retry_trigger_policy", "default_musique_v1")
        )
        self.retry_max_passes = max(1, int(ma_cfg.get("sage_retry_max_passes", 2)))
        self.retry_extra_loops = int(ma_cfg.get("sage_retry_extra_loops", 1))
        self.retry_verification_rounds = int(
            ma_cfg.get("sage_retry_verification_rounds", 2)
        )

        self.planner = Planner(
            llm_client=self.llm,
            prompt_path=ma_cfg.get("sage_planner_prompt"),
        )
        self.verifier = Verifier(
            llm_client=self.llm,
            prompt_path=ma_cfg.get("sage_verifier_prompt"),
            fail_open_on_error=self.fail_open_on_verifier_error,
        )
        self.synthesizer = Synthesizer(
            llm_client=self.llm,
            prompt_path=ma_cfg.get("sage_synth_prompt"),
        )

        self.retriever_prompt = ma_cfg.get("sage_retriever_prompt")
        self._sage_tools = self._build_sage_tools()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    def _build_sage_tools(self) -> ToolRegistry:
        """Build retriever tool registry with search_and_read."""
        from arag.tools.finish import FinishTool
        from arag.tools.search_and_read import SearchAndReadTool

        keyword_tool = self.base_tools.get("keyword_search")
        semantic_tool = self.base_tools.get("semantic_search")
        read_tool = self.base_tools.get("read_chunk")

        sage_reg = ToolRegistry()
        sage_reg.register(SearchAndReadTool(keyword_tool, semantic_tool, read_tool))
        sage_reg.register(read_tool)
        sage_reg.register(FinishTool())
        return sage_reg

    @staticmethod
    def _agent_has_support(result: AgentResult) -> bool:
        return (
            bool(result.evidence_doc_ids)
            or bool(result.retrieved_chunks)
            or result.evidence_count > 0
        ) and not result.unsupported_answer

    @staticmethod
    def _resolve_task_placeholders(
        task: RetrievalTask,
        resolved_answers: dict[int, dict[str, Any]],
    ) -> tuple[str, str, list[int]]:
        query = task.query
        goal = task.goal
        unresolved_dependencies: list[int] = []

        for dep_id in task.depends_on:
            placeholder = f"[answer_{dep_id}]"
            dep_meta = resolved_answers.get(dep_id, {})
            answer = str(dep_meta.get("answer", "")).strip()
            supported = bool(dep_meta.get("supported", False))

            if supported and answer:
                query = query.replace(placeholder, answer)
                goal = goal.replace(placeholder, answer)
            else:
                unresolved_dependencies.append(dep_id)

        return query, goal, unresolved_dependencies

    async def _run_retriever(
        self,
        task: RetrievalTask,
        resolved_answers: dict[int, dict[str, Any]],
        question: str,
        max_loops_override: int | None = None,
    ) -> AgentResult:
        """Run a single retriever agent for one task."""
        async with self._semaphore:
            query, goal, unresolved_dependencies = self._resolve_task_placeholders(
                task,
                resolved_answers,
            )

            task_text = (
                f"Goal: {goal}\n"
                f"Suggested search: {query} (method: {task.search_method})"
            )
            if unresolved_dependencies:
                task_text += (
                    "\nDependency status: unresolved prior tasks "
                    f"{unresolved_dependencies}. Recover missing dependency facts first."
                )

            sq = SubQuestion(
                index=task.id,
                text=task_text,
                search_hints=[query],
                depends_on=task.depends_on,
            )

            loops = max_loops_override if max_loops_override is not None else self.retriever_max_loops
            if self.adaptive_dependency_loops and task.depends_on:
                loops += 1

            agent = SearchAgent(
                llm_client=self.llm,
                tools=self._sage_tools,
                max_loops=loops,
                max_token_budget=self.max_token_budget,
                prompt_path=self.retriever_prompt,
                verbose=self.verbose,
                dual_retrieval_on_low_evidence=self.dual_retrieval_on_low_evidence,
                low_evidence_threshold=self.low_evidence_threshold,
                dual_retrieval_top_k=self.dual_retrieval_top_k,
                extract_evidence=self.extract_evidence,
                extract_prompt_path=self.extract_prompt,
                extract_max_bullets=self.extract_max_bullets,
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
        max_loops_override: int | None = None,
    ) -> tuple[dict[int, AgentResult], list[dict], dict[int, dict[str, Any]]]:
        """Dispatch retriever agents according to plan structure."""
        agent_results: dict[int, AgentResult] = {}
        all_chunks: list[dict] = []
        resolved_answers: dict[int, dict[str, Any]] = {}

        if plan.question_type == "comparison":
            coros = [
                self._run_retriever(task, {}, question, max_loops_override=max_loops_override)
                for task in plan.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    logger.error("Retriever for task %d failed: %s", task.id, result)
                    result = AgentResult(
                        sub_question_index=task.id,
                        answer="",
                        error=str(result),
                    )

                agent_results[task.id] = result
                resolved_answers[task.id] = {
                    "answer": result.answer,
                    "supported": self._agent_has_support(result),
                }
                for chunk in result.retrieved_chunks:
                    chunk["task_id"] = task.id
                all_chunks.extend(result.retrieved_chunks)

        elif plan.question_type == "bridge":
            remaining = {t.id: t for t in plan.tasks}
            completed: set[int] = set()

            while remaining:
                ready = [
                    t for t in remaining.values()
                    if all(d in completed for d in t.depends_on)
                ]
                if not ready:
                    ready = list(remaining.values())

                if len(ready) == 1:
                    result = await self._run_retriever(
                        ready[0],
                        resolved_answers,
                        question,
                        max_loops_override=max_loops_override,
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
                        self._run_retriever(
                            t,
                            dict(resolved_answers),
                            question,
                            max_loops_override=max_loops_override,
                        )
                        for t in ready
                    ]
                    raw_results = await asyncio.gather(
                        *coros,
                        return_exceptions=True,
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
                    resolved_answers[task.id] = {
                        "answer": result.answer,
                        "supported": self._agent_has_support(result),
                    }
                    completed.add(task.id)
                    del remaining[task.id]
                    for chunk in result.retrieved_chunks:
                        chunk["task_id"] = task.id
                    all_chunks.extend(result.retrieved_chunks)

        else:
            task = plan.tasks[0]
            result = await self._run_retriever(
                task,
                {},
                question,
                max_loops_override=max_loops_override,
            )
            if isinstance(result, Exception):
                result = AgentResult(
                    sub_question_index=task.id,
                    answer="",
                    error=str(result),
                )
            agent_results[task.id] = result
            resolved_answers[task.id] = {
                "answer": result.answer,
                "supported": self._agent_has_support(result),
            }
            for chunk in result.retrieved_chunks:
                chunk["task_id"] = task.id
            all_chunks.extend(result.retrieved_chunks)

        return agent_results, all_chunks, resolved_answers

    async def _run_gap_retrieval(
        self,
        gaps: list[dict],
        question: str,
        plan: SagePlan,
        resolved_answers: dict[int, dict[str, Any]],
        gap_round: int,
        max_loops_override: int | None = None,
    ) -> list[dict]:
        """Run gap-filling retrievers for identified evidence gaps."""
        gap_chunks: list[dict] = []

        resolved_entities = [
            meta["answer"]
            for _, meta in sorted(resolved_answers.items())
            if meta.get("supported") and str(meta.get("answer", "")).strip()
        ]
        resolved_suffix = ", ".join(resolved_entities[:2])

        for i, gap in enumerate(gaps[:3]):
            base_query = str(gap.get("query", "")).strip() or question
            if resolved_suffix and resolved_suffix.lower() not in base_query.lower():
                base_query = f"{base_query}, {resolved_suffix}"
            if plan.expected_answer_type and plan.expected_answer_type != "entity":
                if plan.expected_answer_type.lower() not in base_query.lower():
                    base_query = f"{base_query}, {plan.expected_answer_type}"

            task = RetrievalTask(
                id=100 + gap_round * 10 + i,
                query=base_query,
                search_method=gap.get("method", "keyword"),
                goal=gap.get("description", gap.get("query", "")),
            )

            try:
                result = await self._run_retriever(
                    task,
                    {},
                    question,
                    max_loops_override=max_loops_override,
                )
                for chunk in result.retrieved_chunks:
                    chunk["task_id"] = task.id
                gap_chunks.extend(result.retrieved_chunks)
            except Exception as exc:
                logger.error("Gap retriever %d failed: %s", task.id, exc)

        return gap_chunks

    @staticmethod
    def _plan_to_decomposition(plan: SagePlan) -> DecompositionPlan:
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
        seen: set[str] = set()
        unique: list[dict] = []
        for chunk in chunks:
            cid = str(chunk.get("id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(chunk)
        return unique

    @staticmethod
    def _build_extracted_evidence_map(
        agent_results: dict[int, AgentResult],
    ) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for tid, ar in agent_results.items():
            if ar.extracted_evidence:
                out[int(tid)] = list(ar.extracted_evidence)
        return out

    @staticmethod
    def _count_total_evidence_ids(agent_results: dict[int, AgentResult]) -> int:
        return sum(len(ar.evidence_doc_ids or []) for ar in agent_results.values())

    def _compute_retry_reasons(
        self,
        result: PipelineResult,
        plan: SagePlan,
        verification: VerificationResult,
        resolved_answers: dict[int, dict[str, Any]],
    ) -> list[str]:
        reasons: list[str] = []

        if _is_empty_or_refusal(result.final_answer) or _is_invalid_answer_format(result.final_answer):
            reasons.append("final_answer_invalid_or_refusal")

        has_unsupported_dependency = False
        for task in plan.tasks:
            if not task.depends_on:
                continue
            for dep_id in task.depends_on:
                dep_meta = resolved_answers.get(dep_id, {})
                if not dep_meta.get("supported", False):
                    has_unsupported_dependency = True
                    break
            if has_unsupported_dependency:
                break
        if has_unsupported_dependency:
            reasons.append("unsupported_dependency_answer")

        if not verification.parse_ok:
            reasons.append("verifier_parse_failed")
        if not verification.sufficient:
            reasons.append("verifier_insufficient")

        total_evidence_ids = self._count_total_evidence_ids(result.agent_results)
        if result.num_sub_questions >= 3 and total_evidence_ids <= 1:
            reasons.append("deep_chain_low_coverage")

        deduped: list[str] = []
        seen = set()
        for reason in reasons:
            if reason not in seen:
                seen.add(reason)
                deduped.append(reason)
        return deduped

    async def _run_single_pass(
        self,
        question: str,
        pass_id: str,
        retriever_loops_override: int | None = None,
        verification_rounds_override: int | None = None,
    ) -> tuple[
        PipelineResult,
        SagePlan | None,
        VerificationResult | None,
        dict[int, dict[str, Any]],
    ]:
        t0 = time.monotonic()
        result = PipelineResult(question=question, pass_id=pass_id)

        verification: VerificationResult | None = None
        plan: SagePlan | None = None
        resolved_answers: dict[int, dict[str, Any]] = {}

        try:
            plan = await self.planner.plan(question)
            result.decomposition = self._plan_to_decomposition(plan)
            q_type_map = {
                "single": "single_hop",
                "bridge": "bridge",
                "comparison": "comparison",
            }
            result.question_type = q_type_map.get(plan.question_type, "single_hop")
            result.num_sub_questions = len(plan.tasks)

            if self.verbose:
                print(f"\nSAGE Plan ({pass_id}): {plan.question_type}, {len(plan.tasks)} tasks")
                for t in plan.tasks:
                    deps = f" (depends on {t.depends_on})" if t.depends_on else ""
                    print(f"  Task {t.id}: {t.goal} [{t.search_method}] query='{t.query}'{deps}")

            agent_results, all_chunks, resolved_answers = await self._dispatch_tasks(
                plan,
                question,
                max_loops_override=retriever_loops_override,
            )
            result.agent_results = agent_results
            all_chunks = self._deduplicate_chunks(all_chunks)

            if plan.question_type == "comparison":
                result.num_waves = 1
            elif plan.question_type == "bridge":
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

            extracted_map = self._build_extracted_evidence_map(agent_results)
            verification = await self.verifier.verify(
                question,
                plan,
                all_chunks,
                extracted_evidence_by_task=extracted_map,
            )

            max_verif_rounds = (
                verification_rounds_override
                if verification_rounds_override is not None
                else self.max_verification_rounds
            )

            for gap_round in range(max_verif_rounds):
                if verification.sufficient or not verification.gaps:
                    break

                gap_chunks = await self._run_gap_retrieval(
                    verification.gaps,
                    question,
                    plan,
                    resolved_answers,
                    gap_round,
                    max_loops_override=retriever_loops_override,
                )
                all_chunks.extend(gap_chunks)
                all_chunks = self._deduplicate_chunks(all_chunks)

                verification = await self.verifier.verify(
                    question,
                    plan,
                    all_chunks,
                    extracted_evidence_by_task=extracted_map,
                )

            answer, synth_cost = await self.synthesizer.synthesize(
                question,
                verification.verified_chunks if verification else all_chunks,
                agent_results=agent_results,
                expected_answer_type=getattr(plan, "expected_answer_type", None),
            )
            result.final_answer = answer

            if _is_empty_or_refusal(answer) and agent_results:
                fallback = find_best_evidence_backed_answer(agent_results)
                if fallback:
                    logger.warning(
                        "Synthesis produced empty/refusal ('%s'), evidence-backed fallback: '%s'",
                        answer[:40],
                        fallback[:80],
                    )
                    result.final_answer = fallback

            result.total_tokens = sum(ar.total_tokens for ar in agent_results.values())
            result.aggregator_tokens = int(synth_cost * 1_000_000) if synth_cost > 0 else 0
            result.verifier_parse_ok = verification.parse_ok if verification else None

        except Exception as exc:
            logger.error("SAGE pipeline error for '%s': %s", question[:60], exc)
            result.error = str(exc)
            if result.agent_results:
                fallback = find_best_evidence_backed_answer(result.agent_results)
                if fallback:
                    result.final_answer = fallback

        result.wall_clock_seconds = time.monotonic() - t0
        return result, plan, verification, resolved_answers

    async def run(self, question: str) -> PipelineResult:
        """Run the full SAGE pipeline for one question."""
        pass1, plan1, verification1, resolved1 = await self._run_single_pass(
            question,
            pass_id="pass1",
        )

        if not self.retry_pass_enabled or self.retry_max_passes <= 1:
            pass1.retry_trigger_reasons = []
            return pass1

        if self.retry_trigger_policy != "default_musique_v1":
            pass1.retry_trigger_reasons = []
            return pass1

        if plan1 is None or verification1 is None:
            pass1.retry_trigger_reasons = ["pipeline_pass1_error"]
            return pass1

        retry_reasons = self._compute_retry_reasons(
            pass1,
            plan1,
            verification1,
            resolved1,
        )
        pass1.retry_trigger_reasons = retry_reasons

        if not retry_reasons:
            return pass1

        loops_pass2 = self.retriever_max_loops + max(self.retry_extra_loops, 0)
        verif_rounds_pass2 = max(self.max_verification_rounds, self.retry_verification_rounds)

        pass2, _, _, _ = await self._run_single_pass(
            question,
            pass_id="pass2",
            retriever_loops_override=loops_pass2,
            verification_rounds_override=verif_rounds_pass2,
        )
        pass2.retry_trigger_reasons = retry_reasons

        # Preserve total compute spent across both passes.
        pass2.wall_clock_seconds += pass1.wall_clock_seconds
        pass2.total_tokens += pass1.total_tokens
        pass2.aggregator_tokens += pass1.aggregator_tokens

        return pass2

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

        return asyncio.run(self.run(question))
