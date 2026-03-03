"""SAGE pipeline: Planner -> Retriever(s) -> Verifier -> [loop] -> Synthesizer.

Architecture:
  1. Planner: analyzes question, generates retrieval tasks with specific queries
  2. Retriever(s): A-RAG agents with search_and_read tool, dispatched per plan
  3. Verifier: checks evidence sufficiency, filters noise, identifies gaps
  4. Synthesizer: clean synthesis from verified evidence only

v9 additions:
  - Reranker (bge-reranker-v2-m3) integrated into search_and_read
  - Organized evidence by task in synthesizer
  - Chunk truncation at 800 chars

v9.2 additions:
  - Chunk inheritance: dependent agents in bridge questions receive parent
    agents' retrieved chunks as pre-loaded context, fixing the "wrong hop"
    problem where child agents fail to find info about parent-discovered entities.

v9.3 additions (DualRAG-inspired progressive knowledge accumulation):
  - Knowledge Summary: after each hop, build a structured context that tells
    the next agent the full original question, what has been found so far
    (entity chain), and what specifically the next hop needs to find.
  - Structured "Chain of Knowledge" format replaces flat chunk injection.
  - More generous inherited chunk parameters (10 chunks / 1000 chars).
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
from multi_agent.synthesizer import Synthesizer, find_best_evidence_backed_answer, find_any_useful_answer
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

        # v9: Reranker configuration
        self.reranker_enabled = bool(ma_cfg.get("sage_reranker_enabled", False))
        self.reranker_model = str(
            ma_cfg.get("sage_reranker_model", "BAAI/bge-reranker-v2-m3")
        )
        self.reranker_batch_size = int(ma_cfg.get("sage_reranker_batch_size", 16))
        self._reranker = None

        # v9: Evidence truncation
        self.chunk_max_chars = int(ma_cfg.get("sage_chunk_max_chars", 800))

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

    def _get_reranker(self):
        """Lazy-load reranker on first use."""
        if not self.reranker_enabled:
            return None
        if self._reranker is None:
            from multi_agent.reranker import Reranker
            self._reranker = Reranker(
                model_name=self.reranker_model,
                device="cpu",
                batch_size=self.reranker_batch_size,
            )
        return self._reranker

    def _build_sage_tools(self) -> ToolRegistry:
        """Build retriever tool registry with search_and_read."""
        from arag.tools.finish import FinishTool
        from arag.tools.search_and_read import SearchAndReadTool

        keyword_tool = self.base_tools.get("keyword_search")
        semantic_tool = self.base_tools.get("semantic_search")
        read_tool = self.base_tools.get("read_chunk")

        reranker = self._get_reranker()

        sage_reg = ToolRegistry()
        sage_reg.register(SearchAndReadTool(keyword_tool, semantic_tool, read_tool, reranker=reranker))
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

    @staticmethod
    def _format_inherited_chunks(chunks: list[dict], max_chunks: int = 10, max_chars: int = 1000) -> str:
        """Format inherited parent chunks as context text for child agents.

        v9.3: Increased from 5/500 to 10/1000 for better multi-hop coverage.
        """
        if not chunks:
            return ""
        lines = []
        seen_ids: set[str] = set()
        for chunk in chunks[:max_chunks]:
            cid = str(chunk.get("id", "?"))
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            text = chunk.get("text", "")[:max_chars]
            lines.append(f"[Doc {cid}]: {text}")
        return "\n".join(lines)

    @staticmethod
    def _build_knowledge_summary(
        question: str,
        task: "RetrievalTask",
        resolved_answers: dict[int, dict[str, Any]],
        plan_tasks: list["RetrievalTask"],
        inherited_chunks: list[dict] | None = None,
    ) -> str:
        """Build a DualRAG-inspired progressive knowledge summary for the next hop.

        This structured context tells the agent:
        1. The full original question (so it understands the end goal)
        2. What has been established so far (entity chain from prior hops)
        3. What SPECIFICALLY this hop needs to find
        4. Supporting documents from prior hops

        v9.3: Replaces flat chunk injection with structured knowledge accumulation.
        """
        parts: list[str] = []

        # Section 1: Chain of Knowledge header
        parts.append("## Chain of Knowledge (from prior hops)")
        parts.append(f"The original question is: {question}")
        parts.append("")

        # Section 2: What we've established so far (entity chain)
        hop_lines: list[str] = []
        # Build a map from task id -> task for easy lookup
        task_map = {t.id: t for t in plan_tasks}
        # Sort dependency chain in execution order
        completed_ids = sorted(
            tid for tid in resolved_answers
            if resolved_answers[tid].get("answer", "").strip()
        )
        for hop_num, tid in enumerate(completed_ids, start=1):
            dep_task = task_map.get(tid)
            dep_goal = dep_task.goal if dep_task else f"Task {tid}"
            dep_answer = resolved_answers[tid].get("answer", "").strip()
            supported = resolved_answers[tid].get("supported", False)
            status = dep_answer if dep_answer else "(no answer yet)"
            if not supported and dep_answer:
                status = f"{dep_answer} (unverified)"
            hop_lines.append(f"- Hop {hop_num}: {dep_goal} -> Found: {status}")

        if hop_lines:
            parts.append("What we've established so far:")
            parts.extend(hop_lines)
        else:
            parts.append("What we've established so far: (this is the first hop)")
        parts.append("")

        # Section 3: Current task's specific goal (with resolved placeholders)
        # Resolve placeholders in the goal for display
        resolved_goal = task.goal
        for dep_id in task.depends_on:
            placeholder = f"[answer_{dep_id}]"
            dep_meta = resolved_answers.get(dep_id, {})
            answer = str(dep_meta.get("answer", "")).strip()
            if answer:
                resolved_goal = resolved_goal.replace(placeholder, answer)
        parts.append(f"YOUR SPECIFIC TASK: {resolved_goal}")
        parts.append("")

        # Section 4: Supporting documents from prior hops
        if inherited_chunks:
            inherited_text = SagePipeline._format_inherited_chunks(inherited_chunks)
            if inherited_text:
                parts.append("Supporting documents from prior hops (may contain the answer):")
                parts.append("Read these carefully. If you find the answer, use finish() immediately.")
                parts.append("")
                parts.append(inherited_text)

        return "\n".join(parts)

    async def _run_retriever(
        self,
        task: RetrievalTask,
        resolved_answers: dict[int, dict[str, Any]],
        question: str,
        max_loops_override: int | None = None,
        inherited_chunks: list[dict] | None = None,
        **kwargs,
    ) -> AgentResult:
        """Run a single retriever agent for one task.

        Args:
            inherited_chunks: Chunks retrieved by parent/dependency agents.
                Included in the task text so the agent can use them directly
                if they contain the answer, avoiding redundant retrieval.
            **kwargs: Additional context, including:
                knowledge_context: Pre-built knowledge summary string (v9.3).
        """
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

            # v9.3: Inject structured Knowledge Summary (DualRAG-inspired)
            if inherited_chunks or task.depends_on:
                knowledge_context = kwargs.get("knowledge_context")
                if knowledge_context:
                    task_text += f"\n\n{knowledge_context}"
                    logger.info(
                        "Task %d: injected knowledge summary (%d prior hops, %d inherited chunks)",
                        task.id,
                        len([d for d in task.depends_on if d in resolved_answers]),
                        len(inherited_chunks or []),
                    )
                elif inherited_chunks:
                    # Fallback: if no knowledge_context provided, use basic chunk injection
                    inherited_text = self._format_inherited_chunks(inherited_chunks)
                    if inherited_text:
                        task_text += (
                            "\n\n## Evidence from prior tasks (check FIRST before searching)\n"
                            "The documents below were retrieved by earlier agents in the chain. "
                            "The answer to your goal may ALREADY be in these documents. "
                            "Read them carefully. If you find the answer, use finish() immediately.\n\n"
                            f"{inherited_text}"
                        )
                        logger.info(
                            "Task %d: injected %d inherited chunks from parent agents (fallback)",
                            task.id, len(inherited_chunks),
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

            # v9.2: Merge inherited chunks into result so they flow through
            # to verification and synthesis (the agent may not have re-retrieved them)
            if inherited_chunks:
                existing_ids = {str(c.get("id", "")) for c in result.retrieved_chunks}
                for chunk in inherited_chunks:
                    cid = str(chunk.get("id", ""))
                    if cid and cid not in existing_ids:
                        result.retrieved_chunks.append(chunk)
                        existing_ids.add(cid)

            return result


    async def _recovery_retrieval(
        self,
        task: RetrievalTask,
        resolved_answers: dict[int, dict[str, Any]],
        question: str,
        original_result: AgentResult,
    ) -> AgentResult:
        """v10 Ensemble Recovery: when a dependent agent fails, launch multi-strategy recovery.

        Instead of giving up when Agent 1 can't find info about entity X (from Agent 0),
        we try 3 different search strategies:
        1. Entity-only search: just search for X with minimal qualifiers
        2. Entity + context search: X + broader question context
        3. Semantic search: natural language goal as semantic query

        This leverages multi-agent collaboration: the first agent's failure informs
        the recovery agent's strategy.
        """
        # Extract the key entity from resolved dependencies
        entity_parts = []
        for dep_id in task.depends_on:
            dep_meta = resolved_answers.get(dep_id, {})
            answer = str(dep_meta.get("answer", "")).strip()
            if answer and dep_meta.get("supported", False):
                # Clean the answer - take just the entity name
                clean = re.sub(r"\*\*", "", answer)  # remove markdown bold
                clean = re.sub(r"\([^)]*\)", "", clean)  # remove parentheticals
                clean = clean.split(".")[0].strip()  # take first sentence
                clean = clean.split(",")[0].strip()  # take first clause
                if clean and len(clean) < 100:
                    entity_parts.append(clean)

        if not entity_parts:
            return original_result

        search_tool = self._sage_tools.get("search_and_read")
        read_tool = self._sage_tools.get("read_chunk")
        if search_tool is None or read_tool is None:
            return original_result

        from arag.core.context import AgentContext
        context = AgentContext()
        context.source_agent = task.id

        recovery_chunks: list[dict] = []
        recovery_chunk_ids: list[str] = []

        # Resolve the goal with placeholders filled
        goal = task.goal
        for dep_id in task.depends_on:
            placeholder = f"[answer_{dep_id}]"
            dep_meta = resolved_answers.get(dep_id, {})
            answer = str(dep_meta.get("answer", "")).strip()
            if answer:
                goal = goal.replace(placeholder, answer)

        # Strategy 1: Entity-only keyword search (broadest recall)
        for entity in entity_parts:
            try:
                result_text, log = search_tool.execute(
                    context=context, query=entity, method="keyword", top_k=5,
                )
                for cid in log.get("chunk_ids_read", []):
                    cid_s = str(cid)
                    text = ""
                    if read_tool is not None and hasattr(read_tool, "chunks_dict"):
                        text = read_tool.chunks_dict.get(cid_s, "")
                    if text and cid_s not in {str(c.get("id","")) for c in recovery_chunks}:
                        recovery_chunks.append({"id": cid_s, "text": text})
                        recovery_chunk_ids.append(cid_s)
            except Exception as exc:
                logger.debug("Recovery strategy 1 failed: %s", exc)

        # Strategy 2: Entity + goal-specific keywords
        query_parts = list(entity_parts)
        # Extract key terms from the goal (relationship words)
        goal_lower = goal.lower()
        for term in ["spouse", "wife", "husband", "father", "mother", "child", "son",
                      "daughter", "nationality", "born", "birthplace", "death", "died",
                      "director", "performer", "singer", "author", "capital", "country",
                      "city", "location", "parent", "married"]:
            if term in goal_lower:
                query_parts.append(term)
                break
        try:
            result_text, log = search_tool.execute(
                context=context,
                query=", ".join(query_parts[:4]),
                method="keyword",
                top_k=5,
            )
            for cid in log.get("chunk_ids_read", []):
                cid_s = str(cid)
                text = ""
                if read_tool is not None and hasattr(read_tool, "chunks_dict"):
                    text = read_tool.chunks_dict.get(cid_s, "")
                if text and cid_s not in {str(c.get("id","")) for c in recovery_chunks}:
                    recovery_chunks.append({"id": cid_s, "text": text})
                    recovery_chunk_ids.append(cid_s)
        except Exception as exc:
            logger.debug("Recovery strategy 2 failed: %s", exc)

        # Strategy 3: Semantic search with full goal text
        try:
            result_text, log = search_tool.execute(
                context=context,
                query=goal,
                method="semantic",
                top_k=5,
            )
            for cid in log.get("chunk_ids_read", []):
                cid_s = str(cid)
                text = ""
                if read_tool is not None and hasattr(read_tool, "chunks_dict"):
                    text = read_tool.chunks_dict.get(cid_s, "")
                if text and cid_s not in {str(c.get("id","")) for c in recovery_chunks}:
                    recovery_chunks.append({"id": cid_s, "text": text})
                    recovery_chunk_ids.append(cid_s)
        except Exception as exc:
            logger.debug("Recovery strategy 3 failed: %s", exc)

        if not recovery_chunks:
            logger.info("Recovery retrieval for task %d found no new evidence", task.id)
            return original_result

        # Now re-run the agent with the recovered evidence injected
        logger.info(
            "Recovery retrieval for task %d found %d new chunks, launching recovery agent",
            task.id, len(recovery_chunks),
        )

        # Build a recovery context that includes the found evidence
        recovery_context = "## Recovery Evidence (from multi-strategy search)\n"
        recovery_context += "The following documents were found using alternative search strategies.\n"
        recovery_context += "Read them carefully and extract the answer to your task.\n\n"
        for chunk in recovery_chunks[:8]:
            recovery_context += f"[Doc {chunk['id']}]:\n{chunk['text'][:2000]}\n\n"

        # Create a recovery sub-question with the evidence pre-loaded
        recovery_sq = SubQuestion(
            index=task.id,
            text=(
                f"Goal: {goal}\n"
                f"IMPORTANT: Use the recovery evidence below to find the answer. "
                f"If the answer is in the evidence, use finish() immediately.\n\n"
                f"{recovery_context}"
            ),
            search_hints=[", ".join(entity_parts)],
            depends_on=task.depends_on,
        )

        recovery_agent = SearchAgent(
            llm_client=self.llm,
            tools=self._sage_tools,
            max_loops=2,  # Quick pass - evidence is already provided
            max_token_budget=self.max_token_budget,
            prompt_path=self.retriever_prompt,
            verbose=self.verbose,
        )

        try:
            recovery_result = await recovery_agent.run(
                sub_question=recovery_sq,
                original_question=question,
            )
        except Exception as exc:
            logger.error("Recovery agent for task %d failed: %s", task.id, exc)
            # Still merge the raw recovery chunks into original result
            original_result.retrieved_chunks.extend(recovery_chunks)
            original_result.evidence_doc_ids.extend(recovery_chunk_ids)
            return original_result

        # Merge recovery results with original
        if recovery_result.answer and not _is_empty_or_refusal(recovery_result.answer):
            # Recovery found an answer - use it
            merged = recovery_result
            # Also include original chunks
            existing_ids = {str(c.get("id","")) for c in merged.retrieved_chunks}
            for chunk in original_result.retrieved_chunks:
                cid = str(chunk.get("id",""))
                if cid and cid not in existing_ids:
                    merged.retrieved_chunks.append(chunk)
                    existing_ids.add(cid)
            for chunk in recovery_chunks:
                cid = str(chunk.get("id",""))
                if cid and cid not in existing_ids:
                    merged.retrieved_chunks.append(chunk)
                    existing_ids.add(cid)
            return merged
        else:
            # Recovery didn't find answer either, but merge the chunks
            original_result.retrieved_chunks.extend(recovery_chunks)
            seen = set(original_result.evidence_doc_ids)
            for cid in recovery_chunk_ids:
                if cid not in seen:
                    original_result.evidence_doc_ids.append(cid)
                    seen.add(cid)
            return original_result

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
            # v9.2: Track chunks per task for inheritance to dependent agents
            chunks_by_task: dict[int, list[dict]] = {}

            while remaining:
                ready = [
                    t for t in remaining.values()
                    if all(d in completed for d in t.depends_on)
                ]
                if not ready:
                    ready = list(remaining.values())

                if len(ready) == 1:
                    task = ready[0]
                    # v9.2: Collect inherited chunks from all parent tasks
                    inherited = []
                    for dep_id in task.depends_on:
                        inherited.extend(chunks_by_task.get(dep_id, []))

                    # v9.3: Build knowledge summary for this hop
                    knowledge_ctx = None
                    if task.depends_on:
                        knowledge_ctx = self._build_knowledge_summary(
                            question=question,
                            task=task,
                            resolved_answers=resolved_answers,
                            plan_tasks=plan.tasks,
                            inherited_chunks=inherited if inherited else None,
                        )

                    result = await self._run_retriever(
                        task,
                        resolved_answers,
                        question,
                        max_loops_override=max_loops_override,
                        inherited_chunks=inherited if inherited else None,
                        knowledge_context=knowledge_ctx,
                    )
                    if isinstance(result, Exception):
                        result = AgentResult(
                            sub_question_index=task.id,
                            answer="",
                            error=str(result),
                        )
                    results_batch = [(task, result)]
                else:
                    coros = []
                    for t in ready:
                        # v9.2: Collect inherited chunks from all parent tasks
                        inherited = []
                        for dep_id in t.depends_on:
                            inherited.extend(chunks_by_task.get(dep_id, []))

                        # v9.3: Build knowledge summary for this hop
                        knowledge_ctx = None
                        if t.depends_on:
                            knowledge_ctx = self._build_knowledge_summary(
                                question=question,
                                task=t,
                                resolved_answers=dict(resolved_answers),
                                plan_tasks=plan.tasks,
                                inherited_chunks=inherited if inherited else None,
                            )

                        coros.append(
                            self._run_retriever(
                                t,
                                dict(resolved_answers),
                                question,
                                max_loops_override=max_loops_override,
                                inherited_chunks=inherited if inherited else None,
                                knowledge_context=knowledge_ctx,
                            )
                        )
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

                    # v10 Ensemble Recovery: if dependent agent failed, try recovery
                    if (task.depends_on
                        and (result.evidence_count < 2 or result.unsupported_answer)
                        and not isinstance(result, Exception)):
                        logger.info(
                            "Task %d has low evidence (%d) or unsupported answer, "
                            "launching ensemble recovery...",
                            task.id, getattr(result, 'evidence_count', 0),
                        )
                        try:
                            result = await self._recovery_retrieval(
                                task, resolved_answers, question, result,
                            )
                        except Exception as rec_exc:
                            logger.error("Recovery for task %d failed: %s", task.id, rec_exc)

                    # v10 Ensemble Recovery: if dependent agent failed, try recovery
                    if (task.depends_on
                        and (result.evidence_count < 2 or result.unsupported_answer)):
                        logger.info(
                            "Task %d (single-ready) low evidence (%d), launching recovery...",
                            task.id, getattr(result, 'evidence_count', 0),
                        )
                        try:
                            result = await self._recovery_retrieval(
                                task, resolved_answers, question, result,
                            )
                        except Exception as rec_exc:
                            logger.error("Recovery for task %d failed: %s", task.id, rec_exc)

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
                    # v9.2: Store chunks for inheritance to downstream tasks
                    chunks_by_task[task.id] = list(result.retrieved_chunks)

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


    async def _answer_type_critic(
        self,
        question: str,
        answer: str,
        plan: SagePlan,
        agent_results: dict[int, AgentResult],
    ) -> str:
        """v10 Answer Type Critic: validate answer against question type and agent chain.

        For bridge questions: ensure the answer comes from the LAST hop, not intermediate.
        For comparison questions: ensure the answer matches what's being compared.
        For all types: ensure answer matches expected_answer_type.
        """
        if not answer or not plan:
            return answer

        answer_lower = answer.lower().strip()

        # Bridge question check: the answer should come from the last agent
        if plan.question_type == "bridge" and len(plan.tasks) >= 2:
            last_task_id = max(t.id for t in plan.tasks)
            last_agent = agent_results.get(last_task_id)

            if last_agent and last_agent.answer:
                last_answer = last_agent.answer.strip()
                last_clean = re.sub(r"\*\*", "", last_answer)
                last_clean = last_clean.split(".")[0].strip()

                # If the current answer looks like it's from an intermediate hop
                # (matches an earlier agent's answer but not the last), prefer last agent
                for task in plan.tasks:
                    if task.id == last_task_id:
                        continue
                    earlier_agent = agent_results.get(task.id)
                    if not earlier_agent or not earlier_agent.answer:
                        continue
                    earlier_answer = earlier_agent.answer.strip().lower()
                    if (answer_lower in earlier_answer or earlier_answer in answer_lower):
                        # Current answer matches an intermediate hop
                        if last_clean and not _is_empty_or_refusal(last_clean):
                            logger.info(
                                "Critic: answer '%s' matches intermediate hop, "
                                "preferring last agent answer '%s'",
                                answer[:40], last_clean[:40],
                            )
                            return Synthesizer.sanitize_answer(last_clean)

        # Comparison question check for yes/no
        if plan.question_type == "comparison" and plan.expected_answer_type == "yes_no":
            if answer_lower not in ("yes", "no"):
                # Try to extract yes/no from the answer
                if "yes" in answer_lower and "no" not in answer_lower:
                    return "yes"
                elif "no" in answer_lower and "yes" not in answer_lower:
                    return "no"

        return answer

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

            # v9: Truncate chunk text for more efficient context usage
            for chunk in all_chunks:
                if len(chunk.get("text", "")) > self.chunk_max_chars:
                    chunk["text"] = chunk["text"][:self.chunk_max_chars] + "..."

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

                # v9: Truncate gap chunks too
                for chunk in gap_chunks:
                    if len(chunk.get("text", "")) > self.chunk_max_chars:
                        chunk["text"] = chunk["text"][:self.chunk_max_chars] + "..."

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
                fallback = find_any_useful_answer(agent_results)
                if fallback:
                    logger.warning(
                        "Synthesis produced empty/refusal ('%s'), agent fallback: '%s'",
                        answer[:40],
                        fallback[:80],
                    )
                    result.final_answer = fallback

            # v10 Answer Type Critic: validate and fix answer
            if plan and result.final_answer and not _is_empty_or_refusal(result.final_answer):
                result.final_answer = await self._answer_type_critic(
                    question=question,
                    answer=result.final_answer,
                    plan=plan,
                    agent_results=agent_results,
                )

            result.total_tokens = sum(ar.total_tokens for ar in agent_results.values())
            result.aggregator_tokens = int(synth_cost * 1_000_000) if synth_cost > 0 else 0
            result.verifier_parse_ok = verification.parse_ok if verification else None

        except Exception as exc:
            logger.error("SAGE pipeline error for '%s': %s", question[:60], exc)
            result.error = str(exc)
            if result.agent_results:
                fallback = find_any_useful_answer(result.agent_results)
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
