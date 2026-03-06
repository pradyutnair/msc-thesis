from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.m6_orchestrator import M6Orchestrator
from multi_agent.m6_types import AgentMessage, BlackboardState, ClaimRecord, FrontierItem, ManagerDecision, WorkerTask
from multi_agent.search_agent import SearchAgent
from multi_agent.types import DecompositionPlan, PipelineResult, SubQuestion

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_BIRTH_KEYWORDS = {"birthplace", "born", "birth", "native", "hometown"}
_DEATH_KEYWORDS = {"death", "died", "buried", "grave"}
_LOCATION_KEYWORDS = {"located", "headquarters", "based", "situated", "capital", "city"}
_FOUNDING_KEYWORDS = {"founded", "established", "created", "formed", "origin"}


class M6LitCorePipeline:
    """Literature-grounded autonomous multi-agent pipeline."""

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
        self.max_manager_steps = int(ma_cfg.get("m6_max_manager_steps", 12))
        self.max_worker_tasks = int(ma_cfg.get("m6_max_worker_tasks", 6))
        self.max_claims_to_merge = int(ma_cfg.get("m6_max_claims_to_merge", 3))
        self.max_token_budget = int(ma_cfg.get("max_token_budget", 32768))
        self.worker_loops = int(ma_cfg.get("m6_worker_max_loops", 4))
        self.verbose = bool(ma_cfg.get("verbose", False))
        self.extract_evidence = bool(ma_cfg.get("sage_extract_evidence", True))
        self.extract_max_bullets = int(ma_cfg.get("sage_extract_max_bullets", 3))
        self.retriever_prompt = ma_cfg.get("m6_search_prompt") or ma_cfg.get("sage_retriever_prompt")

        self.orchestrator = M6Orchestrator(
            llm_client=self.llm,
            manager_prompt=ma_cfg.get("m6_manager_prompt"),
            extract_prompt=ma_cfg.get("m6_extract_prompt"),
            merge_prompt=ma_cfg.get("m6_merge_prompt"),
            answer_prompt=ma_cfg.get("m6_answer_prompt"),
        )

    def _make_agent(self, cache: EvidenceCache) -> SearchAgent:
        return SearchAgent(
            llm_client=self.llm,
            tools=self.base_tools,
            evidence_cache=cache,
            max_loops=self.worker_loops,
            max_token_budget=self.max_token_budget,
            prompt_path=self.retriever_prompt,
            verbose=self.verbose,
            dual_retrieval_on_low_evidence=True,
            low_evidence_threshold=2,
            dual_retrieval_top_k=5,
            extract_evidence=self.extract_evidence,
            extract_max_bullets=self.extract_max_bullets,
        )

    @staticmethod
    def _question_type(question: str) -> str:
        lowered = question.lower()
        if any(token in lowered for token in ["both", "either", "older", "younger", "same", "compare"]):
            return "comparison"
        return "bridge"

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join((value or "").strip().split())

    def _expand_queries(self, role: str, goal: str, query_hints: list[str]) -> list[str]:
        goal_l = goal.lower()
        out = []
        seen = set()
        for query in query_hints + [goal]:
            q = self._normalize_text(query)
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        if role == "bridge":
            extras = [f"{goal} wikipedia", f"{goal} biography"]
        elif role == "disambiguation":
            extras = [f"{goal} alias", f"{goal} wikipedia disambiguation"]
        elif role == "refuter":
            extras = [f"contradiction {goal}", f"alternative answer {goal}"]
        else:
            extras = []
            if any(word in goal_l for word in _BIRTH_KEYWORDS):
                extras.extend([f"{goal} born", f"{goal} early life"])
            if any(word in goal_l for word in _DEATH_KEYWORDS):
                extras.extend([f"{goal} died", f"{goal} obituary"])
            if any(word in goal_l for word in _LOCATION_KEYWORDS):
                extras.extend([f"{goal} location", f"{goal} city"])
            if any(word in goal_l for word in _FOUNDING_KEYWORDS):
                extras.extend([f"{goal} history", f"{goal} founded"])
        for query in extras:
            q = self._normalize_text(query)
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out[:6]

    def _build_board_state(self, question: str, board: BlackboardState) -> str:
        frontier_lines = []
        for item in board.frontier:
            frontier_lines.append(
                {
                    "id": item.id,
                    "role_hint": item.role_hint,
                    "goal": item.goal,
                    "query_hints": item.query_hints[:4],
                    "depends_on_claim_ids": item.depends_on_claim_ids,
                    "status": item.status,
                    "priority": item.priority,
                    "notes": item.notes,
                }
            )
        claim_lines = []
        for claim in board.claims:
            claim_lines.append(
                {
                    "id": claim.id,
                    "entity": claim.entity,
                    "relation": claim.relation,
                    "value": claim.value,
                    "status": claim.status,
                    "confidence": round(claim.confidence, 3),
                    "supporting_chunk_ids": claim.supporting_chunk_ids[:5],
                    "source_task_ids": claim.source_task_ids,
                    "notes": claim.notes,
                }
            )
        task_lines = []
        for task in board.tasks[-8:]:
            task_lines.append(
                {
                    "id": task.id,
                    "role": task.role,
                    "goal": task.goal,
                    "sub_question": task.sub_question,
                    "query_hints": task.query_hints[:4],
                    "depends_on_claim_ids": task.depends_on_claim_ids,
                    "status": task.status,
                    "answer": task.answer[:160],
                    "confidence": round(task.confidence, 3),
                    "evidence_doc_ids": task.evidence_doc_ids[:6],
                    "notes": task.notes,
                }
            )
        return (
            f"QUESTION: {question}\n"
            f"DRAFT_ANSWER: {board.draft_answer}\n"
            f"SUPPORTED_CLAIM_IDS: {sorted(board.supported_claim_ids())}\n"
            f"FRONTIER: {frontier_lines}\n"
            f"CLAIMS: {claim_lines}\n"
            f"RECENT_TASKS: {task_lines}\n"
        )

    def _task_to_subquestion(self, task: WorkerTask) -> SubQuestion:
        return SubQuestion(
            index=task.id,
            text=task.sub_question or task.goal,
            search_hints=task.query_hints,
            depends_on=[],
        )

    def _dependencies_satisfied(self, task: WorkerTask, board: BlackboardState) -> bool:
        supported = board.supported_claim_ids()
        return all(dep in supported for dep in task.depends_on_claim_ids)

    def _next_claim_id(self, board: BlackboardState) -> int:
        return max((claim.id for claim in board.claims), default=-1) + 1

    def _next_frontier_id(self, board: BlackboardState) -> int:
        return max((item.id for item in board.frontier), default=-1) + 1

    def _next_task_id(self, board: BlackboardState) -> int:
        return max((task.id for task in board.tasks), default=-1) + 1

    def _similar_failure_count(self, board: BlackboardState, role: str, goal: str) -> int:
        goal_norm = self._normalize_text(goal).lower()
        return sum(
            1
            for task in board.tasks
            if task.role == role
            and task.status == "failed"
            and self._normalize_text(task.goal).lower() == goal_norm
        )

    def _apply_worker_updates(self, board: BlackboardState, task: WorkerTask, updates: Any) -> list[int]:
        new_claim_ids = []
        next_claim_id = self._next_claim_id(board)
        next_frontier_id = self._next_frontier_id(board)
        for claim in updates.proposed_claims:
            claim.id = next_claim_id
            next_claim_id += 1
            claim.source_task_ids = sorted(set(claim.source_task_ids + [task.id]))
            board.claims.append(claim)
            new_claim_ids.append(claim.id)
        for frontier in updates.frontier_updates:
            frontier.id = next_frontier_id
            next_frontier_id += 1
            board.frontier.append(frontier)
        if updates.message:
            board.messages.append(
                AgentMessage(
                    task_id=task.id,
                    role=task.role,
                    message_type="proposal",
                    content=updates.message,
                    payload={"new_claim_ids": new_claim_ids},
                )
            )
        return new_claim_ids

    def _apply_claim_merge(self, board: BlackboardState, updates: list[dict[str, Any]]) -> None:
        by_id = {claim.id: claim for claim in board.claims}
        for update in updates:
            claim = by_id.get(int(update.get("claim_id", -1)))
            if claim is None:
                continue
            status = str(update.get("status", claim.status)).strip()
            if status not in {"proposed", "supported", "contested", "rejected"}:
                status = claim.status
            claim.status = status
            claim.confidence = max(claim.confidence, float(update.get("confidence", claim.confidence) or 0.0))
            notes = str(update.get("notes", "")).strip()
            if notes:
                claim.notes = notes

    def _fallback_decision(self, board: BlackboardState, question: str) -> ManagerDecision:
        proposed = [claim for claim in board.claims if claim.status == "proposed"]
        if proposed:
            return ManagerDecision(
                action="merge_claim",
                rationale="fallback_merge_pending_claims",
                claim_ids=[claim.id for claim in proposed[: self.max_claims_to_merge]],
            )
        if not board.draft_answer and any(claim.status == "supported" for claim in board.claims):
            return ManagerDecision(action="compose_answer", rationale="fallback_compose_answer")
        if board.draft_answer and not any(task.role == "refuter" for task in board.tasks):
            return ManagerDecision(action="request_refutation", rationale="fallback_request_refutation")
        open_frontier = [item for item in board.frontier if item.status == "open"]
        if open_frontier and len(board.tasks) < self.max_worker_tasks:
            item = sorted(open_frontier, key=lambda x: (-x.priority, x.id))[0]
            role_action = {
                "bridge": "spawn_bridge_worker",
                "attribute": "spawn_attribute_worker",
                "disambiguation": "spawn_disambiguation_worker",
                "refuter": "request_refutation",
            }[item.role_hint]
            return ManagerDecision(
                action=role_action,
                rationale="fallback_open_frontier",
                task_goal=item.goal,
                sub_question=item.goal,
                query_hints=item.query_hints,
                depends_on_claim_ids=item.depends_on_claim_ids,
                frontier_id=item.id,
            )
        return ManagerDecision(action="terminate", rationale="fallback_terminate")

    def _promote_frontier(self, board: BlackboardState, frontier_id: int | None, status: str = "done") -> None:
        if frontier_id is None:
            return
        for item in board.frontier:
            if item.id == frontier_id:
                item.status = status
                return

    def _to_claim_graph(self, board: BlackboardState) -> list[dict[str, Any]]:
        return [asdict(claim) for claim in board.claims]

    def _to_trace(self, trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return trace

    def _fallback_answer_from_claims(self, board: BlackboardState) -> tuple[str, list[int]]:
        supported = [claim for claim in board.claims if claim.status == "supported" and claim.value]
        proposed = [claim for claim in board.claims if claim.status == "proposed" and claim.value]
        pool = supported or proposed
        if not pool:
            return "", []
        pool = sorted(
            pool,
            key=lambda claim: (claim.confidence, len(claim.supporting_chunk_ids), len(claim.value)),
            reverse=True,
        )
        best = pool[0]
        return best.value.strip(), [best.id]

    def _to_decomposition(self, board: BlackboardState, question_type: str) -> DecompositionPlan:
        sub_questions = [
            SubQuestion(
                index=task.id,
                text=task.sub_question,
                search_hints=list(task.query_hints),
                depends_on=list(task.depends_on_claim_ids),
            )
            for task in board.tasks
        ]
        edges = [(dep, task.id) for task in board.tasks for dep in task.depends_on_claim_ids]
        return DecompositionPlan(question_type=question_type, sub_questions=sub_questions, dependency_edges=edges)

    async def _run_worker(self, cache: EvidenceCache, task: WorkerTask, question: str) -> Any:
        agent = self._make_agent(cache)
        result = await agent.run(sub_question=self._task_to_subquestion(task), original_question=question)
        task.answer = result.answer
        task.confidence = float(result.confidence or 0.0)
        task.evidence_doc_ids = list(result.evidence_doc_ids)
        task.status = "done" if result.answer or result.extracted_evidence else "failed"
        return result

    async def run(self, question: str) -> PipelineResult:
        started = time.monotonic()
        result = PipelineResult(question=question)
        board = BlackboardState(
            frontier=[FrontierItem(id=0, role_hint="bridge", goal=question, query_hints=[question], priority=3)]
        )
        cache = EvidenceCache(enabled=True)
        trace: list[dict[str, Any]] = []
        question_type = self._question_type(question)

        try:
            for step in range(1, self.max_manager_steps + 1):
                board_state = self._build_board_state(question, board)
                decision = await self.orchestrator.decide_action(question, board_state)
                if decision.action == "terminate" and decision.rationale.startswith("fallback"):
                    decision = self._fallback_decision(board, question)
                trace.append({"step": step, "kind": "manager_decision", "decision": asdict(decision)})
                result.manager_actions += 1

                if decision.action.startswith("spawn_") or decision.action == "request_refutation":
                    if len(board.tasks) >= self.max_worker_tasks:
                        trace.append({"step": step, "kind": "manager_skip", "reason": "max_worker_tasks"})
                        continue
                    role = {
                        "spawn_bridge_worker": "bridge",
                        "spawn_attribute_worker": "attribute",
                        "spawn_disambiguation_worker": "disambiguation",
                        "request_refutation": "refuter",
                    }[decision.action]
                    if not decision.task_goal and decision.frontier_id is not None:
                        for item in board.frontier:
                            if item.id == decision.frontier_id:
                                decision.task_goal = item.goal
                                decision.sub_question = decision.sub_question or item.goal
                                decision.query_hints = decision.query_hints or item.query_hints
                                decision.depends_on_claim_ids = decision.depends_on_claim_ids or item.depends_on_claim_ids
                                break
                    if not decision.task_goal:
                        fallback = self._fallback_decision(board, question)
                        decision.task_goal = fallback.task_goal or question
                        decision.sub_question = fallback.sub_question or decision.task_goal
                        decision.query_hints = fallback.query_hints or [decision.task_goal]
                    task = WorkerTask(
                        id=self._next_task_id(board),
                        role=role,
                        goal=self._normalize_text(decision.task_goal or question),
                        sub_question=self._normalize_text(decision.sub_question or decision.task_goal or question),
                        query_hints=self._expand_queries(role, decision.task_goal or question, list(decision.query_hints or [decision.task_goal or question])),
                        depends_on_claim_ids=list(decision.depends_on_claim_ids),
                        source_frontier_id=decision.frontier_id,
                        notes=decision.rationale,
                    )
                    if self._similar_failure_count(board, task.role, task.goal) >= 2:
                        task.status = "failed"
                        task.notes = f"repeated_failure_guard:{task.goal}"
                        board.tasks.append(task)
                        if task.source_frontier_id is not None:
                            self._promote_frontier(board, task.source_frontier_id, status="blocked")
                        trace.append({"step": step, "kind": "manager_skip", "reason": "repeated_failure_guard", "task": asdict(task)})
                        continue
                    if not self._dependencies_satisfied(task, board):
                        task.status = "failed"
                        task.notes = f"blocked_dependencies:{task.depends_on_claim_ids}"
                        board.tasks.append(task)
                        if task.source_frontier_id is not None:
                            self._promote_frontier(board, task.source_frontier_id, status="blocked")
                        continue
                    board.tasks.append(task)
                    worker_result = await self._run_worker(cache, task, question)
                    trace.append(
                        {
                            "step": step,
                            "kind": "worker_result",
                            "task": asdict(task),
                            "answer": worker_result.answer,
                            "confidence": worker_result.confidence,
                            "evidence_doc_ids": worker_result.evidence_doc_ids,
                            "extracted_evidence": worker_result.extracted_evidence,
                        }
                    )
                    board.messages.append(
                        AgentMessage(
                            task_id=task.id,
                            role=task.role,
                            message_type="evidence_update",
                            content=worker_result.answer,
                            payload={
                                "confidence": worker_result.confidence,
                                "evidence_doc_ids": worker_result.evidence_doc_ids,
                                "extracted_evidence": worker_result.extracted_evidence,
                            },
                        )
                    )
                    updates = await self.orchestrator.extract_worker_updates(
                        question=question,
                        task=task,
                        board_state=self._build_board_state(question, board),
                        agent_answer=worker_result.answer,
                        extracted_evidence=worker_result.extracted_evidence,
                        chunk_snippets=worker_result.retrieved_chunks,
                    )
                    new_claim_ids = self._apply_worker_updates(board, task, updates)
                    if task.source_frontier_id is not None:
                        frontier_status = "done" if (task.status == "done" or new_claim_ids or worker_result.extracted_evidence) else "blocked"
                        self._promote_frontier(board, task.source_frontier_id, status=frontier_status)
                    result.agent_results[task.id] = worker_result
                    if new_claim_ids:
                        merge_result = await self.orchestrator.merge_claims(
                            question,
                            self._build_board_state(question, board),
                            [claim for claim in board.claims if claim.id in new_claim_ids],
                        )
                        self._apply_claim_merge(board, merge_result.updates)
                        trace.append({"step": step, "kind": "claim_merge", "updates": merge_result.updates})
                elif decision.action == "merge_claim":
                    targets = [claim for claim in board.claims if claim.id in set(decision.claim_ids) and claim.status == "proposed"]
                    if not targets:
                        targets = [claim for claim in board.claims if claim.status == "proposed"][: self.max_claims_to_merge]
                    if targets:
                        merge_result = await self.orchestrator.merge_claims(question, board_state, targets)
                        self._apply_claim_merge(board, merge_result.updates)
                        trace.append({"step": step, "kind": "claim_merge", "updates": merge_result.updates})
                elif decision.action == "compose_answer":
                    answer, supporting_claim_ids, notes = await self.orchestrator.compose_answer(question, board_state)
                    if not answer.strip():
                        answer, supporting_claim_ids = self._fallback_answer_from_claims(board)
                        notes = f"{notes}|claim_fallback" if notes else "claim_fallback"
                    board.draft_answer = answer
                    board.draft_supporting_claim_ids = supporting_claim_ids
                    trace.append({"step": step, "kind": "compose_answer", "answer": answer, "supporting_claim_ids": supporting_claim_ids, "notes": notes})
                elif decision.action == "terminate":
                    if board.draft_answer:
                        break
                    fallback = self._fallback_decision(board, question)
                    if fallback.action == "terminate":
                        break
                    trace.append({"step": step, "kind": "terminate_redirect", "redirect": asdict(fallback)})
                    decision = fallback
                    continue

            if not board.draft_answer:
                answer, supporting_claim_ids, _ = await self.orchestrator.compose_answer(question, self._build_board_state(question, board))
                if not answer.strip():
                    answer, supporting_claim_ids = self._fallback_answer_from_claims(board)
                board.draft_answer = answer
                board.draft_supporting_claim_ids = supporting_claim_ids

            result.final_answer = self._normalize_text(board.draft_answer)
            result.claim_graph = self._to_claim_graph(board)
            result.autonomy_trace = self._to_trace(trace)
            result.worker_messages = [asdict(message) for message in board.messages]
            result.decomposition = self._to_decomposition(board, question_type)
            result.question_type = question_type
            result.num_sub_questions = len(board.tasks)
            result.num_waves = len(trace)
            result.total_tokens = sum(ar.total_tokens for ar in result.agent_results.values())
            result.cache_analytics = cache.compute_analytics_sync()
            result.retry_trigger_reasons = [
                f"supported_claims={len([c for c in board.claims if c.status == 'supported'])}",
                f"contested_claims={len([c for c in board.claims if c.status == 'contested'])}",
                f"frontier_open={len([f for f in board.frontier if f.status == 'open'])}",
            ]
            result.verifier_parse_ok = True
        except Exception as exc:
            logger.error("M6 pipeline error for '%s': %s", question[:80], exc)
            result.error = str(exc)
            result.claim_graph = self._to_claim_graph(board)
            result.autonomy_trace = self._to_trace(trace)
            result.worker_messages = [asdict(message) for message in board.messages]

        result.wall_clock_seconds = time.monotonic() - started
        return result
