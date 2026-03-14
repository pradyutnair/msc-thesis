"""Blackboard: shared mutable state for adaptive agency multi-agent coordination.

Extends M6's blackboard with per-sub-question retrieval mode tracking
and per-mode token accounting for efficiency analysis.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from multi_agent.types import (
    EntityEntry,
    EvidenceEntry,
    ExecutionLogEntry,
    M6SubQuestion,
    RetrievalMode,
    SubQuestionStatus,
)

logger = logging.getLogger(__name__)

_MAX_LOG_ENTRIES = 200
_EVIDENCE_DISPLAY_CHARS = 800


class Blackboard:
    """In-memory shared state with asyncio.Lock for adaptive agency coordination."""

    def __init__(self, question: str, token_budget: int = 200_000):
        self.question: str = question
        self.token_budget: int = token_budget
        self.tokens_used: int = 0
        self.current_tick: int = 0

        self.warm_start_context: str = ""
        self.expected_answer: str = ""
        self.question_type: str = "unknown"

        self.search_plan: list[M6SubQuestion] = []
        self.evidence: list[EvidenceEntry] = []
        self.entity_registry: list[EntityEntry] = []

        self.execution_log: list[ExecutionLogEntry] = []
        self.terminated: bool = False
        self.termination_reason: str = ""
        self.backtrack_count: int = 0
        self.redecomposition_epoch: int = 0

        self.final_answer: str | None = None
        self.allow_synthesis: bool = False

        self._last_action_time: float = 0.0
        self._lock = asyncio.Lock()
        self._evidence_counter: int = 0

        self.mode_tokens: dict[str, int] = {m.value: 0 for m in RetrievalMode}

    # ── Selective Reads ──────────────────────────────────────────────

    async def read_for_planner(self) -> dict[str, Any]:
        async with self._lock:
            result: dict[str, Any] = {"question": self.question}

            if self.search_plan:
                result["sub_questions"] = [self._sq_to_dict(sq) for sq in self.search_plan]

                failed_sqs = [sq for sq in self.search_plan if sq.status == SubQuestionStatus.FAILED]
                if failed_sqs:
                    lines = []
                    for sq in failed_sqs:
                        lines.append(f"- SQ-{sq.id}: \"{sq.text}\" -> FAILED after {sq.attempt_count} attempts")
                    result["failure_context"] = "\n".join(lines)

                verified_answers = {}
                for sq in self.search_plan:
                    if sq.status == SubQuestionStatus.VERIFIED and sq.answer:
                        verified_answers[sq.text] = sq.answer
                if verified_answers:
                    result["verified_answers"] = verified_answers

            return result

    async def read_for_worker(self, worker_id: str) -> dict[str, Any]:
        async with self._lock:
            available_sqs = []
            for sq in self.search_plan:
                if sq.status in (SubQuestionStatus.READY, SubQuestionStatus.NEEDS_RETRY):
                    d = self._sq_to_dict(sq)
                    d["num_dependents"] = sum(
                        1 for other in self.search_plan if sq.id in other.dependencies
                    )
                    available_sqs.append(d)

            claimed_sq = None
            for sq in self.search_plan:
                if sq.status == SubQuestionStatus.CLAIMED and sq.claimed_by == worker_id:
                    claimed_sq = self._sq_to_dict(sq)
                    break

            entities = {e.name: e.value for e in self.entity_registry}

            claimed_evidence = []
            claimed_sq_id = claimed_sq["id"] if claimed_sq is not None else -1
            if claimed_sq is not None:
                claimed_evidence = [
                    self._ev_to_dict(ev) for ev in self.evidence
                    if ev.sub_question_id == claimed_sq_id
                ]

            blackboard_context = self._build_cross_agent_context(worker_id)

            search_queries: list[str] = []
            if claimed_sq is not None:
                for sq in self.search_plan:
                    if sq.id == claimed_sq["id"]:
                        search_queries = list(getattr(sq, "search_queries", []))
                        break

            return {
                "question": self.question,
                "expected_answer": getattr(self, "expected_answer", ""),
                "available_sub_questions": available_sqs,
                "claimed_sub_question": claimed_sq,
                "entity_registry": entities,
                "claimed_evidence": claimed_evidence,
                "tokens_remaining": self.token_budget - self.tokens_used,
                "blackboard_context": blackboard_context,
                "search_queries": search_queries,
                "warm_start_context": getattr(self, "warm_start_context", ""),
                "redecomposition_epoch": self.redecomposition_epoch,
            }

    async def read_for_synthesizer(self) -> dict[str, Any]:
        async with self._lock:
            all_sqs = [self._sq_to_dict(sq) for sq in self.search_plan]
            verified_evidence = [
                self._ev_to_dict(ev) for ev in self.evidence if ev.verified
            ]
            entities = {e.name: e.value for e in self.entity_registry}

            return {
                "question": self.question,
                "expected_answer": getattr(self, "expected_answer", ""),
                "sub_questions": all_sqs,
                "verified_evidence": verified_evidence,
                "entity_registry": entities,
                "allow_synthesis": self.allow_synthesis,
            }

    async def read_for_coordinator(self) -> dict[str, Any]:
        async with self._lock:
            status_counts: dict[str, int] = {}
            for sq in self.search_plan:
                key = sq.status.value
                status_counts[key] = status_counts.get(key, 0) + 1

            return {
                "status_counts": status_counts,
                "total_sub_questions": len(self.search_plan),
                "tokens_used": self.tokens_used,
                "token_budget": self.token_budget,
                "budget_fraction": self.tokens_used / self.token_budget if self.token_budget > 0 else 1.0,
                "action_count": self.current_tick,
                "terminated": self.terminated,
                "termination_reason": self.termination_reason,
                "backtrack_count": self.backtrack_count,
                "last_action_time": self._last_action_time,
            }

    # ── Mutations ────────────────────────────────────────────────────

    async def set_search_plan(self, sub_questions: list[M6SubQuestion]) -> None:
        async with self._lock:
            self.search_plan = sub_questions
            for sq in self.search_plan:
                if sq.dependencies:
                    sq.status = SubQuestionStatus.BLOCKED
                else:
                    sq.status = SubQuestionStatus.READY
            self._log("planner", "set_search_plan",
                       f"{len(sub_questions)} sub-questions")

    async def reset_search_plan(
        self,
        new_sub_questions: list[M6SubQuestion],
        preserve_verified: bool = True,
    ) -> None:
        async with self._lock:
            self.allow_synthesis = False
            if preserve_verified:
                verified_sq_ids = {
                    sq.id for sq in self.search_plan
                    if sq.status == SubQuestionStatus.VERIFIED
                }
                self.evidence = [
                    ev for ev in self.evidence if ev.sub_question_id in verified_sq_ids
                ]

            self.redecomposition_epoch += 1
            self.search_plan = new_sub_questions
            for sq in self.search_plan:
                if sq.dependencies:
                    sq.status = SubQuestionStatus.BLOCKED
                else:
                    sq.status = SubQuestionStatus.READY
            self._check_unblocks()
            self._log("planner", "reset_search_plan",
                       f"{len(new_sub_questions)} sub-questions (preserve_verified={preserve_verified})")

    async def claim_sub_question(self, sq_id: int, worker_id: str) -> bool:
        async with self._lock:
            sq = self._get_sq(sq_id)
            if sq is None:
                return False
            if sq.status not in (SubQuestionStatus.READY, SubQuestionStatus.NEEDS_RETRY):
                return False
            sq.status = SubQuestionStatus.CLAIMED
            sq.claimed_by = worker_id
            sq.attempt_count += 1
            self._log(worker_id, "claim",
                       f"SQ-{sq_id} (attempt {sq.attempt_count}, mode={sq.mode.value})")
            return True

    async def post_evidence(
        self,
        entries: list[EvidenceEntry],
        sq_id: int,
        answer: str,
        worker_id: str,
    ) -> None:
        async with self._lock:
            for entry in entries:
                entry.id = f"ev_{sq_id}_{self._evidence_counter}"
                entry.sub_question_id = sq_id
                entry.retriever_id = worker_id
                self._evidence_counter += 1
                self.evidence.append(entry)

            sq = self._get_sq(sq_id)
            if sq is not None:
                sq.answer = answer
                sq.status = SubQuestionStatus.EVIDENCE_FOUND
                sq.claimed_by = worker_id
            self._log(worker_id, "post_evidence",
                       f"SQ-{sq_id}: {len(entries)} entries, answer='{answer[:60]}'")

    async def post_entity(self, entity: EntityEntry) -> None:
        async with self._lock:
            self.entity_registry = [
                e for e in self.entity_registry if e.name != entity.name
            ]
            self.entity_registry.append(entity)
            self._log("system", "post_entity",
                       f"{entity.name}={entity.value}")
            self._check_unblocks()

    async def verify_sub_question(
        self, sq_id: int, verified: bool, agent_id: str = "worker",
    ) -> None:
        async with self._lock:
            sq = self._get_sq(sq_id)
            if sq is None:
                return

            if verified:
                sq.status = SubQuestionStatus.VERIFIED
                for ev in self.evidence:
                    if ev.sub_question_id == sq_id:
                        ev.verified = True
                entity_value = sq.answer or "unknown"
                entity = EntityEntry(
                    name=f"answer_{sq_id}",
                    value=entity_value,
                    source_evidence_id=f"ev_{sq_id}_0",
                    verified=True,
                )
                self.entity_registry = [
                    e for e in self.entity_registry
                    if e.name != entity.name
                ]
                self.entity_registry.append(entity)
                self._check_unblocks()
                self._log(agent_id, "verify",
                           f"SQ-{sq_id} VERIFIED (answer='{sq.answer[:60] if sq.answer else ''}')")
            else:
                if sq.attempt_count >= sq.max_attempts:
                    sq.status = SubQuestionStatus.FAILED
                    self._log(agent_id, "verify",
                               f"SQ-{sq_id} FAILED (max attempts)")
                    self._propagate_failure(sq_id)
                else:
                    sq.status = SubQuestionStatus.NEEDS_RETRY
                    sq.claimed_by = None
                    self._log(agent_id, "verify",
                               f"SQ-{sq_id} NEEDS_RETRY (attempt {sq.attempt_count}/{sq.max_attempts})")

    async def record_action(self, tokens: int = 0) -> None:
        async with self._lock:
            if tokens > 0:
                self.tokens_used += tokens
            self.current_tick += 1
            self._last_action_time = time.monotonic()

    async def record_mode_tokens(self, mode: RetrievalMode, tokens: int) -> None:
        """Track token usage per retrieval mode for efficiency analysis."""
        async with self._lock:
            self.mode_tokens[mode.value] = self.mode_tokens.get(mode.value, 0) + tokens

    async def set_final_answer(self, answer: str) -> None:
        async with self._lock:
            self.final_answer = answer
            self._log("system", "set_final_answer", answer[:80])

    async def terminate(self, reason: str) -> None:
        async with self._lock:
            self.terminated = True
            self.termination_reason = reason
            self._log("system", "terminate", reason)

    async def salvage_answer(self) -> str:
        """Return the last verified sub-question answer, or empty."""
        async with self._lock:
            for sq in reversed(self.search_plan):
                if sq.status == SubQuestionStatus.VERIFIED and sq.answer:
                    return sq.answer
            return ""

    # ── Internal Helpers (must be called under lock) ─────────────────

    def _build_cross_agent_context(self, worker_id: str) -> str:
        lines: list[str] = []
        for sq in self.search_plan:
            if sq.status not in (
                SubQuestionStatus.EVIDENCE_FOUND,
                SubQuestionStatus.VERIFIED,
            ):
                continue
            answer_str = sq.answer or "unknown"
            lines.append(f"- Sub-question {sq.id}: \"{sq.text}\"")
            lines.append(f"  Answer: {answer_str}")

            sq_evidence = [
                ev for ev in self.evidence if ev.sub_question_id == sq.id
            ]
            for ev in sq_evidence[:5]:
                snippet = ev.content[:_EVIDENCE_DISPLAY_CHARS].replace("\n", " ")
                lines.append(f"  Evidence [{ev.source_chunk_id}]: {snippet}")

        if not lines:
            return "No findings from other agents yet."
        return "\n".join(lines)

    def _get_sq(self, sq_id: int) -> M6SubQuestion | None:
        for sq in self.search_plan:
            if sq.id == sq_id:
                return sq
        return None

    def _check_unblocks(self) -> None:
        for sq in self.search_plan:
            if sq.status != SubQuestionStatus.BLOCKED:
                continue
            all_deps_met = True
            for dep_id in sq.dependencies:
                dep_sq = self._get_sq(dep_id)
                if dep_sq is None or dep_sq.status != SubQuestionStatus.VERIFIED:
                    all_deps_met = False
                    break
            if all_deps_met:
                sq.status = SubQuestionStatus.READY
                logger.info("Unblocked SQ-%d (all dependencies verified)", sq.id)

    def _propagate_failure(self, failed_sq_id: int) -> None:
        for sq in self.search_plan:
            if sq.status != SubQuestionStatus.BLOCKED:
                continue
            if failed_sq_id not in sq.dependencies:
                continue
            sq.status = SubQuestionStatus.FAILED
            sq.answer = None
            logger.info("Propagated failure: SQ-%d FAILED (depends on failed SQ-%d)",
                        sq.id, failed_sq_id)
            self._log("system", "propagate_failure",
                       f"SQ-{sq.id} FAILED (dependency SQ-{failed_sq_id} failed)")
            self._propagate_failure(sq.id)

    def _log(self, agent_id: str, action: str, details: str = "") -> None:
        entry = ExecutionLogEntry(
            tick=self.current_tick,
            agent_id=agent_id,
            action=action,
            details=details,
        )
        self.execution_log.append(entry)
        if len(self.execution_log) > _MAX_LOG_ENTRIES:
            self.execution_log = self.execution_log[-_MAX_LOG_ENTRIES:]

    # ── Serialization Helpers ────────────────────────────────────────

    @staticmethod
    def _sq_to_dict(sq: M6SubQuestion) -> dict[str, Any]:
        return {
            "id": sq.id,
            "text": sq.text,
            "dependencies": sq.dependencies,
            "known_entities": sq.known_entities,
            "unknown_entities": sq.unknown_entities,
            "search_hints": sq.search_hints,
            "search_queries": getattr(sq, "search_queries", []),
            "mode": sq.mode.value,
            "status": sq.status.value,
            "claimed_by": sq.claimed_by,
            "answer": sq.answer,
            "attempt_count": sq.attempt_count,
            "max_attempts": sq.max_attempts,
        }

    @staticmethod
    def _ev_to_dict(ev: EvidenceEntry) -> dict[str, Any]:
        return {
            "id": ev.id,
            "sub_question_id": ev.sub_question_id,
            "content": ev.content[:_EVIDENCE_DISPLAY_CHARS],
            "source_chunk_id": ev.source_chunk_id,
            "relevance_score": ev.relevance_score,
            "verified": ev.verified,
            "retriever_id": ev.retriever_id,
        }

    async def get_snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "question": self.question,
                "expected_answer": getattr(self, "expected_answer", ""),
                "question_type": getattr(self, "question_type", "unknown"),
                "current_tick": self.current_tick,
                "tokens_used": self.tokens_used,
                "token_budget": self.token_budget,
                "terminated": self.terminated,
                "termination_reason": self.termination_reason,
                "final_answer": self.final_answer,
                "backtrack_count": self.backtrack_count,
                "sub_questions": [self._sq_to_dict(sq) for sq in self.search_plan],
                "evidence": [self._ev_to_dict(ev) for ev in self.evidence],
                "evidence_count": len(self.evidence),
                "verified_evidence_count": sum(1 for ev in self.evidence if ev.verified),
                "entity_registry": {e.name: e.value for e in self.entity_registry},
                "execution_log": [
                    {"tick": e.tick, "agent": e.agent_id, "action": e.action, "details": e.details}
                    for e in self.execution_log[-50:]
                ],
                "mode_tokens": dict(self.mode_tokens),
            }
