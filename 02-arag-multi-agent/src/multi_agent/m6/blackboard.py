"""Blackboard: shared mutable state for M6 multi-agent coordination.

Created fresh per question. All mutations are protected by asyncio.Lock.
Provides selective read methods per agent type to reduce prompt bloat.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from multi_agent.m6.types import (
    Contradiction,
    EntityEntry,
    EvidenceEntry,
    ExecutionLogEntry,
    KnowledgeGap,
    M6SubQuestion,
    SubQuestionStatus,
)

logger = logging.getLogger(__name__)

_MAX_LOG_ENTRIES = 200


class Blackboard:
    """In-memory shared state with asyncio.Lock for M6 coordination.

    Thread-safe for concurrent agent access within a single asyncio event loop.
    """

    def __init__(self, question: str, token_budget: int = 200_000):
        self.question: str = question
        self.token_budget: int = token_budget
        self.tokens_used: int = 0
        self.current_tick: int = 0

        # Core state
        self.search_plan: list[M6SubQuestion] = []
        self.evidence: list[EvidenceEntry] = []
        self.entity_registry: list[EntityEntry] = []
        self.knowledge_gaps: list[KnowledgeGap] = []
        self.contradictions: list[Contradiction] = []

        # Execution tracking
        self.execution_log: list[ExecutionLogEntry] = []
        self.terminated: bool = False
        self.termination_reason: str = ""
        self.backtrack_count: int = 0

        # Internal
        self._lock = asyncio.Lock()
        self._evidence_counter: int = 0

    # ── Selective Reads ──────────────────────────────────────────────

    async def read_for_decomposer(self) -> dict[str, Any]:
        """Decomposer only needs the original question."""
        async with self._lock:
            return {"question": self.question}

    async def read_for_retriever(self, retriever_id: str) -> dict[str, Any]:
        """Retriever needs: ready/needs_retry sub-Qs, entity registry,
        relevant evidence for its claimed sub-Q, budget info."""
        async with self._lock:
            available_sqs = [
                self._sq_to_dict(sq) for sq in self.search_plan
                if sq.status in (SubQuestionStatus.READY, SubQuestionStatus.NEEDS_RETRY)
            ]
            claimed_sq = None
            for sq in self.search_plan:
                if sq.status == SubQuestionStatus.CLAIMED and sq.claimed_by == retriever_id:
                    claimed_sq = self._sq_to_dict(sq)
                    break

            entities = {e.name: e.value for e in self.entity_registry}

            # Evidence for the claimed sub-question
            claimed_evidence = []
            if claimed_sq is not None:
                sq_id = claimed_sq["id"]
                claimed_evidence = [
                    self._ev_to_dict(ev) for ev in self.evidence
                    if ev.sub_question_id == sq_id
                ]

            return {
                "question": self.question,
                "available_sub_questions": available_sqs,
                "claimed_sub_question": claimed_sq,
                "entity_registry": entities,
                "claimed_evidence": claimed_evidence,
                "tokens_remaining": self.token_budget - self.tokens_used,
            }

    async def read_for_critic(self) -> dict[str, Any]:
        """Critic needs: sub-Qs with EVIDENCE_FOUND, their evidence,
        entities, existing contradictions."""
        async with self._lock:
            pending_sqs = [
                self._sq_to_dict(sq) for sq in self.search_plan
                if sq.status == SubQuestionStatus.EVIDENCE_FOUND
            ]

            # Gather evidence for each pending sub-question
            sq_evidence: dict[int, list[dict]] = {}
            for sq_dict in pending_sqs:
                sq_id = sq_dict["id"]
                sq_evidence[sq_id] = [
                    self._ev_to_dict(ev) for ev in self.evidence
                    if ev.sub_question_id == sq_id
                ]

            entities = {e.name: e.value for e in self.entity_registry}

            return {
                "pending_sub_questions": pending_sqs,
                "sub_question_evidence": sq_evidence,
                "entity_registry": entities,
                "contradictions": [
                    {"sub_question_ids": c.sub_question_ids, "description": c.description}
                    for c in self.contradictions
                ],
            }

    async def read_for_synthesizer(self) -> dict[str, Any]:
        """Synthesizer needs: question, verified evidence, entities,
        all sub-Qs with statuses."""
        async with self._lock:
            all_sqs = [self._sq_to_dict(sq) for sq in self.search_plan]
            verified_evidence = [
                self._ev_to_dict(ev) for ev in self.evidence if ev.verified
            ]
            entities = {e.name: e.value for e in self.entity_registry}

            return {
                "question": self.question,
                "sub_questions": all_sqs,
                "verified_evidence": verified_evidence,
                "entity_registry": entities,
            }

    async def read_for_coordinator(self) -> dict[str, Any]:
        """Coordinator needs: status counts, budget, termination flags."""
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
                "current_tick": self.current_tick,
                "terminated": self.terminated,
                "termination_reason": self.termination_reason,
                "backtrack_count": self.backtrack_count,
            }

    # ── Mutations ────────────────────────────────────────────────────

    async def set_search_plan(self, sub_questions: list[M6SubQuestion]) -> None:
        """Decomposer writes the DAG. Sets initial READY/BLOCKED statuses."""
        async with self._lock:
            self.search_plan = sub_questions
            # Set initial statuses based on dependencies
            for sq in self.search_plan:
                if sq.dependencies:
                    sq.status = SubQuestionStatus.BLOCKED
                else:
                    sq.status = SubQuestionStatus.READY
            self._log("decomposer", "set_search_plan",
                       f"{len(sub_questions)} sub-questions")

    async def claim_sub_question(self, sq_id: int, retriever_id: str) -> bool:
        """Atomic check-and-claim. Returns True if successfully claimed."""
        async with self._lock:
            sq = self._get_sq(sq_id)
            if sq is None:
                return False
            if sq.status not in (SubQuestionStatus.READY, SubQuestionStatus.NEEDS_RETRY):
                return False
            sq.status = SubQuestionStatus.CLAIMED
            sq.claimed_by = retriever_id
            sq.attempt_count += 1
            self._log(retriever_id, "claim",
                       f"SQ-{sq_id} (attempt {sq.attempt_count})")
            return True

    async def post_evidence(
        self,
        entries: list[EvidenceEntry],
        sq_id: int,
        answer: str,
        retriever_id: str,
    ) -> None:
        """Retriever posts evidence + answer. Status → EVIDENCE_FOUND."""
        async with self._lock:
            for entry in entries:
                entry.id = f"ev_{sq_id}_{self._evidence_counter}"
                entry.sub_question_id = sq_id
                entry.retriever_id = retriever_id
                self._evidence_counter += 1
                self.evidence.append(entry)

            sq = self._get_sq(sq_id)
            if sq is not None:
                sq.answer = answer
                sq.status = SubQuestionStatus.EVIDENCE_FOUND
                sq.claimed_by = retriever_id
            self._log(retriever_id, "post_evidence",
                       f"SQ-{sq_id}: {len(entries)} entries, answer='{answer[:60]}'")

    async def post_entity(self, entity: EntityEntry) -> None:
        """Post a resolved entity. Triggers unblock check."""
        async with self._lock:
            # Replace existing entity with same name if present
            self.entity_registry = [
                e for e in self.entity_registry if e.name != entity.name
            ]
            self.entity_registry.append(entity)
            self._log("system", "post_entity",
                       f"{entity.name}={entity.value}")
            self._check_unblocks()

    async def verify_sub_question(self, sq_id: int, verified: bool) -> None:
        """Critic verdict: VERIFIED or NEEDS_RETRY/FAILED.

        On verify: marks evidence verified, posts entity, triggers unblocks.
        On reject: increments attempt or marks FAILED.
        """
        async with self._lock:
            sq = self._get_sq(sq_id)
            if sq is None:
                return

            if verified:
                sq.status = SubQuestionStatus.VERIFIED
                # Mark evidence as verified
                for ev in self.evidence:
                    if ev.sub_question_id == sq_id:
                        ev.verified = True
                # Always post entity so dependents can resolve [answer_N]
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
                self._log("critic", "verify",
                           f"SQ-{sq_id} VERIFIED (answer='{sq.answer[:60] if sq.answer else ''}')")
            else:
                if sq.attempt_count >= sq.max_attempts:
                    sq.status = SubQuestionStatus.FAILED
                    self._log("critic", "verify",
                               f"SQ-{sq_id} FAILED (max attempts)")
                    # Propagate failure to dependents that can never be unblocked
                    self._propagate_failure(sq_id)
                else:
                    sq.status = SubQuestionStatus.NEEDS_RETRY
                    sq.claimed_by = None
                    self._log("critic", "verify",
                               f"SQ-{sq_id} NEEDS_RETRY (attempt {sq.attempt_count}/{sq.max_attempts})")

    async def backtrack_sub_question(self, sq_id: int, reason: str) -> None:
        """Backtrack: remove evidence, remove entity, reset status,
        recursively re-block dependents."""
        async with self._lock:
            sq = self._get_sq(sq_id)
            if sq is None:
                return

            # Remove evidence for this sub-question
            self.evidence = [
                ev for ev in self.evidence if ev.sub_question_id != sq_id
            ]

            # Remove entity
            entity_name = f"answer_{sq_id}"
            self.entity_registry = [
                e for e in self.entity_registry if e.name != entity_name
            ]

            # Reset status
            if sq.attempt_count >= sq.max_attempts:
                sq.status = SubQuestionStatus.FAILED
            else:
                sq.status = SubQuestionStatus.NEEDS_RETRY
                sq.claimed_by = None
            sq.answer = None

            self.backtrack_count += 1

            # Recursively re-block dependents
            self._reblock_dependents(sq_id)

            self._log("critic", "backtrack",
                       f"SQ-{sq_id}: {reason}")

    async def post_knowledge_gap(self, gap: KnowledgeGap) -> None:
        """Critic posts a knowledge gap for a sub-question."""
        async with self._lock:
            self.knowledge_gaps.append(gap)
            self._log("critic", "knowledge_gap",
                       f"SQ-{gap.sub_question_id}: {gap.description[:80]}")

    async def post_contradiction(self, contradiction: Contradiction) -> None:
        """Critic posts a detected contradiction."""
        async with self._lock:
            self.contradictions.append(contradiction)
            self._log("critic", "contradiction",
                       f"SQs {contradiction.sub_question_ids}: {contradiction.description[:80]}")

    async def add_tokens(self, tokens: int) -> None:
        """Track token usage."""
        async with self._lock:
            self.tokens_used += tokens

    async def increment_tick(self) -> None:
        """Advance tick counter."""
        async with self._lock:
            self.current_tick += 1

    async def terminate(self, reason: str) -> None:
        """Mark the blackboard as terminated."""
        async with self._lock:
            self.terminated = True
            self.termination_reason = reason
            self._log("system", "terminate", reason)

    # ── Internal Helpers (must be called under lock) ─────────────────

    def _get_sq(self, sq_id: int) -> M6SubQuestion | None:
        for sq in self.search_plan:
            if sq.id == sq_id:
                return sq
        return None

    def _check_unblocks(self) -> None:
        """Transition BLOCKED → READY when all dependencies are VERIFIED."""
        resolved_entity_names = {e.name for e in self.entity_registry}
        for sq in self.search_plan:
            if sq.status != SubQuestionStatus.BLOCKED:
                continue
            # Check: all dependencies must be VERIFIED
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
        """When a SQ fails, mark BLOCKED dependents as FAILED if they
        can never be unblocked (because a required dependency is FAILED)."""
        for sq in self.search_plan:
            if sq.status != SubQuestionStatus.BLOCKED:
                continue
            if failed_sq_id not in sq.dependencies:
                continue
            # This SQ depends on a FAILED SQ — it can never be unblocked
            sq.status = SubQuestionStatus.FAILED
            sq.answer = None
            logger.info("Propagated failure: SQ-%d FAILED (depends on failed SQ-%d)",
                        sq.id, failed_sq_id)
            self._log("system", "propagate_failure",
                       f"SQ-{sq.id} FAILED (dependency SQ-{failed_sq_id} failed)")
            # Recurse: this SQ's dependents also fail
            self._propagate_failure(sq.id)

    def _reblock_dependents(self, sq_id: int) -> None:
        """Recursively re-block sub-questions that depend on sq_id."""
        for sq in self.search_plan:
            if sq_id in sq.dependencies and sq.status in (
                SubQuestionStatus.READY,
                SubQuestionStatus.CLAIMED,
                SubQuestionStatus.EVIDENCE_FOUND,
                SubQuestionStatus.VERIFIED,
                SubQuestionStatus.NEEDS_RETRY,
            ):
                # Remove evidence and entity for this dependent too
                self.evidence = [
                    ev for ev in self.evidence if ev.sub_question_id != sq.id
                ]
                entity_name = f"answer_{sq.id}"
                self.entity_registry = [
                    e for e in self.entity_registry if e.name != entity_name
                ]
                sq.status = SubQuestionStatus.BLOCKED
                sq.answer = None
                sq.claimed_by = None
                logger.info("Re-blocked SQ-%d (depends on backtracked SQ-%d)", sq.id, sq_id)
                # Recurse
                self._reblock_dependents(sq.id)

    def _log(self, agent_id: str, action: str, details: str = "") -> None:
        """Append to execution log (bounded)."""
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
            "content": ev.content[:500],
            "source_chunk_id": ev.source_chunk_id,
            "relevance_score": ev.relevance_score,
            "verified": ev.verified,
            "retriever_id": ev.retriever_id,
        }

    async def get_snapshot(self) -> dict[str, Any]:
        """Full snapshot for diagnostics / pipeline result."""
        async with self._lock:
            return {
                "question": self.question,
                "current_tick": self.current_tick,
                "tokens_used": self.tokens_used,
                "token_budget": self.token_budget,
                "terminated": self.terminated,
                "termination_reason": self.termination_reason,
                "backtrack_count": self.backtrack_count,
                "sub_questions": [self._sq_to_dict(sq) for sq in self.search_plan],
                "evidence_count": len(self.evidence),
                "verified_evidence_count": sum(1 for ev in self.evidence if ev.verified),
                "entity_registry": {e.name: e.value for e in self.entity_registry},
                "knowledge_gaps": [
                    {"sq_id": g.sub_question_id, "desc": g.description}
                    for g in self.knowledge_gaps
                ],
                "contradictions": [
                    {"sq_ids": c.sub_question_ids, "desc": c.description}
                    for c in self.contradictions
                ],
                "execution_log": [
                    {"tick": e.tick, "agent": e.agent_id, "action": e.action, "details": e.details}
                    for e in self.execution_log[-50:]
                ],
            }
