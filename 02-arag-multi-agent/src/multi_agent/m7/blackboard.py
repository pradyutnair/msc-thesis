"""M7 Blackboard: CORAL collaborative evidence-pooling coordination.

Thread-safe shared state for multi-agent coordination with:
- Typed sub-questions with dependency graph management
- Evidence pool: merged documents from predecessor hops
- Adaptive execution levels for concurrent/sequential processing
- Placeholder resolution and collaborative answer correction
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class SQStatus(str, Enum):
    BLOCKED = "blocked"
    READY = "ready"
    ANSWERED = "answered"
    FAILED = "failed"


@dataclass
class SubQuestion:
    """A typed sub-question in the decomposition plan."""
    index: int
    text: str
    depends_on: list[int] = field(default_factory=list)
    answer_type: str = ""
    search_queries: list[str] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)
    answer: str | None = None
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)
    evidence_leads: list[str] = field(default_factory=list)
    status: SQStatus = SQStatus.READY
    # Optional fields carried from decomposer
    search_hints: list[str] = field(default_factory=list)
    known_entities: list[str] = field(default_factory=list)


@dataclass
class DecompositionPlan:
    """Output of the decomposer agent."""
    question_type: str  # comparison, bridge, intersection, single_hop
    expected_answer: str  # e.g., "a person name"
    sub_questions: list[SubQuestion] = field(default_factory=list)
    dependency_edges: list[list[int]] = field(default_factory=list)


class Blackboard:
    """Thread-safe shared state for CORAL multi-agent coordination."""

    def __init__(self, question: str):
        self.question = question
        self.plan: DecompositionPlan | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Plan management
    # ------------------------------------------------------------------

    def set_plan(self, plan: DecompositionPlan) -> None:
        """Set the decomposition plan and compute initial statuses."""
        with self._lock:
            self.plan = plan
            for sq in plan.sub_questions:
                if sq.depends_on:
                    sq.status = SQStatus.BLOCKED
                else:
                    sq.status = SQStatus.READY
                # Derive answer_type from unknown_entities
                if sq.unknown_entities:
                    sq.answer_type = sq.unknown_entities[0]

    # ------------------------------------------------------------------
    # Evidence Pool (CORAL core)
    # ------------------------------------------------------------------

    def get_evidence_pool(self, sq_index: int) -> list[dict[str, Any]]:
        """Return ALL documents from predecessor hops in the dependency
        chain (transitive closure), merged and deduplicated.

        This is the shared evidence pool that workers search before doing
        any new corpus retrieval.
        """
        with self._lock:
            if self.plan is None:
                return []
            sq = self._get_sq(sq_index)
            if sq is None:
                return []

            # Collect all ancestor indices via BFS
            ancestor_indices = self._get_all_ancestors(sq_index)
            if not ancestor_indices:
                return []

            # Merge documents from all ancestors, deduplicate by doc_id
            merged: dict[str, dict[str, Any]] = {}
            for anc_idx in ancestor_indices:
                anc_sq = self._get_sq(anc_idx)
                if anc_sq is None or not anc_sq.retrieved_docs:
                    continue
                for doc in anc_sq.retrieved_docs:
                    doc_id = doc.get("doc_id", "")
                    if doc_id and doc_id not in merged:
                        enriched = dict(doc)
                        enriched["source_worker"] = anc_idx
                        merged[doc_id] = enriched

            return list(merged.values())

    def _get_all_ancestors(self, sq_index: int) -> set[int]:
        """BFS to find ALL ancestors in the dependency chain (transitive)."""
        if self.plan is None:
            return set()

        ancestors: set[int] = set()
        queue: deque[int] = deque()

        sq = self._get_sq(sq_index)
        if sq is None:
            return ancestors

        # Seed with direct dependencies
        for dep_idx in sq.depends_on:
            queue.append(dep_idx)

        while queue:
            current = queue.popleft()
            if current in ancestors:
                continue
            ancestors.add(current)
            # Add this node's dependencies too
            dep_sq = self._get_sq(current)
            if dep_sq is not None:
                for dep_idx in dep_sq.depends_on:
                    if dep_idx not in ancestors:
                        queue.append(dep_idx)

        return ancestors

    # ------------------------------------------------------------------
    # Answer management
    # ------------------------------------------------------------------

    def post_answer(
        self,
        sq_index: int,
        answer: str,
        docs: list[dict[str, Any]],
        evidence_leads: list[str],
    ) -> None:
        """Post an answer for a sub-question and unblock dependents."""
        with self._lock:
            sq = self._get_sq(sq_index)
            if sq is None:
                return
            sq.answer = answer
            sq.retrieved_docs = docs
            sq.evidence_leads = evidence_leads
            sq.status = SQStatus.ANSWERED
            logger.info("SQ-%d answered: '%s'", sq_index, answer[:60])
            self._check_unblocks()

    def correct_answer(self, sq_index: int, new_answer: str) -> None:
        """Collaborative correction: update a previously posted answer."""
        with self._lock:
            sq = self._get_sq(sq_index)
            if sq is not None and sq.answer != new_answer:
                old = sq.answer
                sq.answer = new_answer
                logger.info(
                    "Collaborative correction SQ-%d: '%s' -> '%s'",
                    sq_index, old, new_answer,
                )

    def mark_failed(self, sq_index: int) -> None:
        """Mark a sub-question as failed and propagate to dependents."""
        with self._lock:
            sq = self._get_sq(sq_index)
            if sq is None:
                return
            sq.status = SQStatus.FAILED
            logger.warning("SQ-%d failed", sq_index)
            self._propagate_failure(sq_index)

    def get_answer(self, sq_index: int) -> str:
        """Get the current answer for a sub-question."""
        with self._lock:
            sq = self._get_sq(sq_index)
            return sq.answer if sq and sq.answer else ""

    def get_documents_for(self, sq_index: int) -> list[dict[str, Any]]:
        """Get the documents retrieved for a sub-question."""
        with self._lock:
            sq = self._get_sq(sq_index)
            if sq is not None and sq.retrieved_docs:
                return list(sq.retrieved_docs)
            return []

    # ------------------------------------------------------------------
    # Placeholder resolution
    # ------------------------------------------------------------------

    def resolve_placeholders(self, text: str) -> str:
        """Replace [answer_N] placeholders with actual answers."""
        with self._lock:
            return self._resolve_internal(text)

    def _resolve_internal(self, text: str) -> str:
        """Internal placeholder resolution (caller holds lock)."""
        if self.plan is None:
            return text
        for match in re.findall(r"\[answer_(\d+)\]", text):
            idx = int(match)
            sq = self._get_sq(idx)
            if sq and sq.answer:
                text = text.replace(f"[answer_{idx}]", sq.answer)
        return text

    # ------------------------------------------------------------------
    # Adaptive execution
    # ------------------------------------------------------------------

    def get_execution_levels(self) -> list[list[SubQuestion]]:
        """Compute execution levels for adaptive scheduling.

        Level 0 = sub-questions with no dependencies (run concurrently).
        Level 1 = sub-questions depending only on level 0.
        And so on.

        Returns a list of levels, each level is a list of SubQuestion objects.
        """
        with self._lock:
            if self.plan is None:
                return []

            sqs = self.plan.sub_questions
            if not sqs:
                return []

            # Map index -> SubQuestion
            sq_map = {sq.index: sq for sq in sqs}

            # Compute level for each SQ via topological layering
            levels_map: dict[int, int] = {}
            all_indices = [sq.index for sq in sqs]

            def _compute_level(idx: int, visited: set[int]) -> int:
                if idx in levels_map:
                    return levels_map[idx]
                if idx in visited:
                    # Cycle detected, treat as level 0
                    return 0
                visited.add(idx)
                sq = sq_map.get(idx)
                if sq is None or not sq.depends_on:
                    levels_map[idx] = 0
                    return 0
                max_dep_level = 0
                for dep_idx in sq.depends_on:
                    dep_level = _compute_level(dep_idx, visited)
                    max_dep_level = max(max_dep_level, dep_level)
                levels_map[idx] = max_dep_level + 1
                return max_dep_level + 1

            for idx in all_indices:
                _compute_level(idx, set())

            # Group by level
            max_level = max(levels_map.values()) if levels_map else 0
            result: list[list[SubQuestion]] = []
            for level in range(max_level + 1):
                level_sqs = [
                    sq_map[idx] for idx, lvl in levels_map.items()
                    if lvl == level and idx in sq_map
                ]
                if level_sqs:
                    # Sort within a level by index for determinism
                    level_sqs.sort(key=lambda sq: sq.index)
                    result.append(level_sqs)

            return result

    # ------------------------------------------------------------------
    # Synthesis support
    # ------------------------------------------------------------------

    def get_all_answers(self) -> dict[int, str]:
        """Return all answered sub-question answers."""
        with self._lock:
            if self.plan is None:
                return {}
            return {
                sq.index: sq.answer
                for sq in self.plan.sub_questions
                if sq.status == SQStatus.ANSWERED and sq.answer
            }

    def get_synthesis_context(self) -> dict[str, Any]:
        """Build context for the synthesizer agent."""
        with self._lock:
            if self.plan is None:
                return {"question": self.question}
            sqs = []
            for sq in self.plan.sub_questions:
                sqs.append({
                    "index": sq.index,
                    "text": sq.text,
                    "status": sq.status.value,
                    "answer": sq.answer,
                    "answer_type": sq.answer_type,
                    "depends_on": sq.depends_on,
                })
            return {
                "question": self.question,
                "question_type": self.plan.question_type,
                "expected_answer": self.plan.expected_answer,
                "sub_questions": sqs,
                "answers": {
                    sq.index: sq.answer
                    for sq in self.plan.sub_questions
                    if sq.answer
                },
            }

    def get_snapshot(self) -> dict[str, Any]:
        """Full state snapshot for diagnostics."""
        with self._lock:
            if self.plan is None:
                return {"question": self.question, "plan": None}
            return {
                "question": self.question,
                "question_type": self.plan.question_type,
                "expected_answer": self.plan.expected_answer,
                "sub_questions": [
                    {
                        "index": sq.index,
                        "text": sq.text,
                        "status": sq.status.value,
                        "answer": sq.answer,
                        "answer_type": sq.answer_type,
                        "depends_on": sq.depends_on,
                        "num_docs": len(sq.retrieved_docs),
                        "evidence_leads": sq.evidence_leads,
                    }
                    for sq in self.plan.sub_questions
                ],
            }

    # ------------------------------------------------------------------
    # Internal helpers (must hold lock)
    # ------------------------------------------------------------------

    def _get_sq(self, index: int) -> SubQuestion | None:
        if self.plan is None:
            return None
        for sq in self.plan.sub_questions:
            if sq.index == index:
                return sq
        return None

    def _check_unblocks(self) -> None:
        """Unblock sub-questions whose dependencies are all answered."""
        if self.plan is None:
            return
        answered_ids = {
            sq.index for sq in self.plan.sub_questions
            if sq.status == SQStatus.ANSWERED
        }
        for sq in self.plan.sub_questions:
            if sq.status != SQStatus.BLOCKED:
                continue
            if all(dep in answered_ids for dep in sq.depends_on):
                sq.status = SQStatus.READY
                logger.info("Unblocked SQ-%d", sq.index)

    def _propagate_failure(self, failed_idx: int) -> None:
        """Propagate failure to blocked dependents."""
        if self.plan is None:
            return
        for sq in self.plan.sub_questions:
            if sq.status == SQStatus.BLOCKED and failed_idx in sq.depends_on:
                sq.status = SQStatus.FAILED
                logger.warning("Propagated failure to SQ-%d", sq.index)
                self._propagate_failure(sq.index)
