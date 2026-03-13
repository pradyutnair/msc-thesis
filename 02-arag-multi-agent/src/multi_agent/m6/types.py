"""M6 dataclasses: sub-questions, evidence, entities, and pipeline result."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SubQuestionStatus(str, Enum):
    """Lifecycle status of a sub-question on the blackboard."""

    BLOCKED = "blocked"            # Dependencies unresolved
    READY = "ready"                # Dependencies met, unclaimed
    CLAIMED = "claimed"            # Retriever working on it
    EVIDENCE_FOUND = "evidence_found"  # Retriever done, awaiting verification
    VERIFIED = "verified"          # Critic approved
    FAILED = "failed"              # Max attempts exhausted
    NEEDS_RETRY = "needs_retry"    # Critic rejected, retry available


@dataclass
class M6SubQuestion:
    """A sub-question in the blackboard's search plan (DAG node)."""

    id: int
    text: str
    dependencies: list[int] = field(default_factory=list)
    known_entities: list[str] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)
    search_hints: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    status: SubQuestionStatus = SubQuestionStatus.READY
    claimed_by: str | None = None
    answer: str | None = None
    attempt_count: int = 0
    max_attempts: int = 3


@dataclass
class EvidenceEntry:
    """A piece of evidence supporting a sub-question answer."""

    id: str                        # "ev_{sq_id}_{counter}"
    sub_question_id: int
    content: str
    source_chunk_id: str
    relevance_score: float
    verified: bool = False
    retriever_id: str = ""


@dataclass
class EntityEntry:
    """A resolved entity posted to the blackboard's entity registry."""

    name: str                      # e.g. "answer_0"
    value: str                     # e.g. "Germany"
    source_evidence_id: str
    confidence: float = 1.0
    verified: bool = False


@dataclass
class KnowledgeGap:
    """A gap identified by the Critic that needs filling."""

    sub_question_id: int
    description: str
    suggested_query: str = ""


@dataclass
class Contradiction:
    """A contradiction detected between evidence or sub-answers."""

    sub_question_ids: list[int] = field(default_factory=list)
    description: str = ""
    resolution: str | None = None


@dataclass
class ExecutionLogEntry:
    """A single log entry in the blackboard's execution history."""

    tick: int
    agent_id: str
    action: str
    details: str = ""


@dataclass
class M6PipelineResult:
    """End-to-end result from the M6 pipeline.

    Compatible with eval scripts via qid/pred_answer/gold_answer fields.
    """

    qid: str = ""
    question: str = ""
    gold_answer: str = ""
    pred_answer: str = ""
    question_type: str = "unknown"
    num_sub_questions: int = 0
    num_workers: int = 0
    total_ticks: int = 0
    total_tokens: int = 0
    wall_clock_seconds: float = 0.0
    backtrack_count: int = 0
    sub_question_details: list[dict[str, Any]] = field(default_factory=list)
    entity_registry: dict[str, str] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    verified_count: int = 0
    failed_count: int = 0
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    knowledge_gaps: list[dict[str, Any]] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str = ""
    error: str | None = None
