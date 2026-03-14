"""Adaptive agency dataclasses: retrieval modes, sub-questions, evidence, entities, pipeline result."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    """Retrieval paradigm assigned to each sub-question by the planner."""

    STRUCTURED = "structured"
    AGENTIC = "agentic"
    AGGREGATE = "aggregate"


# ── Pydantic schemas for decomposer output (used for validation + guided decoding) ──

class DecomposedSubQuestion(BaseModel):
    """Single sub-question in the planner's decomposition output."""

    index: int
    text: str
    mode: Literal["structured", "agentic", "aggregate"] = "structured"
    search_queries: list[str] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)
    known_entities: list[str] = Field(default_factory=list)
    unknown_entities: list[str] = Field(default_factory=list)
    depends_on: list[int] = Field(default_factory=list)


class DecompositionResult(BaseModel):
    """Full decomposition output from the planner LLM."""

    question_type: Literal["comparison", "bridge", "intersection", "single_hop"]
    expected_answer: str = Field(
        description="What TYPE of answer the original question expects, e.g. 'a person name', 'a film name', 'yes or no', 'a year'."
    )
    sub_questions: list[DecomposedSubQuestion]
    dependency_edges: list[list[int]] = Field(default_factory=list)


class SubQuestionStatus(str, Enum):
    """Lifecycle status of a sub-question on the blackboard."""

    BLOCKED = "blocked"
    READY = "ready"
    CLAIMED = "claimed"
    EVIDENCE_FOUND = "evidence_found"
    VERIFIED = "verified"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"


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
    mode: RetrievalMode = RetrievalMode.STRUCTURED
    status: SubQuestionStatus = SubQuestionStatus.READY
    claimed_by: str | None = None
    answer: str | None = None
    attempt_count: int = 0
    max_attempts: int = 3


@dataclass
class EvidenceEntry:
    """A piece of evidence supporting a sub-question answer."""

    id: str
    sub_question_id: int
    content: str
    source_chunk_id: str
    relevance_score: float
    verified: bool = False
    retriever_id: str = ""


@dataclass
class EntityEntry:
    """A resolved entity posted to the blackboard's entity registry."""

    name: str
    value: str
    source_evidence_id: str
    confidence: float = 1.0
    verified: bool = False


@dataclass
class ExecutionLogEntry:
    """A single log entry in the blackboard's execution history."""

    tick: int
    agent_id: str
    action: str
    details: str = ""


@dataclass
class M6PipelineResult:
    """End-to-end result from the adaptive agency pipeline."""

    qid: str = ""
    question: str = ""
    gold_answer: str = ""
    pred_answer: str = ""
    question_type: str = "unknown"
    expected_answer: str = ""
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
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    termination_reason: str = ""
    error: str | None = None
    mode_distribution: dict[str, int] = field(default_factory=dict)
    mode_tokens: dict[str, int] = field(default_factory=dict)
    decomposition_text: str = ""
