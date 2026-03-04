"""Dataclasses for the MA²RAG multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class SubQuestion:
    """A single sub-question produced by the Decomposer."""

    index: int
    text: str
    search_hints: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)
    placeholder: str | None = None


@dataclass
class DecompositionPlan:
    """Full output of the Decomposer for one question."""

    question_type: Literal["comparison", "bridge", "single_hop"]
    sub_questions: list[SubQuestion] = field(default_factory=list)
    dependency_edges: list[tuple[int, int]] = field(default_factory=list)

    raw_llm_output: str = ""
    parse_retries: int = 0


@dataclass
class AgentResult:
    """Result from a single search agent handling one sub-question."""

    sub_question_index: int
    answer: str
    evidence_doc_ids: list[str] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    loops: int = 0
    total_tokens: int = 0
    wall_clock_seconds: float = 0.0
    confidence: float = 1.0
    error: str | None = None

    # Full text of every chunk this agent read — populated by SearchAgent.
    # Each entry: {"id": str, "text": str}
    retrieved_chunks: list[dict] = field(default_factory=list)

    # Reliability / synthesis helpers
    unsupported_answer: bool = False
    extracted_evidence: list[str] = field(default_factory=list)
    evidence_count: int = 0


@dataclass
class ScoutResult:
    """Result from the OSPREY Phase 1 Scout agent."""

    answer: str
    chunks: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    is_confident: bool = False
    agent_result: AgentResult | None = None
    wall_clock_seconds: float = 0.0


@dataclass
class CachedDocument:
    """A document stored in the shared evidence cache."""

    doc_id: str
    text: str
    embedding: np.ndarray | None = None
    source_agent: int = -1
    retrieval_score: float = 0.0


@dataclass
class CacheAnalytics:
    """Statistics from the evidence cache."""

    total_puts: int = 0
    duplicate_hits: int = 0
    cross_agent_reuses: int = 0
    unique_docs: int = 0
    total_gets: int = 0
    get_hit_rate: float = 0.0


@dataclass
class PipelineResult:
    """End-to-end result from the multi-agent pipeline."""

    question: str
    decomposition: DecompositionPlan | None = None
    agent_results: dict[int, AgentResult] = field(default_factory=dict)
    final_answer: str = ""
    cache_analytics: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    wall_clock_seconds: float = 0.0

    question_type: str = "single_hop"
    num_sub_questions: int = 0
    num_waves: int = 0
    aggregator_tokens: int = 0
    error: str | None = None

    # Retry / verification metadata
    pass_id: str = "pass1"
    retry_trigger_reasons: list[str] = field(default_factory=list)
    verifier_parse_ok: bool | None = None

    # OSPREY-specific fields
    scout_answer: str = ""
    scout_confidence: float = 0.0
    osprey_fast_exit: bool = False
    scout_chunks: list[dict] = field(default_factory=list)
