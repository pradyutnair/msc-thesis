from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


WorkerRole = Literal["bridge", "attribute", "disambiguation", "refuter"]
ClaimStatus = Literal["proposed", "supported", "contested", "rejected"]


@dataclass
class ClaimRecord:
    id: int
    entity: str
    relation: str
    value: str
    status: ClaimStatus = "proposed"
    confidence: float = 0.0
    supporting_chunk_ids: list[str] = field(default_factory=list)
    source_task_ids: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class FrontierItem:
    id: int
    role_hint: WorkerRole
    goal: str
    query_hints: list[str] = field(default_factory=list)
    depends_on_claim_ids: list[int] = field(default_factory=list)
    status: Literal["open", "done", "blocked"] = "open"
    priority: int = 1
    notes: str = ""


@dataclass
class WorkerTask:
    id: int
    role: WorkerRole
    goal: str
    sub_question: str
    query_hints: list[str] = field(default_factory=list)
    depends_on_claim_ids: list[int] = field(default_factory=list)
    source_frontier_id: int | None = None
    status: Literal["pending", "done", "failed"] = "pending"
    answer: str = ""
    confidence: float = 0.0
    evidence_doc_ids: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class AgentMessage:
    task_id: int
    role: WorkerRole
    message_type: Literal["proposal", "evidence_update", "challenge", "resolution"]
    content: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManagerDecision:
    action: Literal[
        "spawn_bridge_worker",
        "spawn_attribute_worker",
        "spawn_disambiguation_worker",
        "request_refutation",
        "merge_claim",
        "compose_answer",
        "terminate",
    ]
    rationale: str = ""
    task_goal: str = ""
    sub_question: str = ""
    query_hints: list[str] = field(default_factory=list)
    depends_on_claim_ids: list[int] = field(default_factory=list)
    frontier_id: int | None = None
    claim_ids: list[int] = field(default_factory=list)


@dataclass
class WorkerExtraction:
    proposed_claims: list[ClaimRecord] = field(default_factory=list)
    frontier_updates: list[FrontierItem] = field(default_factory=list)
    message: str = ""
    parse_ok: bool = True


@dataclass
class ClaimMergeResult:
    updates: list[dict[str, Any]] = field(default_factory=list)
    parse_ok: bool = True


@dataclass
class BlackboardState:
    frontier: list[FrontierItem] = field(default_factory=list)
    claims: list[ClaimRecord] = field(default_factory=list)
    tasks: list[WorkerTask] = field(default_factory=list)
    messages: list[AgentMessage] = field(default_factory=list)
    draft_answer: str = ""
    draft_supporting_claim_ids: list[int] = field(default_factory=list)

    def supported_claim_ids(self) -> set[int]:
        return {claim.id for claim in self.claims if claim.status == "supported"}
