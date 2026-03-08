"""Shared Blackboard state for autonomous multi-agent collaborative search."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hop:
    id: int
    question: str
    resolved_question: str = ""
    status: str = "pending"  # pending | investigating | resolved | stuck
    depends_on: list[int] = field(default_factory=list)
    answer: str | None = None
    evidence: list[dict] = field(default_factory=list)  # [{id, text, source_agent}]
    confidence: float = 0.0
    assigned_to: str | None = None
    attempt_count: int = 0


@dataclass
class EntityInfo:
    name: str
    facts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source_chunks: list[str] = field(default_factory=list)
    discovered_by: str = ""


@dataclass
class Blackboard:
    question: str
    question_type: str = "single_hop"  # comparison | bridge | single_hop
    expected_answer_type: str = "entity"
    hop_chain: list[Hop] = field(default_factory=list)
    entity_kb: dict[str, EntityInfo] = field(default_factory=dict)
    strategy_notes: str = ""

    def get_actionable_hops(self) -> list[Hop]:
        """Return hops that are pending and have all dependencies resolved."""
        actionable = []
        for hop in self.hop_chain:
            if hop.status != "pending":
                continue
            deps_resolved = all(
                self.hop_chain[d].status == "resolved"
                for d in hop.depends_on
                if d < len(self.hop_chain)
            )
            if deps_resolved:
                actionable.append(hop)
        return actionable

    def resolve_hop(
        self,
        hop_id: int,
        answer: str,
        evidence: list[dict],
        confidence: float,
    ) -> None:
        if hop_id < 0 or hop_id >= len(self.hop_chain):
            return
        hop = self.hop_chain[hop_id]
        hop.answer = answer
        hop.evidence = evidence
        hop.confidence = confidence
        hop.status = "resolved" if confidence > 0.0 else "stuck"

    def resolve_placeholders(self, hop: Hop) -> str:
        """Replace [hop_N] placeholders with resolved answers."""
        text = hop.question
        for match in re.finditer(r"\[hop_(\d+)\]", text):
            ref_id = int(match.group(1))
            if ref_id < len(self.hop_chain) and self.hop_chain[ref_id].answer:
                text = text.replace(match.group(0), self.hop_chain[ref_id].answer)
        return text

    def add_entity(
        self,
        name: str,
        facts: list[str],
        confidence: float,
        chunks: list[str],
        agent_id: str,
    ) -> None:
        key = name.lower().strip()
        if key in self.entity_kb:
            existing = self.entity_kb[key]
            seen = set(existing.facts)
            for f in facts:
                if f not in seen:
                    existing.facts.append(f)
                    seen.add(f)
            existing.confidence = max(existing.confidence, confidence)
            seen_chunks = set(existing.source_chunks)
            for c in chunks:
                if c not in seen_chunks:
                    existing.source_chunks.append(c)
                    seen_chunks.add(c)
        else:
            self.entity_kb[key] = EntityInfo(
                name=name,
                facts=list(facts),
                confidence=confidence,
                source_chunks=list(chunks),
                discovered_by=agent_id,
            )

    def get_context_for_investigator(self, hop: Hop) -> str:
        """Build a formatted summary of the blackboard for an investigator."""
        parts = []

        # Resolved hops
        resolved = [h for h in self.hop_chain if h.status == "resolved" and h.id != hop.id]
        if resolved:
            parts.append("=== RESOLVED HOPS (findings from other investigators) ===")
            for h in resolved:
                parts.append(f"Hop {h.id}: {h.resolved_question or h.question}")
                parts.append(f"  Answer: {h.answer}")
                if h.evidence:
                    top_evidence = h.evidence[:2]
                    for ev in top_evidence:
                        text = str(ev.get("text", ""))[:200]
                        parts.append(f"  Evidence: {text}")
                parts.append("")

        # Entity KB
        if self.entity_kb:
            parts.append("=== KNOWN ENTITIES ===")
            for key, info in self.entity_kb.items():
                facts_str = "; ".join(info.facts[:5])
                parts.append(f"- {info.name}: {facts_str}")
            parts.append("")

        if not parts:
            return "No prior findings from other agents."

        return "\n".join(parts)

    def revise_hop(self, hop_id: int, new_question: str) -> None:
        if hop_id < 0 or hop_id >= len(self.hop_chain):
            return
        hop = self.hop_chain[hop_id]
        hop.question = new_question
        hop.status = "pending"
        hop.answer = None
        hop.evidence = []
        hop.confidence = 0.0

    def add_hop(self, question: str, depends_on: list[int]) -> Hop:
        new_id = len(self.hop_chain)
        hop = Hop(
            id=new_id,
            question=question,
            depends_on=depends_on,
        )
        self.hop_chain.append(hop)
        return hop

    def get_evidence_summary(self) -> str:
        """Build a summary of all evidence for answer generation."""
        parts = []
        for hop in self.hop_chain:
            status = hop.status.upper()
            q = hop.resolved_question or hop.question
            parts.append(f"Hop {hop.id} [{status}]: {q}")
            if hop.answer:
                parts.append(f"  Answer: {hop.answer} (confidence: {hop.confidence:.1f})")
            if hop.evidence:
                for ev in hop.evidence[:3]:
                    text = str(ev.get("text", ""))[:300]
                    parts.append(f"  Evidence: {text}")
            parts.append("")

        if self.entity_kb:
            parts.append("=== ENTITY KNOWLEDGE BASE ===")
            for key, info in self.entity_kb.items():
                parts.append(f"{info.name}:")
                for fact in info.facts[:5]:
                    parts.append(f"  - {fact}")
            parts.append("")

        return "\n".join(parts)

    def get_state_summary(self) -> str:
        """Short summary for strategist review."""
        parts = [f"Question: {self.question}",
                 f"Type: {self.question_type} | Expected: {self.expected_answer_type}",
                 f"Hops: {len(self.hop_chain)}", ""]

        for hop in self.hop_chain:
            q = hop.resolved_question or hop.question
            deps = f" (depends on: {hop.depends_on})" if hop.depends_on else ""
            parts.append(f"Hop {hop.id} [{hop.status}]{deps}: {q}")
            if hop.answer:
                parts.append(f"  -> Answer: {hop.answer} (conf: {hop.confidence:.2f}, attempt: {hop.attempt_count})")
            parts.append("")

        if self.entity_kb:
            parts.append("Entity KB:")
            for key, info in self.entity_kb.items():
                facts_str = "; ".join(info.facts[:3])
                parts.append(f"  {info.name}: {facts_str}")

        return "\n".join(parts)

    def all_hops_resolved(self) -> bool:
        return all(h.status == "resolved" for h in self.hop_chain)

    def apply_revisions(self, revisions: list[dict]) -> None:
        """Apply revision actions from strategist review."""
        for rev in revisions:
            action = rev.get("action", "")
            if action == "revise_hop":
                hop_id = rev.get("hop_id", -1)
                new_q = rev.get("new_question", "")
                if new_q:
                    self.revise_hop(hop_id, new_q)
            elif action == "add_hop":
                q = rev.get("question", "")
                deps = rev.get("depends_on", [])
                if q:
                    self.add_hop(q, deps)
            elif action == "retry_hop":
                hop_id = rev.get("hop_id", -1)
                if 0 <= hop_id < len(self.hop_chain):
                    self.hop_chain[hop_id].status = "pending"
                    self.hop_chain[hop_id].attempt_count += 1
