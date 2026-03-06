from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.m6_types import (
    ClaimMergeResult,
    ClaimRecord,
    FrontierItem,
    ManagerDecision,
    WorkerExtraction,
    WorkerTask,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MANAGER_PROMPT = _PROMPTS_DIR / "m6_manager.txt"
_EXTRACT_PROMPT = _PROMPTS_DIR / "m6_extract_claims.txt"
_MERGE_PROMPT = _PROMPTS_DIR / "m6_merge_claims.txt"
_ANSWER_PROMPT = _PROMPTS_DIR / "m6_answer.txt"

_ALLOWED_ACTIONS = {
    "spawn_bridge_worker",
    "spawn_attribute_worker",
    "spawn_disambiguation_worker",
    "request_refutation",
    "merge_claim",
    "compose_answer",
    "terminate",
}


def _clean_json(raw: str) -> str:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<(?:think|thnk)[^>]*>.*?</(?:think|thnk)>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(?:think|thnk)[^>]*>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    if "{" in text and "}" in text:
        candidates.append(text[text.find("{") : text.rfind("}") + 1])
    if "[" in text and "]" in text:
        candidates.append(text[text.find("[") : text.rfind("]") + 1])
    out: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in out:
            out.append(candidate)
            out.append(re.sub(r",(\s*[}\]])", r"\1", candidate))
    return out


def _parse_json_like(raw: str) -> Any:
    text = _clean_json(raw)
    last_error: Exception | None = None
    for candidate in _json_candidates(text):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception as exc:
                last_error = exc
    raise ValueError(f"Unable to parse JSON-like output: {last_error}")


def _clamp_confidence(value: object, default: float = 0.0) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value_f))


class M6Orchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        manager_prompt: str | Path | None = None,
        extract_prompt: str | Path | None = None,
        merge_prompt: str | Path | None = None,
        answer_prompt: str | Path | None = None,
    ):
        self.llm = llm_client
        self.manager_prompt = Path(manager_prompt) if manager_prompt else _MANAGER_PROMPT
        self.extract_prompt = Path(extract_prompt) if extract_prompt else _EXTRACT_PROMPT
        self.merge_prompt = Path(merge_prompt) if merge_prompt else _MERGE_PROMPT
        self.answer_prompt = Path(answer_prompt) if answer_prompt else _ANSWER_PROMPT

        self._manager_template = self.manager_prompt.read_text(encoding="utf-8")
        self._extract_template = self.extract_prompt.read_text(encoding="utf-8")
        self._merge_template = self.merge_prompt.read_text(encoding="utf-8")
        self._answer_template = self.answer_prompt.read_text(encoding="utf-8")

    async def decide_action(self, question: str, board_state: str) -> ManagerDecision:
        prompt = self._manager_template.replace("{question}", question).replace("{board_state}", board_state)
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        raw = (response.get("message", {}) or {}).get("content", "")
        try:
            data = _parse_json_like(raw)
            action = str(data.get("action", "")).strip()
            if action not in _ALLOWED_ACTIONS:
                raise ValueError(f"invalid action: {action}")
            return ManagerDecision(
                action=action,
                rationale=str(data.get("rationale", "")).strip(),
                task_goal=str(data.get("task_goal", "")).strip(),
                sub_question=str(data.get("sub_question", "")).strip(),
                query_hints=[str(x).strip() for x in data.get("query_hints", []) if str(x).strip()],
                depends_on_claim_ids=[int(x) for x in data.get("depends_on_claim_ids", []) if str(x).strip()],
                frontier_id=int(data["frontier_id"]) if str(data.get("frontier_id", "")).strip() else None,
                claim_ids=[int(x) for x in data.get("claim_ids", []) if str(x).strip()],
            )
        except Exception as exc:
            logger.warning("Manager decision parse fallback: %s", exc)
            return ManagerDecision(action="terminate", rationale=f"fallback:{exc}")

    async def extract_worker_updates(
        self,
        question: str,
        task: WorkerTask,
        board_state: str,
        agent_answer: str,
        extracted_evidence: list[str],
        chunk_snippets: list[dict[str, Any]],
    ) -> WorkerExtraction:
        prompt = (
            self._extract_template
            .replace("{question}", question)
            .replace("{board_state}", board_state)
            .replace("{task_role}", task.role)
            .replace("{task_goal}", task.goal)
            .replace("{task_sub_question}", task.sub_question)
            .replace("{agent_answer}", agent_answer or "")
            .replace("{evidence_lines}", "\n".join(f"- {line}" for line in extracted_evidence) or "- None")
            .replace("{chunk_snippets}", self._format_chunks(chunk_snippets))
        )
        try:
            response = await self.llm.async_chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
            )
            raw = (response.get("message", {}) or {}).get("content", "")
            data = _parse_json_like(raw)
            proposed_claims = [
                self._claim_from_payload(item, task.id)
                for item in data.get("proposed_claims", [])
                if isinstance(item, dict)
            ]
            frontier_updates = [
                self._frontier_from_payload(item, task.role)
                for item in data.get("frontier_updates", [])
                if isinstance(item, dict)
            ]
            return WorkerExtraction(
                proposed_claims=[claim for claim in proposed_claims if claim.value],
                frontier_updates=[frontier for frontier in frontier_updates if frontier.goal],
                message=str(data.get("message", "")).strip(),
                parse_ok=True,
            )
        except Exception as exc:
            logger.warning("Worker extraction fallback for task %s: %s", task.id, exc)
            fallback_claims: list[ClaimRecord] = []
            if agent_answer.strip():
                fallback_claims.append(
                    ClaimRecord(
                        id=-1,
                        entity=task.goal[:80],
                        relation=task.role,
                        value=agent_answer.strip(),
                        status="proposed",
                        confidence=0.35 if extracted_evidence else 0.2,
                        supporting_chunk_ids=[
                            str(chunk.get("id", ""))
                            for chunk in chunk_snippets[:3]
                            if str(chunk.get("id", ""))
                        ],
                        source_task_ids=[task.id],
                        notes=f"fallback_extract:{exc}",
                    )
                )
            return WorkerExtraction(
                proposed_claims=fallback_claims,
                message=f"fallback_extract:{exc}",
                parse_ok=False,
            )

    async def merge_claims(self, question: str, board_state: str, claims: list[ClaimRecord]) -> ClaimMergeResult:
        claim_block = "\n".join(
            json.dumps(self._claim_dict(claim), ensure_ascii=False)
            for claim in claims
        ) or "No claims provided."
        prompt = (
            self._merge_template
            .replace("{question}", question)
            .replace("{board_state}", board_state)
            .replace("{claim_block}", claim_block)
        )
        try:
            response = await self.llm.async_chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
            )
            raw = (response.get("message", {}) or {}).get("content", "")
            data = _parse_json_like(raw)
            updates = []
            for item in data.get("updates", []):
                if not isinstance(item, dict):
                    continue
                updates.append(
                    {
                        "claim_id": int(item["claim_id"]),
                        "status": str(item.get("status", "proposed")).strip(),
                        "confidence": _clamp_confidence(item.get("confidence"), default=0.0),
                        "notes": str(item.get("notes", "")).strip(),
                    }
                )
            return ClaimMergeResult(updates=updates, parse_ok=True)
        except Exception as exc:
            logger.warning("Claim merge fallback: %s", exc)
            updates = []
            for claim in claims:
                updates.append(
                    {
                        "claim_id": claim.id,
                        "status": "supported"
                        if claim.supporting_chunk_ids and claim.value
                        else ("contested" if claim.relation == "refuter" else "proposed"),
                        "confidence": max(claim.confidence, 0.25 if claim.value else 0.0),
                        "notes": f"fallback_merge:{exc}",
                    }
                )
            return ClaimMergeResult(updates=updates, parse_ok=False)

    async def compose_answer(self, question: str, board_state: str) -> tuple[str, list[int], str]:
        prompt = self._answer_template.replace("{question}", question).replace("{board_state}", board_state)
        try:
            response = await self.llm.async_chat(
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
            )
            raw = (response.get("message", {}) or {}).get("content", "")
            data = _parse_json_like(raw)
            return (
                str(data.get("answer", "")).strip(),
                [int(x) for x in data.get("supporting_claim_ids", []) if str(x).strip()],
                str(data.get("notes", "")).strip(),
            )
        except Exception as exc:
            logger.warning("Answer composition fallback: %s", exc)
            return "", [], f"fallback_answer:{exc}"

    @staticmethod
    def _claim_from_payload(item: dict[str, Any], task_id: int) -> ClaimRecord:
        return ClaimRecord(
            id=-1,
            entity=str(item.get("entity", "")).strip(),
            relation=str(item.get("relation", "fact")).strip() or "fact",
            value=str(item.get("value", item.get("answer", ""))).strip(),
            status="proposed",
            confidence=_clamp_confidence(item.get("confidence"), default=0.0),
            supporting_chunk_ids=[str(x) for x in item.get("supporting_chunk_ids", []) if str(x).strip()],
            source_task_ids=[task_id],
            notes=str(item.get("notes", "")).strip(),
        )

    @staticmethod
    def _frontier_from_payload(item: dict[str, Any], default_role: str) -> FrontierItem:
        role = str(item.get("role_hint", default_role)).strip().lower()
        if role not in {"bridge", "attribute", "disambiguation", "refuter"}:
            role = default_role if default_role in {"bridge", "attribute", "disambiguation", "refuter"} else "bridge"
        return FrontierItem(
            id=-1,
            role_hint=role,
            goal=str(item.get("goal", "")).strip(),
            query_hints=[str(x).strip() for x in item.get("query_hints", []) if str(x).strip()],
            depends_on_claim_ids=[int(x) for x in item.get("depends_on_claim_ids", []) if str(x).strip()],
            priority=max(1, min(3, int(item.get("priority", 1) or 1))),
            notes=str(item.get("notes", "")).strip(),
        )

    @staticmethod
    def _claim_dict(claim: ClaimRecord) -> dict[str, Any]:
        return {
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

    @staticmethod
    def _format_chunks(chunks: list[dict[str, Any]], limit: int = 6) -> str:
        lines = []
        for chunk in chunks[:limit]:
            lines.append(f"[Chunk {chunk.get('id', '?')}] {str(chunk.get('text', ''))[:650]}")
        return "\n\n".join(lines) if lines else "No chunks retrieved."
