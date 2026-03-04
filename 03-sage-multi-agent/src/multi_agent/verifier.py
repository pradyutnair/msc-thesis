"""SAGE Verifier: checks evidence sufficiency and identifies gaps."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from arag.core.llm import LLMClient
from multi_agent.planner import SagePlan

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "sage_verifier.txt"


@dataclass
class VerificationResult:
    """Output of the Verifier agent."""

    sufficient: bool
    verified_chunks: list[dict] = field(default_factory=list)
    irrelevant_chunk_ids: list[str] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)
    raw_llm_output: str = ""
    parse_ok: bool = True
    failure_reason: str | None = None


class Verifier:
    """Check if retrieved evidence is sufficient to answer the question.

    Combines D2Plan's noise filtering with gap-targeted follow-up detection.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
        fail_open_on_error: bool = False,
    ):
        self.llm = llm_client
        self.fail_open_on_error = fail_open_on_error
        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "").lower()).strip()
        return text

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {
            tok
            for tok in re.findall(r"[a-z0-9]+", Verifier._normalize_text(text))
            if len(tok) >= 3
        }

    @classmethod
    def _is_supported_by_task_chunks(
        cls,
        evidence_line: str,
        task_chunks: list[dict],
        overlap_threshold: float = 0.15,
    ) -> bool:
        if not evidence_line.strip() or not task_chunks:
            return False

        e_tokens = cls._token_set(evidence_line)
        if not e_tokens:
            return False

        best = 0.0
        for chunk in task_chunks:
            c_tokens = cls._token_set(chunk.get("text", ""))
            if not c_tokens:
                continue
            inter = len(e_tokens & c_tokens)
            union = max(len(e_tokens | c_tokens), 1)
            jaccard = inter / union
            best = max(best, jaccard)
            if jaccard >= overlap_threshold:
                return True
        return best >= overlap_threshold

    def _build_fallback_gaps(
        self,
        question: str,
        plan: SagePlan,
        chunks: list[dict],
        extracted_evidence_by_task: dict[int, list[str]] | None = None,
        reason: str = "verifier_fallback",
    ) -> list[dict]:
        """Build deterministic fallback gaps when verifier output is invalid."""
        chunks_by_task: dict[int, list[dict]] = {}
        for chunk in chunks:
            tid_raw = chunk.get("task_id")
            try:
                tid = int(tid_raw)
            except (TypeError, ValueError):
                continue
            chunks_by_task.setdefault(tid, []).append(chunk)

        gaps: list[dict] = []
        extracted_evidence_by_task = extracted_evidence_by_task or {}

        for task in plan.tasks:
            task_chunks = chunks_by_task.get(task.id, [])
            has_task_chunks = bool(task_chunks)
            unsupported_extractions = False

            extracted_lines = extracted_evidence_by_task.get(task.id, [])
            if extracted_lines and task_chunks:
                for line in extracted_lines:
                    if not self._is_supported_by_task_chunks(line, task_chunks):
                        unsupported_extractions = True
                        break

            if not has_task_chunks or unsupported_extractions:
                query = task.query.strip() or task.goal.strip() or question.strip()
                if plan.expected_answer_type and plan.expected_answer_type != "entity":
                    query = f"{query}, {plan.expected_answer_type}"
                gaps.append(
                    {
                        "description": (
                            f"Task {task.id} unresolved"
                            if not unsupported_extractions
                            else f"Task {task.id} extraction unsupported by evidence"
                        ),
                        "query": query,
                        "method": task.search_method or "keyword",
                        "reason": reason,
                        "task_id": task.id,
                    }
                )

        if not gaps:
            query = plan.tasks[-1].query if plan.tasks else question
            if plan.expected_answer_type and plan.expected_answer_type != "entity":
                query = f"{query}, {plan.expected_answer_type}"
            gaps.append(
                {
                    "description": "Verifier fallback gap",
                    "query": query,
                    "method": "keyword",
                    "reason": reason,
                    "task_id": plan.tasks[-1].id if plan.tasks else -1,
                }
            )

        return gaps[:3]

    def _find_unsupported_extraction_tasks(
        self,
        chunks: list[dict],
        extracted_evidence_by_task: dict[int, list[str]] | None,
    ) -> list[int]:
        """Return task IDs whose extracted evidence is not supported by task chunks."""
        extracted_evidence_by_task = extracted_evidence_by_task or {}
        if not extracted_evidence_by_task:
            return []

        chunks_by_task: dict[int, list[dict]] = {}
        for chunk in chunks:
            tid_raw = chunk.get("task_id")
            try:
                tid = int(tid_raw)
            except (TypeError, ValueError):
                continue
            chunks_by_task.setdefault(tid, []).append(chunk)

        unsupported: list[int] = []
        for task_id_raw, lines in extracted_evidence_by_task.items():
            try:
                task_id = int(task_id_raw)
            except (TypeError, ValueError):
                continue

            normalized_lines = [str(line).strip() for line in (lines or []) if str(line).strip()]
            if not normalized_lines:
                continue

            task_chunks = chunks_by_task.get(task_id, [])
            if not task_chunks:
                unsupported.append(task_id)
                continue

            for line in normalized_lines:
                if not self._is_supported_by_task_chunks(line, task_chunks):
                    unsupported.append(task_id)
                    break

        return sorted(set(unsupported))

    async def verify(
        self,
        question: str,
        plan: SagePlan,
        chunks: list[dict],
        extracted_evidence_by_task: dict[int, list[str]] | None = None,
    ) -> VerificationResult:
        """Verify evidence sufficiency.

        Parameters
        ----------
        question : str
            The original question.
        plan : SagePlan
            The retrieval plan from the Planner.
        chunks : list[dict]
            All retrieved chunks, each with 'id' and 'text' keys.
        extracted_evidence_by_task : dict[int, list[str]], optional
            Per-task extracted evidence bullets from worker agents.

        Returns
        -------
        VerificationResult
        """
        evidence_lines = []
        for chunk in chunks:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")[:800]
            task_id = chunk.get("task_id", "?")
            evidence_lines.append(f"[Chunk {cid} | Task {task_id}]\n{text}")
        evidence_text = (
            "\n\n---\n\n".join(evidence_lines)
            if evidence_lines
            else "No evidence retrieved."
        )

        plan_lines = []
        for task in plan.tasks:
            plan_lines.append(
                f"Task {task.id}: {task.goal} (query: '{task.query}')"
            )
        plan_text = "\n".join(plan_lines)

        extracted_lines: list[str] = []
        extracted_evidence_by_task = extracted_evidence_by_task or {}
        for task_id, items in sorted(extracted_evidence_by_task.items()):
            for item in items:
                if item.strip():
                    extracted_lines.append(f"[Task {task_id}] {item.strip()}")
        extracted_text = (
            "\n".join(extracted_lines)
            if extracted_lines
            else "No extracted evidence provided."
        )

        prompt = (
            self._prompt_template
            .replace("{question}", question)
            .replace("{question_type}", plan.question_type)
            .replace("{retrieval_plan}", plan_text)
            .replace("{evidence}", evidence_text)
            .replace("{expected_answer_type}", plan.expected_answer_type)
            .replace("{extracted_evidence}", extracted_text)
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm.async_chat(
                messages=messages, tools=None, temperature=0.0,
            )
            raw = response["message"].get("content", "")
            parsed = self._parse_response(raw, chunks)

            unsupported_extraction_tasks = self._find_unsupported_extraction_tasks(
                chunks,
                extracted_evidence_by_task,
            )
            if unsupported_extraction_tasks:
                parsed.sufficient = False
                fallback_gaps = self._build_fallback_gaps(
                    question,
                    plan,
                    chunks,
                    extracted_evidence_by_task=extracted_evidence_by_task,
                    reason="unsupported_extracted_evidence",
                )
                existing = parsed.gaps if isinstance(parsed.gaps, list) else []
                seen = {
                    (str(g.get("task_id", "")), str(g.get("query", "")))
                    for g in existing
                }
                for gap in fallback_gaps:
                    key = (str(gap.get("task_id", "")), str(gap.get("query", "")))
                    if key not in seen:
                        existing.append(gap)
                        seen.add(key)
                parsed.gaps = existing[:3]
                parsed.failure_reason = (
                    "unsupported_extracted_evidence"
                    if not parsed.failure_reason
                    else f"{parsed.failure_reason},unsupported_extracted_evidence"
                )

            if not parsed.sufficient and not parsed.gaps:
                parsed.gaps = self._build_fallback_gaps(
                    question,
                    plan,
                    chunks,
                    extracted_evidence_by_task=extracted_evidence_by_task,
                    reason="parsed_insufficient_no_gaps",
                )
            return parsed
        except Exception as exc:
            logger.error("Verifier error: %s", exc)
            if self.fail_open_on_error:
                return VerificationResult(
                    sufficient=True,
                    verified_chunks=chunks,
                    parse_ok=False,
                    failure_reason=f"verifier_error:{exc}",
                )

            return VerificationResult(
                sufficient=False,
                verified_chunks=chunks,
                gaps=self._build_fallback_gaps(
                    question,
                    plan,
                    chunks,
                    extracted_evidence_by_task=extracted_evidence_by_task,
                    reason="verifier_exception",
                ),
                parse_ok=False,
                failure_reason=f"verifier_error:{exc}",
            )

    def _parse_response(
        self,
        raw: str,
        all_chunks: list[dict],
    ) -> VerificationResult:
        raw_clean = raw.strip()
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```$", "", raw_clean)
        raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL)
        raw_clean = re.sub(r"<think>.*", "", raw_clean, flags=re.DOTALL)
        raw_clean = raw_clean.strip()

        try:
            data = json.loads(raw_clean)
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict, got {type(data).__name__}")
        except (json.JSONDecodeError, ValueError):
            logger.warning("Verifier JSON parse failed")
            return VerificationResult(
                sufficient=bool(self.fail_open_on_error),
                verified_chunks=all_chunks,
                raw_llm_output=raw,
                parse_ok=False,
                failure_reason="verifier_json_parse_failed",
            )

        sufficient = bool(data.get("sufficient", True))

        verified_ids = set(
            str(cid) for cid in data.get("verified_chunks", [])
        )
        irrelevant_ids = set(
            str(cid) for cid in data.get("irrelevant_chunks", [])
        )

        if verified_ids:
            verified_chunks = [
                c for c in all_chunks
                if str(c.get("id", "")) in verified_ids
            ]
        else:
            verified_chunks = [
                c for c in all_chunks
                if str(c.get("id", "")) not in irrelevant_ids
            ]

        if not verified_chunks:
            verified_chunks = all_chunks

        gaps_raw = data.get("gaps", [])
        gaps = gaps_raw if isinstance(gaps_raw, list) else []

        return VerificationResult(
            sufficient=sufficient,
            verified_chunks=verified_chunks,
            irrelevant_chunk_ids=list(irrelevant_ids),
            gaps=gaps,
            raw_llm_output=raw,
            parse_ok=True,
            failure_reason=None,
        )
