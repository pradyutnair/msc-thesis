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


class Verifier:
    """Check if retrieved evidence is sufficient to answer the question.

    Combines D2Plan's noise filtering with gap-targeted follow-up detection.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
    ):
        self.llm = llm_client
        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    async def verify(
        self,
        question: str,
        plan: SagePlan,
        chunks: list[dict],
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

        Returns
        -------
        VerificationResult
        """
        # Build evidence text
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

        # Build plan text
        plan_lines = []
        for task in plan.tasks:
            plan_lines.append(
                f"Task {task.id}: {task.goal} (query: '{task.query}')"
            )
        plan_text = "\n".join(plan_lines)

        # Use .replace() instead of .format() because the template contains
        # literal { } in JSON examples that would confuse str.format().
        prompt = (
            self._prompt_template
            .replace("{question}", question)
            .replace("{question_type}", plan.question_type)
            .replace("{retrieval_plan}", plan_text)
            .replace("{evidence}", evidence_text)
            .replace("{expected_answer_type}", plan.expected_answer_type)
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.llm.async_chat(
                messages=messages, tools=None, temperature=0.0,
            )
            raw = response["message"].get("content", "")
            return self._parse_response(raw, chunks)
        except Exception as exc:
            logger.error("Verifier error: %s", exc)
            # On error, assume sufficient (don't block pipeline)
            return VerificationResult(sufficient=True, verified_chunks=chunks)

    def _parse_response(
        self, raw: str, all_chunks: list[dict]
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
            logger.warning("Verifier JSON parse failed, assuming sufficient")
            return VerificationResult(
                sufficient=True, verified_chunks=all_chunks, raw_llm_output=raw,
            )

        sufficient = data.get("sufficient", True)

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
            # If no verified list, keep all non-irrelevant
            verified_chunks = [
                c for c in all_chunks
                if str(c.get("id", "")) not in irrelevant_ids
            ]

        # If filtering removed everything, keep all
        if not verified_chunks:
            verified_chunks = all_chunks

        gaps = data.get("gaps", [])

        return VerificationResult(
            sufficient=sufficient,
            verified_chunks=verified_chunks,
            irrelevant_chunk_ids=list(irrelevant_ids),
            gaps=gaps,
            raw_llm_output=raw,
        )
