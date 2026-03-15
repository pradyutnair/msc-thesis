"""SAGE Synthesizer: generates final answer from clean verified evidence."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from arag.core.llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "sage_synth.txt"

_REFUSAL_PATTERNS = [
    "cannot be determined",
    "insufficient information",
    "not mentioned",
    "no evidence",
    "unable to determine",
    "no information",
    "cannot determine",
    "not enough information",
    "information is not available",
    "not provided in",
    "no relevant information",
    "not available",
]


class Synthesizer:
    """Generate final answer from verified evidence only.

    Receives the question and verified evidence chunks. If extracted evidence
    is available from workers, it is prioritized to reduce synthesis noise.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
    ):
        self.llm = llm_client
        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    @staticmethod
    def _agent_has_evidence(agent_result: object) -> bool:
        evidence_doc_ids = getattr(agent_result, "evidence_doc_ids", []) or []
        retrieved_chunks = getattr(agent_result, "retrieved_chunks", []) or []
        unsupported = bool(getattr(agent_result, "unsupported_answer", False))
        return (bool(evidence_doc_ids) or bool(retrieved_chunks)) and not unsupported

    def _build_extracted_evidence_text(self, agent_results: dict | None) -> str:
        if not agent_results:
            return ""
        lines: list[str] = []
        for tid in sorted(agent_results.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            ar = agent_results[tid]
            extracted = getattr(ar, "extracted_evidence", []) or []
            if not extracted:
                continue
            for bullet in extracted:
                b = str(bullet).strip()
                if b:
                    lines.append(f"[Task {tid}] {b}")
        return "\n".join(lines)

    def _build_raw_evidence_text(self, verified_chunks: list[dict]) -> str:
        evidence_lines = []
        for chunk in verified_chunks:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")
            evidence_lines.append(f"[Document {cid}]\n{text}")
        return "\n\n---\n\n".join(evidence_lines) if evidence_lines else "No evidence available."

    async def synthesize(
        self,
        question: str,
        verified_chunks: list[dict],
        agent_results: dict | None = None,
        expected_answer_type: str | None = None,
    ) -> tuple[str, float]:
        """Synthesize answer from verified evidence.

        Returns
        -------
        tuple[str, float]
            (answer, usd_cost)
        """
        findings_lines = []
        if agent_results:
            for tid in sorted(agent_results.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                ar = agent_results[tid]
                answer = getattr(ar, "answer", "") or ""
                if answer:
                    findings_lines.append(f"- Agent {tid}: {answer}")
        findings_text = "\n".join(findings_lines) if findings_lines else "None"

        extracted_text = self._build_extracted_evidence_text(agent_results)
        raw_evidence_text = self._build_raw_evidence_text(verified_chunks)
        if extracted_text:
            evidence_text = (
                "## Extracted Evidence (primary)\n"
                f"{extracted_text}\n\n"
                "## Verified Raw Evidence (fallback)\n"
                f"{raw_evidence_text}"
            )
        else:
            evidence_text = raw_evidence_text

        answer_type_hint = expected_answer_type or "entity"

        prompt = (
            self._prompt_template
            .replace("{question}", question)
            .replace("{agent_findings}", findings_text)
            .replace("{evidence}", evidence_text)
            .replace("{expected_answer_type}", answer_type_hint)
        )
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm.async_chat(
            messages=messages, tools=None, temperature=0.0,
        )
        raw = response["message"].get("content", "")
        cost = response.get("cost", 0.0)

        answer = self._extract_answer(raw)

        if self._is_empty_or_refusal(answer):
            logger.warning(
                "Empty/refusal answer detected ('%s'), retrying with stronger prompt",
                answer[:60],
            )
            retry_answer, retry_cost = await self._retry_synthesis(
                question,
                findings_text,
                evidence_text,
                answer_type_hint,
            )
            cost += retry_cost
            if retry_answer and not self._is_empty_or_refusal(retry_answer):
                answer = retry_answer
                logger.info("Retry succeeded -> '%s'", answer[:80])
            else:
                logger.warning("Retry also produced empty/refusal, keeping original")

        logger.info("Synthesized -> '%s'", answer[:80])
        return answer, cost

    async def _retry_synthesis(
        self,
        question: str,
        findings_text: str,
        evidence_text: str,
        answer_type_hint: str,
    ) -> tuple[str, float]:
        """Retry synthesis with a forceful prompt that demands a concrete answer."""
        retry_prompt = (
            f"## Question\n{question}\n\n"
            f"## Expected Answer Type\n{answer_type_hint}\n\n"
            f"## Agent Findings\n{findings_text}\n\n"
            f"## Evidence\n{evidence_text}\n\n"
            "## CRITICAL INSTRUCTION\n"
            "Your previous attempt refused to answer or returned an empty answer. "
            "You MUST provide a concrete answer grounded in the evidence.\n\n"
            "Rules:\n"
            "- Prefer direct evidence; fallback to strongest supported agent finding.\n"
            "- Do not output XML/markdown wrappers.\n"
            "- Respond with ONLY the answer, 1-8 words.\n\n"
            "FINAL ANSWER:"
        )
        messages = [{"role": "user", "content": retry_prompt}]
        response = await self.llm.async_chat(
            messages=messages,
            tools=None,
            temperature=0.2,
        )
        raw = response["message"].get("content", "")
        cost = response.get("cost", 0.0)
        answer = self._extract_answer(raw)
        return answer, cost

    @staticmethod
    def _is_empty_or_refusal(answer: str) -> bool:
        if not answer or not answer.strip():
            return True
        lower = answer.lower().strip()
        return any(pattern in lower for pattern in _REFUSAL_PATTERNS)

    @staticmethod
    def sanitize_answer(answer: str) -> str:
        cleaned = str(answer or "").strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        cleaned = re.sub(r"\*\*+", "", cleaned)
        cleaned = re.sub(r"`+", "", cleaned)
        cleaned = re.sub(r"^(?:final\s*answer\s*:|answer\s*:)+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^[-\s]+", "", cleaned).strip()
        cleaned = cleaned.strip("\"'")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _extract_answer(cls, raw: str) -> str:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)

        m = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
        else:
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            candidate = lines[-1] if lines else raw.strip()

        return cls.sanitize_answer(candidate)


def find_best_evidence_backed_answer(agent_results: dict) -> str:
    """Pick best fallback answer from evidence-backed agents only."""
    candidates: list[tuple[int, str]] = []
    for tid in sorted(agent_results.keys(), key=lambda x: int(x) if str(x).isdigit() else 0, reverse=True):
        ar = agent_results[tid]
        ans = str(getattr(ar, "answer", "") or "").strip()
        if not ans:
            continue
        if Synthesizer._is_empty_or_refusal(ans):
            continue
        if not Synthesizer._agent_has_evidence(ar):
            continue
        candidates.append((int(tid) if str(tid).isdigit() else -1, Synthesizer.sanitize_answer(ans)))

    return candidates[0][1] if candidates else ""
