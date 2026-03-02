"""SAGE Synthesizer: generates final answer from clean verified evidence."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from arag.core.llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "sage_synth.txt"


class Synthesizer:
    """Generate final answer from verified evidence only.

    Receives ONLY the question and verified evidence chunks — no search
    history, no tool trajectories, no agent reasoning. This clean context
    eliminates "lost in the middle" and synthesis noise.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
    ):
        self.llm = llm_client
        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    async def synthesize(
        self,
        question: str,
        verified_chunks: list[dict],
        agent_results: dict | None = None,
        expected_answer_type: str | None = None,
    ) -> tuple[str, float]:
        """Synthesize answer from verified evidence.

        Parameters
        ----------
        question : str
        verified_chunks : list[dict]
            Verified evidence chunks with 'id' and 'text'.
        agent_results : dict, optional
            Mapping of task_id -> AgentResult with intermediate answers.
        expected_answer_type : str, optional
            Hint from planner: person, location, date, number, yes_no, entity.

        Returns (answer, cost).
        """
        # Build agent findings summary
        findings_lines = []
        if agent_results:
            for tid in sorted(agent_results.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                ar = agent_results[tid]
                answer = getattr(ar, "answer", "") or ""
                if answer:
                    findings_lines.append(f"- Agent {tid}: {answer}")
        findings_text = (
            "\n".join(findings_lines) if findings_lines else "None"
        )

        evidence_lines = []
        for chunk in verified_chunks:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")
            evidence_lines.append(f"[Document {cid}]\n{text}")
        evidence_text = (
            "\n\n---\n\n".join(evidence_lines)
            if evidence_lines
            else "No evidence available."
        )

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
        logger.info("Synthesized -> '%s'", answer[:80])
        return answer, cost

    @staticmethod
    def _extract_answer(raw: str) -> str:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        m = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
        if m:
            answer = m.group(1).strip()
        else:
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            answer = lines[-1] if lines else raw.strip()
        # Strip XML-style tag leaks
        answer = re.sub(r"</?(?:answer|your[_ ]?answer|response|result)[^>]*>", "", answer, flags=re.IGNORECASE).strip()
        return answer
