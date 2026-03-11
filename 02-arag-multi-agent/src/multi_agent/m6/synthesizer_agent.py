"""SynthesizerAgent: combines verified evidence into a final answer.

Acts once when ALL sub-questions are VERIFIED or FAILED. Builds structured
evidence blocks, runs synthesis LLM call, optional consistency check,
extracts FINAL ANSWER, and terminates the blackboard.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from pathlib import Path
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.m6.autonomous_agent import AutonomousAgent
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.types import SubQuestionStatus

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesizer.txt"
_CONSISTENCY_PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesizer_consistency.txt"


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def _infer_expected_answer_type(question: str) -> str:
    q = (question or "").strip().lower()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had|can|could|would|should)\b", q):
        return "yes_no"
    if q.startswith("when ") or "what year" in q or "what date" in q:
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith("who "):
        return "person"
    if q.startswith("how many ") or q.startswith("how much "):
        return "number"
    return "entity"


def _normalize_answer(answer: str, question: str) -> str:
    """Full normalization pipeline matching gold label granularity."""
    text = (answer or "").strip()

    # 1. Strip LLM artifacts
    text = re.sub(r"^\s*(final answer\s*:|answer\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the answer is|answer is)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")

    # 2. Yes/no detection from question form
    expected_type = _infer_expected_answer_type(question)
    if expected_type == "yes_no":
        lowered = text.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"

    # 3. Strip sentence wrappers (conservative — only known patterns)
    for pattern in [
        r"^(.*?)\s+was\s+born\s+first\.?$",
        r"^(.*?)\s+was\s+(?:produced|released|published|created|formed|founded)\s+first\.?$",
        r"^(.*?)\s+(?:died|passed away)\s+(?:first|earlier|before)\.?$",
        r"^(.*?)\s+is\s+(?:the\s+)?(?:older|younger|taller|shorter|bigger|smaller)\.?$",
        r"^(.*?)\s+is\s+the\s+answer\.?$",
        r"^The\s+answer\s+is\s+(.+?)\.?$",
    ]:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            break

    # 4. Strip parenthetical qualifiers ("Paris (France)" → "Paris")
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)

    # 5. Strip trailing punctuation
    text = re.sub(r"\s*[\.,;:!?]+$", "", text)

    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Truncate verbose answers — if still > 60 chars, take first phrase
    if len(text) > 60:
        m = re.match(r"^(.{3,50}?)[.,;]", text)
        if m:
            text = m.group(1).strip()

    return text


def _is_refusal(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if lowered in ("none", "n/a", "unknown", "null", ""):
        return True
    patterns = [
        "cannot be determined", "insufficient information", "not mentioned",
        "no evidence", "unable to determine", "not enough information",
        "unknown", "cannot determine", "no information",
        "not found in the provided", "the question is invalid",
        "the question is asking", "not applicable",
        "no answer", "i don't know", "i cannot",
    ]
    return any(p in lowered for p in patterns)


class SynthesizerAgent(AutonomousAgent):
    """Synthesizes final answer from verified evidence on the blackboard."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
        consistency_prompt_path: str | Path | None = None,
        enable_consistency_check: bool = True,
    ):
        super().__init__(agent_id="synthesizer", agent_type="synthesizer")
        self.llm = llm_client
        self.enable_consistency_check = enable_consistency_check
        self._acted = False

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

        cons_path = Path(consistency_prompt_path) if consistency_prompt_path else _CONSISTENCY_PROMPT_PATH
        if cons_path.exists():
            self._consistency_template = cons_path.read_text(encoding="utf-8")
        else:
            self._consistency_template = None

    async def _async_chat(self, **kwargs):
        """Run synchronous llm.chat in executor to avoid blocking event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.llm.chat, **kwargs),
        )

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_synthesizer()

    def should_act(self, observation: dict[str, Any]) -> bool:
        if self._acted:
            return False
        sqs = observation.get("sub_questions", [])
        if not sqs:
            return False

        terminal_statuses = {SubQuestionStatus.VERIFIED.value, SubQuestionStatus.FAILED.value}
        n_total = len(sqs)
        n_terminal = sum(1 for sq in sqs if sq["status"] in terminal_statuses)

        # Only synthesize when ALL sub-questions are complete.
        # Partial synthesis was removed — it caused correctness issues
        # (e.g., "Are both X and Y Z?" needs all answers, not just one).
        return n_terminal == n_total

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        self._acted = True
        question = observation["question"]
        sub_questions = observation["sub_questions"]
        verified_evidence = observation["verified_evidence"]
        entity_registry = observation["entity_registry"]

        evidence_blocks = self._build_evidence_blocks(sub_questions, verified_evidence, entity_registry)

        prompt = self._prompt_template.format(
            question=question,
            evidence_blocks=evidence_blocks,
            entity_registry=self._format_entities(entity_registry, sub_questions),
        )
        messages = [{"role": "user", "content": prompt}]

        total_tokens = 0
        try:
            response = await self._async_chat(messages=messages, tools=None, temperature=0.0)
            raw = response["message"].get("content", "")
            total_tokens += int(response.get("cost", 0.0) * 1_000_000)
        except Exception as exc:
            logger.error("Synthesizer LLM error: %s", exc)
            answer = await blackboard.salvage_answer()
            answer = _normalize_answer(answer, question)
            await blackboard.set_final_answer(answer)
            await blackboard.terminate("SYNTHESIZED_FALLBACK")
            return 0

        answer = self._extract_answer(raw)

        if not answer or _is_refusal(answer):
            answer = await blackboard.salvage_answer()
            logger.info("Synthesizer: extraction empty/refusal, salvaged: '%s'", answer[:80])

        answer = _normalize_answer(answer, question)

        if self.enable_consistency_check and self._consistency_template and answer:
            answer, cons_tokens = await self._consistency_check(
                question, answer, evidence_blocks,
            )
            total_tokens += cons_tokens

        await blackboard.set_final_answer(answer)
        await blackboard.terminate("SYNTHESIZED")
        logger.info("Synthesizer: '%s'", answer[:80])

        return total_tokens

    def _build_evidence_blocks(
        self,
        sub_questions: list[dict[str, Any]],
        verified_evidence: list[dict[str, Any]],
        entity_registry: dict[str, str],
    ) -> str:
        """Build structured evidence blocks per sub-question."""
        ev_by_sq: dict[int, list[dict]] = {}
        for ev in verified_evidence:
            sq_id = ev["sub_question_id"]
            ev_by_sq.setdefault(sq_id, []).append(ev)

        blocks: list[str] = []
        for sq in sub_questions:
            sq_id = sq["id"]
            status = sq["status"]
            answer = sq.get("answer", "(no answer)")
            entity_key = f"answer_{sq_id}"
            resolved_answer = entity_registry.get(entity_key, answer)

            evidence_texts = []
            for ev in ev_by_sq.get(sq_id, []):
                evidence_texts.append(f"  [{ev['source_chunk_id']}] {ev['content'][:1500]}")

            evidence_str = "\n".join(evidence_texts) if evidence_texts else "  (no evidence)"

            block = (
                f"### Sub-Question {sq_id}: {sq['text']}\n"
                f"**Status**: {status}\n"
                f"**Answer**: {resolved_answer}\n"
                f"**Evidence**:\n{evidence_str}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)

    @staticmethod
    def _format_entities(
        entity_registry: dict[str, str],
        sub_questions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Format entity registry with sub-question context for clearer mapping."""
        if not entity_registry:
            return "None"
        if sub_questions:
            sq_map = {sq["id"]: sq["text"] for sq in sub_questions}
            lines = []
            for key, val in entity_registry.items():
                try:
                    sq_id = int(key.split("_")[1])
                    sq_text = sq_map.get(sq_id, "")
                    if sq_text:
                        lines.append(f"- SQ{sq_id} \"{sq_text[:80]}\": {val}")
                    else:
                        lines.append(f"- {key} = {val}")
                except (ValueError, IndexError):
                    lines.append(f"- {key} = {val}")
            return "\n".join(lines)
        return "\n".join(f"- {key} = {val}" for key, val in entity_registry.items())

    @staticmethod
    def _extract_answer(raw: str) -> str:
        """Extract FINAL ANSWER from LLM response."""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        # Strip markdown bold markers (LLM sometimes outputs **FINAL ANSWER**: ...)
        raw = raw.replace("**", "")
        match = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        return lines[-1] if lines else raw.strip()

    async def _consistency_check(
        self,
        question: str,
        answer: str,
        evidence_blocks: str,
    ) -> tuple[str, int]:
        """Optional second LLM call to verify answer consistency."""
        prompt = self._consistency_template.format(
            question=question,
            answer=answer,
            evidence_blocks=evidence_blocks,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._async_chat(messages=messages, tools=None, temperature=0.0)
            raw = response["message"].get("content", "")
            tokens = int(response.get("cost", 0.0) * 1_000_000)
        except Exception as exc:
            logger.error("Consistency check error: %s", exc)
            return answer, 0

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)

        revised = self._extract_answer(raw) if "FINAL ANSWER" in raw.upper() else raw.strip()
        if len(revised) > 100:
            lines = [l.strip() for l in revised.split("\n") if l.strip()]
            revised = lines[-1] if lines else revised
        revised = _normalize_answer(revised, question)
        # Don't let consistency check override with a refusal or empty answer
        if revised and revised.lower() != answer.lower() and not _is_refusal(revised):
            logger.info("Consistency check revised: '%s' → '%s'", answer[:40], revised[:40])
            return revised, tokens
        return answer, tokens
