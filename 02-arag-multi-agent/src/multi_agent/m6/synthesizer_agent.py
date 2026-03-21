"""SynthesizerAgent: combines verified evidence into a final answer.

Acts once when ALL sub-questions are VERIFIED or FAILED and the planner
has signaled allow_synthesis. Builds structured evidence blocks, runs
synthesis LLM call, optional consistency check, and terminates the blackboard.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from pathlib import Path
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.m6.answer_utils import extract_answer, is_refusal, normalize_answer
from multi_agent.m6.autonomous_agent import AutonomousAgent
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.types import SubQuestionStatus


def _answer_type_matches(answer: str, expected_answer: str, question: str) -> bool:
    """Check if answer plausibly matches the expected answer type.

    Returns False only for clear mismatches (person name when date expected, etc.).
    Returns True when unsure — we only reject obvious mismatches.
    """
    if not answer or not expected_answer:
        return True

    ans_lower = answer.lower().strip()
    exp_lower = expected_answer.lower().strip()
    q_lower = question.lower().strip()

    # Detect if answer looks like a year/date
    import re
    is_date_answer = bool(re.match(r"^\d{3,4}$", ans_lower) or
                          re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}\s+\w+\s+\d{4})\b", ans_lower, re.IGNORECASE))

    # Detect if question expects a date/time
    expects_date = any(w in exp_lower for w in ["year", "date", "when", "month", "time"]) or q_lower.startswith("when ")

    # Detect if question expects a place
    expects_place = any(w in exp_lower for w in ["place", "location", "city", "country", "county", "region", "where"]) or q_lower.startswith("where ")

    # Detect if question expects a person
    expects_person = any(w in exp_lower for w in ["person", "who", "name of a person"]) or q_lower.startswith("who ")

    # Reject: expects a date but got a non-date string with no digits and no month names
    _month_names = {"january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december",
                    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
                    "mid-january", "mid-february", "mid-march", "mid-april", "mid-may", "mid-june",
                    "mid-july", "mid-august", "mid-september", "mid-october", "mid-november", "mid-december"}
    has_date_content = any(c.isdigit() for c in ans_lower) or any(m in ans_lower for m in _month_names)
    if expects_date and not has_date_content:
        return False

    # Reject: expects a place but got what looks like a person name (two capitalized words, no place indicators)
    if expects_place and not expects_person:
        # Simple heuristic: if the answer has no place-like words and the question isn't asking for a person
        place_indicators = ["county", "city", "river", "lake", "ocean", "mountain", "island",
                           "delta", "valley", "sea", "gulf", "bay", "province", "state",
                           "district", "region", "peninsula", "strait", "channel"]
        if not any(w in ans_lower for w in place_indicators) and not any(c.isdigit() for c in ans_lower):
            # Could still be a place name like "Paris" or "Tokyo" — only reject if very short
            # and question clearly asks for a geographic feature
            if any(w in q_lower for w in ["body of water", "river", "lake", "ocean", "sea", "gulf"]):
                water_words = ["river", "lake", "ocean", "sea", "gulf", "bay", "delta",
                              "strait", "channel", "creek", "stream", "waterway"]
                if not any(w in ans_lower for w in water_words):
                    return False

    return True

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesizer.txt"
_CONSISTENCY_PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesizer_consistency.txt"


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
        if not observation.get("allow_synthesis", False):
            return False
        sqs = observation.get("sub_questions", [])
        if not sqs:
            return False

        terminal_statuses = {SubQuestionStatus.VERIFIED.value, SubQuestionStatus.FAILED.value}
        return all(sq["status"] in terminal_statuses for sq in sqs)

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        self._acted = True
        question = observation["question"]
        sub_questions = observation["sub_questions"]
        verified_evidence = observation["verified_evidence"]
        entity_registry = observation["entity_registry"]
        expected_answer = observation.get("expected_answer", "an entity")

        evidence_blocks = self._build_evidence_blocks(sub_questions, verified_evidence, entity_registry)

        prompt = self._prompt_template
        prompt = prompt.replace("{question}", question)
        prompt = prompt.replace("{evidence_blocks}", evidence_blocks)
        prompt = prompt.replace("{entity_registry}", self._format_entities(entity_registry, sub_questions))
        prompt = prompt.replace("{expected_answer}", expected_answer or "an entity")
        messages = [{"role": "user", "content": prompt}]

        total_tokens = 0
        try:
            response = await self._async_chat(messages=messages, tools=None, temperature=0.0)
            raw = response["message"].get("content", "")
            total_tokens += int(response.get("cost", 0.0) * 1_000_000)
        except Exception as exc:
            logger.error("Synthesizer LLM error: %s", exc)
            answer = await blackboard.salvage_answer()
            answer = normalize_answer(answer, question)
            await blackboard.set_final_answer(answer)
            await blackboard.terminate("SYNTHESIZED_FALLBACK")
            return 0

        answer = extract_answer(raw)

        if not answer or is_refusal(answer):
            answer = await blackboard.salvage_answer()
            logger.info("Synthesizer: extraction empty/refusal, salvaged: '%s'", answer[:80])

        answer = normalize_answer(answer, question)

        # v25: Reject answers that clearly mismatch expected type
        if answer and not _answer_type_matches(answer, expected_answer, question):
            logger.info("Synthesizer: type mismatch, answer '%s' doesn't match expected '%s'",
                        answer[:40], expected_answer[:40])
            answer = ""

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
        """Format entity registry with sub-question context."""
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

        revised = extract_answer(raw) if "FINAL ANSWER" in raw.upper() else ""
        revised = normalize_answer(revised, question) if revised else ""

        # Only accept revision if it looks like a clean entity answer
        _sentence_starts = ("the proposed", "the answer", "the correct", "based on",
                            "according to", "the evidence", "neither", "there is no")
        is_sentence = any(revised.lower().startswith(s) for s in _sentence_starts)

        if (revised
            and len(revised) < 80
            and not is_sentence
            and revised.lower() != answer.lower()
            and not is_refusal(revised)):
            logger.info("Consistency check revised: %s -> %s", answer[:40], revised[:40])
            return revised, tokens
        return answer, tokens
        return answer, tokens
