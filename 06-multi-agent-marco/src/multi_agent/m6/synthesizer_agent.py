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
            answer = normalize_answer(answer, question)
            await blackboard.set_final_answer(answer)
            await blackboard.terminate("SYNTHESIZED_FALLBACK")
            return 0

        answer = extract_answer(raw)

        if not answer or is_refusal(answer):
            answer = await blackboard.salvage_answer()
            logger.info("Synthesizer: extraction empty/refusal, salvaged: '%s'", answer[:80])

        answer = normalize_answer(answer, question)

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

        revised = extract_answer(raw) if "FINAL ANSWER" in raw.upper() else raw.strip()
        if len(revised) > 100:
            lines = [line.strip() for line in revised.split("\n") if line.strip()]
            revised = lines[-1] if lines else revised
        revised = normalize_answer(revised, question)

        if revised and revised.lower() != answer.lower() and not is_refusal(revised):
            logger.info("Consistency check revised: '%s' -> '%s'", answer[:40], revised[:40])
            return revised, tokens
        return answer, tokens
