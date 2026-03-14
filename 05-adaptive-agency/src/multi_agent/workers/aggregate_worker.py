"""AggregateWorker: synthesize answer from blackboard context only (no retrieval).

Used for final comparison/intersection hops where all evidence has already been
gathered by prior workers. Single LLM call using entity registry + prior answers.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.autonomous_agent import AutonomousAgent
from multi_agent.blackboard import Blackboard
from multi_agent.types import EvidenceEntry, RetrievalMode, SubQuestionStatus
from multi_agent.utils import clean_answer, resolve_placeholders

logger = logging.getLogger(__name__)

_AGGREGATE_PROMPT = """You are answering a sub-question using ONLY the context already gathered by other agents. Do NOT search for new information.

Original question: {original_question}
Expected answer type: {expected_answer}

Sub-question to answer: {sub_question}

Resolved answers from prior sub-questions:
{entity_context}

Evidence gathered so far:
{evidence_context}

Instructions:
- Your answer MUST be of the expected answer type above (e.g., if "a film name", output the film name — NOT a year or number).
- COMPARISON: Compare resolved answers numerically/logically, then output the ENTITY NAME that wins. Never output a date or number when the question asks "which".
- INTERSECTION: Find what the resolved answers have in common. Output the shared entity.
- YES/NO: If the question asks whether something is true, output "yes" or "no".

Concise answer (1-5 words, matching expected answer type):"""


class AggregateWorker(AutonomousAgent):
    """Blackboard-only synthesis worker for a specific sub-question. No retrieval."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        assigned_sq_id: int,
    ):
        super().__init__(agent_id=agent_id, agent_type="aggregate_worker")
        self.llm = llm_client
        self.assigned_sq_id = assigned_sq_id
        self._done = False
        self._last_epoch = 0

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_worker(self.agent_id)

    def should_act(self, observation: dict[str, Any]) -> bool:
        epoch = observation.get("redecomposition_epoch", 0)
        if epoch > self._last_epoch:
            self._done = False
            self._last_epoch = epoch
        if self._done:
            return False
        if observation["claimed_sub_question"] is not None:
            return True
        for sq in observation["available_sub_questions"]:
            if sq["id"] == self.assigned_sq_id:
                return True
        return False

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        sq_dict = observation["claimed_sub_question"]
        if sq_dict is None:
            claimed = await blackboard.claim_sub_question(self.assigned_sq_id, self.agent_id)
            if not claimed:
                return 0
            observation = await self.observe(blackboard)
            sq_dict = observation["claimed_sub_question"]
            if sq_dict is None:
                return 0

        sq_id = sq_dict["id"]
        entity_registry = observation["entity_registry"]
        resolved_text = resolve_placeholders(sq_dict["text"], entity_registry)
        original_question = observation.get("question", resolved_text)
        blackboard_context = observation.get("blackboard_context", "")

        expected_answer = observation.get("expected_answer", "") or "a short factual answer"

        entity_lines = []
        for key, val in entity_registry.items():
            entity_lines.append(f"- {key} = {val}")
        entity_context = "\n".join(entity_lines) if entity_lines else "No resolved entities yet."

        prompt = _AGGREGATE_PROMPT.format(
            original_question=original_question,
            expected_answer=expected_answer,
            sub_question=resolved_text,
            entity_context=entity_context,
            evidence_context=blackboard_context or "No evidence gathered yet.",
        )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                self.llm.chat,
                messages=[{"role": "user", "content": prompt}],
                tools=None,
                temperature=0.0,
            ),
        )

        raw = response["message"].get("content", "")
        tokens = int(response.get("cost", 0.0) * 1_000_000)

        answer = clean_answer(raw)

        await blackboard.post_evidence([], sq_id, answer, self.agent_id)
        is_usable = bool(answer) and answer.lower() not in ("unknown", "error", "")
        await blackboard.verify_sub_question(sq_id, verified=is_usable)
        await blackboard.record_mode_tokens(RetrievalMode.AGGREGATE, tokens)

        self._done = is_usable
        logger.info(
            "%s: SQ-%d -> '%s' (%d tokens, no retrieval)",
            self.agent_id, sq_id, answer[:60], tokens,
        )
        return tokens

