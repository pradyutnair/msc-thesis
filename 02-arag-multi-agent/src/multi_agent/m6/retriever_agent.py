"""RetrieverAgent: claims and answers sub-questions using BaseAgent ReAct loop.

Claim-before-work pattern: atomically claims a READY sub-question, resolves
[answer_N] placeholders from entity registry, runs BaseAgent with enriched
prompt, extracts evidence from trajectory, and posts results to blackboard.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from arag.agent.base import BaseAgent
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from multi_agent.m6.autonomous_agent import AutonomousAgent
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.types import EvidenceEntry, SubQuestionStatus

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "retriever.txt"


class RetrieverAgent(AutonomousAgent):
    """Claims sub-questions and runs ReAct search loops to find evidence."""

    def __init__(
        self,
        agent_id: str,
        llm_client: LLMClient,
        tools: ToolRegistry,
        prompt_path: str | Path | None = None,
        max_loops: int = 5,
        max_token_budget: int = 32_000,
        auto_verify: bool = False,
    ):
        super().__init__(agent_id=agent_id, agent_type="retriever")
        self.llm = llm_client
        self.tools = tools
        self.max_loops = max_loops
        self.max_token_budget = max_token_budget
        self.auto_verify = auto_verify

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

        # Track which sub-question is currently claimed
        self._current_sq_id: int | None = None

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_retriever(self.agent_id)

    def should_act(self, observation: dict[str, Any]) -> bool:
        # Act if we have a claimed sub-question or there are available ones
        if observation["claimed_sub_question"] is not None:
            return True
        return len(observation["available_sub_questions"]) > 0

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        sq_dict = observation["claimed_sub_question"]
        if sq_dict is None:
            sq_dict = await self._try_claim(observation, blackboard)
            if sq_dict is None:
                return 0

        sq_id = sq_dict["id"]
        self._current_sq_id = sq_id
        entity_registry = observation["entity_registry"]

        resolved_text = self._resolve_placeholders(sq_dict["text"], entity_registry)

        original_question = observation.get("question", resolved_text)
        blackboard_context = observation.get("blackboard_context", "No findings from other agents yet.")
        knowledge_gaps = observation.get("knowledge_gaps", {}).get(sq_id, [])

        system_prompt = self._build_prompt(
            resolved_text,
            entity_registry,
            original_question=original_question,
            blackboard_context=blackboard_context,
            knowledge_gaps=knowledge_gaps,
        )

        agent = BaseAgent(
            llm_client=self.llm,
            tools=self.tools,
            system_prompt=system_prompt,
            max_loops=self.max_loops,
            max_token_budget=self.max_token_budget,
        )

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, agent.run, resolved_text)
        except Exception as exc:
            logger.error("%s: BaseAgent error for SQ-%d: %s", self.agent_id, sq_id, exc)
            await blackboard.post_evidence([], sq_id, f"Error: {exc}", self.agent_id)
            self._current_sq_id = None
            return 0

        answer = result.get("answer", "")
        answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
        answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
        answer = re.sub(r"\*(.+?)\*", r"\1", answer)
        answer = answer.split("\n")[0].strip()
        answer = answer.strip().strip("\"'`*")
        answer = re.sub(r"\s*[\.,;:!?]+$", "", answer)
        trajectory = result.get("trajectory", [])
        total_tokens = result.get("total_retrieved_tokens", 0)

        evidence_entries = self._extract_evidence(sq_id, trajectory)
        await blackboard.post_evidence(evidence_entries, sq_id, answer, self.agent_id)

        # Entities are only posted by verify_sub_question (critic or auto-verify),
        # ensuring dependents only unblock after verification.
        if self.auto_verify and answer:
            await blackboard.verify_sub_question(sq_id, verified=True)

        logger.info(
            "%s: SQ-%d → '%s' (%d evidence entries, %d tokens)",
            self.agent_id, sq_id, answer[:60],
            len(evidence_entries), total_tokens,
        )

        self._current_sq_id = None
        return total_tokens

    async def _try_claim(
        self,
        observation: dict[str, Any],
        blackboard: Blackboard,
    ) -> dict[str, Any] | None:
        """Try to claim a READY or NEEDS_RETRY sub-question.

        Priority: retries first, then by number of downstream dependents
        (critical-path heuristic), then by ID.
        """
        available = observation["available_sub_questions"]
        available.sort(
            key=lambda sq: (
                0 if sq["status"] == SubQuestionStatus.NEEDS_RETRY.value else 1,
                -sq.get("num_dependents", 0),
                sq["id"],
            )
        )
        for sq_dict in available:
            claimed = await blackboard.claim_sub_question(sq_dict["id"], self.agent_id)
            if claimed:
                return sq_dict
        return None

    def _resolve_placeholders(self, text: str, entity_registry: dict[str, str]) -> str:
        """Replace [answer_N] placeholders with resolved entity values."""
        def replacer(match: re.Match) -> str:
            key = match.group(1)  # "answer_0"
            return entity_registry.get(key, match.group(0))

        return re.sub(r"\[(answer_\d+)\]", replacer, text)

    def _build_prompt(
        self,
        resolved_text: str,
        entity_registry: dict[str, str],
        original_question: str = "",
        blackboard_context: str = "",
        knowledge_gaps: list[dict] | None = None,
    ) -> str:
        """Build system prompt with cross-agent context and retry feedback."""
        context_parts = [blackboard_context] if blackboard_context else []
        if entity_registry:
            resolved = [f"{k} = {v}" for k, v in entity_registry.items()]
            context_parts.append("Resolved entities: " + ", ".join(resolved))

        if knowledge_gaps:
            gap_lines = ["Previous attempt feedback (use this to improve your search):"]
            for gap in knowledge_gaps:
                gap_lines.append(f"- {gap['description']}")
                if gap.get("suggested_query"):
                    gap_lines.append(f"  Try searching: {gap['suggested_query']}")
            context_parts.append("\n".join(gap_lines))

        full_context = "\n".join(context_parts) if context_parts else "No findings from other agents yet."

        return self._prompt_template.format(
            sub_question=resolved_text,
            original_question=original_question or resolved_text,
            blackboard_context=full_context,
        )

    def _extract_evidence(self, sq_id: int, trajectory: list[dict]) -> list[EvidenceEntry]:
        """Extract evidence entries from BaseAgent trajectory (read_chunk results)."""
        entries: list[EvidenceEntry] = []
        for step in trajectory:
            if step.get("tool_name") != "read_chunk":
                continue
            result_text = step.get("tool_result", "")
            if not result_text or "(already read)" in result_text:
                continue

            chunk_ids = step.get("arguments", {}).get(
                "chunk_ids",
                step.get("arguments", {}).get("chunk_id", []),
            )
            if isinstance(chunk_ids, (str, int)):
                chunk_ids = [str(chunk_ids)]

            for cid in chunk_ids:
                entries.append(EvidenceEntry(
                    id="",  # Will be set by blackboard
                    sub_question_id=sq_id,
                    content=result_text[:2000],
                    source_chunk_id=str(cid),
                    relevance_score=0.5,
                    retriever_id=self.agent_id,
                ))
        return entries
