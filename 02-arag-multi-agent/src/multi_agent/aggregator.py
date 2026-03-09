"""Aggregator: 3-phase evidence synthesis for multi-agent answers."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.evidence_cache import EvidenceCache
from multi_agent.types import AgentResult, DecompositionPlan

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "aggregator.txt"

_CHAIN_INSTRUCTIONS = {
    "comparison": (
        "Compare the sub-answers directly. Identify the relevant attribute "
        "for each entity and determine the answer to the comparison question."
    ),
    "bridge": (
        "Chain the sub-answers sequentially. The answer to sub-question 0 "
        "feeds into sub-question 1, and so on. Follow the chain to reach "
        "the final answer."
    ),
    "single_hop": (
        "The sub-answer directly answers the original question. Verify it "
        "against the evidence and restate it."
    ),
}


class Aggregator:
    """Aggregate sub-question answers into a final answer.

    3 phases: evidence assembly → CoT synthesis → self-verification (optional).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        evidence_cache: EvidenceCache | None = None,
        enable_self_verify: bool = True,
        prompt_path: str | Path | None = None,
    ):
        self.llm = llm_client
        self.cache = evidence_cache
        self.enable_self_verify = enable_self_verify

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    async def _assemble_evidence(
        self, plan: DecompositionPlan, agent_results: dict[int, AgentResult],
    ) -> str:
        blocks: list[str] = []
        for sq in plan.sub_questions:
            result = agent_results.get(sq.index)
            if result is None:
                blocks.append(f"### Sub-Q {sq.index}: {sq.text}\n**Answer**: (no result)\n**Evidence**: None\n")
                continue
            evidence_text = await self._get_evidence_for_agent(result)
            blocks.append(f"### Sub-Q {sq.index}: {sq.text}\n**Answer**: {result.answer}\n**Evidence**:\n{evidence_text}\n")
        return "\n".join(blocks)

    async def _get_evidence_for_agent(self, result: AgentResult) -> str:
        lines: list[str] = []
        if self.cache is not None and self.cache.enabled:
            for doc_id in result.evidence_doc_ids:
                doc = await self.cache.get_by_id(doc_id)
                if doc:
                    lines.append(f"[{doc.doc_id}] {doc.text[:500]}")
        if not lines:
            for entry in result.trajectory:
                if entry.get("tool_name") == "read_chunk":
                    text = entry.get("tool_result", "")
                    if text and "(already read)" not in text:
                        lines.append(text[:500])
        return "\n".join(lines[:5]) if lines else "(No evidence retrieved)"

    async def _synthesize(self, question: str, plan: DecompositionPlan, evidence_blocks: str) -> tuple[str, float]:
        chain_instruction = _CHAIN_INSTRUCTIONS.get(plan.question_type, _CHAIN_INSTRUCTIONS["single_hop"])
        prompt = self._prompt_template.format(
            question=question, question_type=plan.question_type,
            evidence_blocks=evidence_blocks, chain_instruction=chain_instruction,
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
        raw = response["message"].get("content", "")
        cost = response.get("cost", 0.0)
        answer = self._extract_final_answer(raw)
        logger.info("Synthesis answer: '%s'", answer[:80])
        return answer, cost

    @staticmethod
    def _extract_final_answer(raw: str) -> str:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        match = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines[-1] if lines else raw.strip()

    async def _self_verify(self, question: str, answer: str, evidence_blocks: str) -> tuple[str, float]:
        prompt = (
            f"Question: {question}\nProposed Answer: {answer}\n\n"
            f"Evidence:\n{evidence_blocks}\n\n"
            f"Is the proposed answer correct and fully supported by the evidence? "
            f"If yes, restate it. If no, provide the corrected answer.\n"
            f"Reply with ONLY the final answer (no explanation)."
        )
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
        raw = response["message"].get("content", "")
        cost = response.get("cost", 0.0)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        verified = raw.strip()
        if verified and verified != answer:
            logger.info("Self-verify revised: '%s' → '%s'", answer[:40], verified[:40])
            return verified, cost
        return answer, cost

    async def aggregate(
        self, question: str, plan: DecompositionPlan, agent_results: dict[int, AgentResult],
    ) -> tuple[str, int]:
        """Returns (final_answer, approx_token_cost)."""
        if plan.question_type == "single_hop" and len(agent_results) == 1 and not self.enable_self_verify:
            result = next(iter(agent_results.values()))
            return result.answer, 0

        evidence_blocks = await self._assemble_evidence(plan, agent_results)
        answer, synth_cost = await self._synthesize(question, plan, evidence_blocks)
        total_cost = synth_cost

        if self.enable_self_verify:
            answer, verify_cost = await self._self_verify(question, answer, evidence_blocks)
            total_cost += verify_cost

        approx_tokens = int(total_cost * 1_000_000) if total_cost > 0 else 0
        return answer, approx_tokens
