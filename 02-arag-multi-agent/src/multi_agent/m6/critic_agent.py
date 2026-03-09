"""CriticAgent: per-hop verification, gap detection, and backtracking.

Acts when sub-questions have status EVIDENCE_FOUND. Single LLM call per
sub-question checks sufficiency, correctness, and consistency. Verdicts
trigger verify/retry/backtrack actions on the blackboard.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.m6.autonomous_agent import AutonomousAgent
from multi_agent.m6.blackboard import Blackboard
from multi_agent.m6.types import Contradiction, KnowledgeGap

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "critic.txt"


class CriticAgent(AutonomousAgent):
    """Verifies evidence and answers for sub-questions, triggers backtracking."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
        enable_backtracking: bool = True,
    ):
        super().__init__(agent_id="critic", agent_type="critic")
        self.llm = llm_client
        self.enable_backtracking = enable_backtracking

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_critic()

    def should_act(self, observation: dict[str, Any]) -> bool:
        return len(observation["pending_sub_questions"]) > 0

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        total_tokens = 0
        pending_sqs = observation["pending_sub_questions"]
        sq_evidence = observation["sub_question_evidence"]
        entity_registry = observation["entity_registry"]
        verified_answers = observation.get("verified_answers", {})

        for sq_dict in pending_sqs:
            sq_id = sq_dict["id"]
            evidence = sq_evidence.get(sq_id, [])
            answer = sq_dict.get("answer", "")

            tokens = await self._verify_one(
                sq_dict, evidence, answer, entity_registry,
                verified_answers, blackboard,
            )
            total_tokens += tokens

        return total_tokens

    async def _verify_one(
        self,
        sq_dict: dict[str, Any],
        evidence: list[dict[str, Any]],
        answer: str,
        entity_registry: dict[str, str],
        verified_answers: dict[int, dict[str, str]],
        blackboard: Blackboard,
    ) -> int:
        """Verify one sub-question's evidence and answer."""
        sq_id = sq_dict["id"]

        evidence_text = ""
        if evidence:
            parts = []
            for ev in evidence:
                parts.append(f"[{ev['source_chunk_id']}] {ev['content']}")
            evidence_text = "\n\n".join(parts)
        else:
            evidence_text = "(No evidence found)"

        # Build verified context for cross-SQ consistency checking
        if verified_answers:
            v_lines = []
            for v_id, v_info in verified_answers.items():
                v_lines.append(f"- SQ-{v_id}: \"{v_info['text']}\" → {v_info['answer']}")
            verified_context = "\n".join(v_lines)
        else:
            verified_context = "(No verified answers yet)"

        prompt = self._prompt_template.format(
            sub_question=sq_dict["text"],
            answer=answer or "(no answer)",
            evidence=evidence_text,
            entity_registry=json.dumps(entity_registry, indent=2) if entity_registry else "{}",
            verified_context=verified_context,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
        except Exception as exc:
            logger.error("Critic LLM error for SQ-%d: %s", sq_id, exc)
            # On error, retry rather than blindly passing
            await blackboard.verify_sub_question(sq_id, verified=False)
            return 0

        raw = response["message"].get("content", "")
        tokens = int(response.get("cost", 0.0) * 1_000_000)
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)

        verdict = self._parse_verdict(raw)
        logger.info("Critic SQ-%d: verdict=%s", sq_id, verdict["verdict"])

        if verdict["verdict"] == "PASS":
            await blackboard.verify_sub_question(sq_id, verified=True)

        elif verdict["verdict"] == "FAIL_INSUFFICIENT":
            await blackboard.post_knowledge_gap(KnowledgeGap(
                sub_question_id=sq_id,
                description=verdict.get("reason", "Insufficient evidence"),
                suggested_query=verdict.get("suggested_query", ""),
            ))
            await blackboard.verify_sub_question(sq_id, verified=False)

        elif verdict["verdict"] == "FAIL_CONTRADICTION":
            if self.enable_backtracking:
                await blackboard.post_contradiction(Contradiction(
                    sub_question_ids=[sq_id],
                    description=verdict.get("reason", "Contradiction detected"),
                ))
                await blackboard.backtrack_sub_question(
                    sq_id, reason=verdict.get("reason", "Contradiction"),
                )
            else:
                await blackboard.verify_sub_question(sq_id, verified=False)

        elif verdict["verdict"] == "FAIL_WRONG_ANSWER":
            if self.enable_backtracking:
                await blackboard.backtrack_sub_question(
                    sq_id, reason=verdict.get("reason", "Wrong answer"),
                )
            else:
                await blackboard.verify_sub_question(sq_id, verified=False)

        else:
            # Unknown verdict → retry rather than blindly passing
            logger.warning("Critic unknown verdict '%s' for SQ-%d, defaulting to NEEDS_RETRY",
                           verdict["verdict"], sq_id)
            await blackboard.verify_sub_question(sq_id, verified=False)

        return tokens

    def _parse_verdict(self, raw: str) -> dict[str, str]:
        """Parse critic verdict from LLM response.

        Expected format:
            VERDICT: PASS|FAIL_INSUFFICIENT|FAIL_CONTRADICTION|FAIL_WRONG_ANSWER
            REASON: <explanation>
            SUGGESTED_QUERY: <optional>
        """
        verdict = "PASS"
        reason = ""
        suggested_query = ""

        # Try JSON first
        try:
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)
            return {
                "verdict": data.get("verdict", "PASS").upper(),
                "reason": data.get("reason", ""),
                "suggested_query": data.get("suggested_query", ""),
            }
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fall back to line-based parsing
        for line in raw.split("\n"):
            line = line.strip()
            upper = line.upper()
            if upper.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
            elif upper.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()
            elif upper.startswith("SUGGESTED_QUERY:") or upper.startswith("SUGGESTED QUERY:"):
                suggested_query = line.split(":", 1)[1].strip()

        # Normalize verdict
        valid_verdicts = {"PASS", "FAIL_INSUFFICIENT", "FAIL_CONTRADICTION", "FAIL_WRONG_ANSWER"}
        if verdict not in valid_verdicts:
            # Try to match partial
            for v in valid_verdicts:
                if v in raw.upper():
                    verdict = v
                    break
            else:
                verdict = "PASS"

        return {"verdict": verdict, "reason": reason, "suggested_query": suggested_query}
