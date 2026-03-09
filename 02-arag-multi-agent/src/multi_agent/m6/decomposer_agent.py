"""DecomposerAgent: decomposes a multi-hop question into a DAG of sub-questions.

Acts on tick 1 for initial decomposition. Can re-decompose when majority of
sub-questions have FAILED, preserving already-verified answers. Falls back
to a single sub-question on parse failure.
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
from multi_agent.m6.types import M6SubQuestion, SubQuestionStatus

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "decomposer.txt"


class DecomposerAgent(AutonomousAgent):
    """Decomposes the original question into a dependency-aware search plan.

    Supports re-decomposition when majority of sub-questions fail.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
        max_parse_retries: int = 3,
        max_redecompositions: int = 1,
    ):
        super().__init__(agent_id="decomposer", agent_type="decomposer")
        self.llm = llm_client
        self.max_parse_retries = max_parse_retries
        self.max_redecompositions = max_redecompositions
        self._decompose_count = 0

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_decomposer()

    def should_act(self, observation: dict[str, Any]) -> bool:
        # Initial decomposition
        if self._decompose_count == 0:
            return True

        # Re-decomposition: check if majority of SQs failed
        if self._decompose_count > self.max_redecompositions:
            return False

        sqs = observation.get("sub_questions", [])
        if not sqs:
            return False

        n_failed = sum(1 for sq in sqs if sq["status"] == SubQuestionStatus.FAILED.value)
        n_verified = sum(1 for sq in sqs if sq["status"] == SubQuestionStatus.VERIFIED.value)
        n_total = len(sqs)

        # Re-decompose if: majority failed AND not all verified AND all are terminal
        terminal = {SubQuestionStatus.VERIFIED.value, SubQuestionStatus.FAILED.value}
        all_terminal = all(sq["status"] in terminal for sq in sqs)
        if all_terminal and n_failed > n_verified and n_failed > 0:
            logger.info(
                "Decomposer: triggering re-decomposition (%d/%d failed, %d verified)",
                n_failed, n_total, n_verified,
            )
            return True

        return False

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        question = observation["question"]
        is_redecomposition = self._decompose_count > 0
        self._decompose_count += 1

        # Build prompt — include failure context for re-decomposition
        if is_redecomposition:
            failure_context = observation.get("failure_context", "")
            verified_answers = observation.get("verified_answers", {})
            prompt = self._build_redecompose_prompt(question, failure_context, verified_answers)
        else:
            prompt = self._prompt_template.replace("{question}", question)

        messages = [{"role": "user", "content": prompt}]

        total_tokens = 0
        for attempt in range(self.max_parse_retries):
            try:
                response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
                total_tokens += response.get("cost", 0.0) * 1_000_000
                raw = response["message"].get("content", "")
                sub_questions = self._parse_response(raw)

                if is_redecomposition:
                    await blackboard.reset_search_plan(
                        sub_questions,
                        preserve_verified=True,
                    )
                    logger.info(
                        "Re-decomposed '%s' → %d sub-questions (preserving verified)",
                        question[:60], len(sub_questions),
                    )
                else:
                    await blackboard.set_search_plan(sub_questions)
                    logger.info(
                        "Decomposed '%s' → %d sub-questions",
                        question[:60], len(sub_questions),
                    )
                return int(total_tokens)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Decomposer parse error (attempt %d/%d): %s",
                    attempt + 1, self.max_parse_retries, exc,
                )
                continue

        # Fallback: single sub-question
        logger.warning("Decomposer fallback to single sub-question")
        fallback = [M6SubQuestion(
            id=0,
            text=question,
            dependencies=[],
            status=SubQuestionStatus.READY,
        )]
        if is_redecomposition:
            await blackboard.reset_search_plan(fallback, preserve_verified=True)
        else:
            await blackboard.set_search_plan(fallback)
        return int(total_tokens)

    def _build_redecompose_prompt(
        self,
        question: str,
        failure_context: str,
        verified_answers: dict[str, str],
    ) -> str:
        """Build prompt for re-decomposition with failure context."""
        verified_str = ""
        if verified_answers:
            parts = [f"- {k}: {v}" for k, v in verified_answers.items()]
            verified_str = "Already verified answers (DO NOT re-ask these):\n" + "\n".join(parts)

        return f"""The previous decomposition of this question mostly failed. Re-decompose with a different strategy.

Question: {question}

{verified_str}

Previous failure context:
{failure_context}

Try a DIFFERENT decomposition strategy:
- If bridge decomposition failed, try a more direct approach
- If sub-questions were too specific, make them broader
- Use simpler, more searchable sub-questions
- If an entity wasn't found, try searching for it differently

{self._prompt_template.split("Question:")[0]}
Question: {question}"""

    def _parse_response(self, raw: str) -> list[M6SubQuestion]:
        """Extract JSON from LLM response and build sub-question list."""
        raw_clean = raw.strip()
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```$", "", raw_clean)
        raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL)
        raw_clean = re.sub(r"<think>.*", "", raw_clean, flags=re.DOTALL)
        raw_clean = raw_clean.strip()

        data = json.loads(raw_clean)

        sub_questions: list[M6SubQuestion] = []
        for sq_data in data.get("sub_questions", []):
            sq = M6SubQuestion(
                id=int(sq_data["index"]),
                text=str(sq_data["text"]),
                dependencies=[int(d) for d in sq_data.get("depends_on", [])],
                known_entities=list(sq_data.get("known_entities", [])),
                unknown_entities=list(sq_data.get("unknown_entities", [])),
                search_hints=list(sq_data.get("search_hints", [])),
            )
            sub_questions.append(sq)

        if not sub_questions:
            raise ValueError("No sub_questions in response")

        # Validate DAG
        self._validate_dag(sub_questions)

        return sub_questions

    @staticmethod
    def _validate_dag(sub_questions: list[M6SubQuestion]) -> None:
        """Kahn's algorithm to verify DAG (no cycles)."""
        indices = {sq.id for sq in sub_questions}
        adj: dict[int, list[int]] = {i: [] for i in indices}
        in_degree: dict[int, int] = {i: 0 for i in indices}

        for sq in sub_questions:
            for dep_id in sq.dependencies:
                if dep_id not in indices:
                    raise ValueError(f"Dependency {dep_id} not in sub-question indices")
                adj[dep_id].append(sq.id)
                in_degree[sq.id] += 1

        queue = [i for i in indices if in_degree[i] == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for nb in adj[node]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        if visited != len(indices):
            raise ValueError("Dependencies contain a cycle")
