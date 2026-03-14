"""PlannerAgent: decompose -> monitor -> signal synthesis lifecycle.

Phases:
  decompose -> monitor -> (re-decompose | signal_synthesis) -> done

The planner decomposes the question into a sub-question DAG, monitors
worker progress, and signals the SynthesizerAgent when ready.
"""

from __future__ import annotations

import asyncio
import functools
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

_DECOMPOSE_PROMPT_PATH = Path(__file__).parent / "prompts" / "decomposer.txt"


class PlannerAgent(AutonomousAgent):
    """Decompose -> monitor -> signal synthesis agent.

    Phases:
      - decompose: break question into sub-question DAG
      - monitor: watch blackboard, decide re-decompose or signal synthesis
      - signal_synthesis: set allow_synthesis flag for SynthesizerAgent
      - done: terminal
    """

    def __init__(
        self,
        llm_client: LLMClient,
        decompose_prompt_path: str | Path | None = None,
        max_parse_retries: int = 3,
        max_redecompositions: int = 1,
    ):
        super().__init__(agent_id="planner", agent_type="planner")
        self.llm = llm_client
        self.max_parse_retries = max_parse_retries
        self.max_redecompositions = max_redecompositions

        self._phase = "decompose"
        self._decompose_count = 0
        self._question_type: str = "unknown"

        d_path = Path(decompose_prompt_path) if decompose_prompt_path else _DECOMPOSE_PROMPT_PATH
        self._decompose_template = d_path.read_text(encoding="utf-8")

    async def _async_chat(self, **kwargs):
        """Run synchronous llm.chat in executor to avoid blocking event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(self.llm.chat, **kwargs),
        )

    async def decompose_first(self, blackboard: Blackboard) -> int:
        """Run decomposition upfront (before coordinator loop).

        Returns the number of sub-questions produced so the pipeline
        can create the right number of worker agents.
        """
        obs = await self.observe(blackboard)
        await self._decompose(obs, blackboard)
        self._phase = "monitor"
        return len(blackboard.search_plan)

    async def observe(self, blackboard: Blackboard) -> dict[str, Any]:
        return await blackboard.read_for_planner()

    def should_act(self, observation: dict[str, Any]) -> bool:
        if self._phase == "done":
            return False

        if self._phase == "decompose":
            return True

        if self._phase == "monitor":
            sqs = observation.get("sub_questions", [])
            if not sqs:
                return False

            terminal = {SubQuestionStatus.VERIFIED.value, SubQuestionStatus.FAILED.value}
            n_total = len(sqs)
            n_terminal = sum(1 for sq in sqs if sq["status"] in terminal)
            n_verified = sum(1 for sq in sqs if sq["status"] == SubQuestionStatus.VERIFIED.value)
            n_failed = sum(1 for sq in sqs if sq["status"] == SubQuestionStatus.FAILED.value)

            if n_terminal == n_total:
                if n_failed > n_verified and self._decompose_count <= self.max_redecompositions:
                    logger.info(
                        "Planner: re-decomposition (%d/%d failed, %d verified)",
                        n_failed, n_total, n_verified,
                    )
                    self._phase = "decompose"
                    return True
                self._phase = "signal_synthesis"
                return True

            return False

        return False

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        if self._phase == "decompose":
            tokens = await self._decompose(observation, blackboard)
            self._phase = "monitor"
            return tokens

        if self._phase == "signal_synthesis":
            blackboard.allow_synthesis = True
            self._phase = "done"
            logger.info("Planner: signaled allow_synthesis, handing off to SynthesizerAgent")
            return 0

        return 0

    # ── Decompose ─────────────────────────────────────────────────────

    async def _decompose(
        self, observation: dict[str, Any], blackboard: Blackboard,
    ) -> int:
        question = observation["question"]
        is_redecomposition = self._decompose_count > 0
        self._decompose_count += 1

        if is_redecomposition:
            failure_context = observation.get("failure_context", "")
            verified_answers = observation.get("verified_answers", {})
            prompt = self._build_redecompose_prompt(question, failure_context, verified_answers)
        else:
            prompt = self._decompose_template.replace("{question}", question)

        messages = [{"role": "user", "content": prompt}]

        total_tokens = 0
        for attempt in range(self.max_parse_retries):
            response = await self._async_chat(messages=messages, tools=None, temperature=0.0)
            total_tokens += int(response.get("cost", 0.0) * 1_000_000)
            raw = response["message"].get("content", "")

            try:
                sub_questions = self._parse_decomposition(raw)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("Planner parse error (attempt %d/%d): %s", attempt + 1, self.max_parse_retries, exc)
                continue

            if is_redecomposition:
                await blackboard.reset_search_plan(sub_questions, preserve_verified=True)
            else:
                await blackboard.set_search_plan(sub_questions)
                blackboard.expected_answer = getattr(self, "_expected_answer", "")

            logger.info(
                "Planner: %sdecomposed '%s' -> %d sub-questions",
                "re-" if is_redecomposition else "", question[:60], len(sub_questions),
            )
            return int(total_tokens)

        logger.warning("Planner: fallback to single sub-question")
        fallback = [M6SubQuestion(id=0, text=question, dependencies=[], status=SubQuestionStatus.READY)]
        if is_redecomposition:
            await blackboard.reset_search_plan(fallback, preserve_verified=True)
        else:
            await blackboard.set_search_plan(fallback)
        return int(total_tokens)

    def _build_redecompose_prompt(
        self, question: str, failure_context: str, verified_answers: dict[str, str],
    ) -> str:
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

{self._decompose_template.split("Question:")[0]}
Question: {question}"""

    def _parse_decomposition(self, raw: str) -> list[M6SubQuestion]:
        raw_clean = raw.strip()
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```$", "", raw_clean)
        raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL)
        raw_clean = re.sub(r"<think>.*", "", raw_clean, flags=re.DOTALL)
        raw_clean = raw_clean.strip()

        data = json.loads(raw_clean)
        self._question_type = data.get("question_type", "unknown")
        self._expected_answer = data.get("expected_answer", "")
        sub_questions: list[M6SubQuestion] = []
        for sq_data in data.get("sub_questions", []):
            sq = M6SubQuestion(
                id=int(sq_data["index"]),
                text=str(sq_data["text"]),
                dependencies=[int(d) for d in sq_data.get("depends_on", [])],
                known_entities=list(sq_data.get("known_entities", [])),
                unknown_entities=list(sq_data.get("unknown_entities", [])),
                search_hints=list(sq_data.get("search_hints", [])),
                search_queries=list(sq_data.get("search_queries", [])),
            )
            sub_questions.append(sq)

        if not sub_questions:
            raise ValueError("No sub_questions in response")
        self._validate_dag(sub_questions)
        return sub_questions

    @staticmethod
    def _validate_dag(sub_questions: list[M6SubQuestion]) -> None:
        """Validate that dependencies form a DAG (no cycles)."""
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
