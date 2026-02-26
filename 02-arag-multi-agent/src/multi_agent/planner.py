"""SAGE Planner: analyzes question and generates a structured retrieval plan."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from arag.core.llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "sage_planner.txt"


@dataclass
class RetrievalTask:
    """A single retrieval task in the SAGE plan."""

    id: int
    query: str
    search_method: str = "keyword"
    goal: str = ""
    depends_on: list[int] = field(default_factory=list)


@dataclass
class SagePlan:
    """Full output of the Planner for one question."""

    question_type: str  # "single", "bridge", "comparison"
    tasks: list[RetrievalTask] = field(default_factory=list)
    expected_answer_type: str = "entity"
    raw_llm_output: str = ""
    parse_retries: int = 0


class Planner:
    """Analyze a question and produce a structured retrieval plan.

    Single LLM call with structured JSON output. Falls back to a single
    keyword search task if JSON parsing fails after retries.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path | None = None,
        max_retries: int = 2,
    ):
        self.llm = llm_client
        self.max_retries = max_retries

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    async def plan(self, question: str) -> SagePlan:
        """Generate a retrieval plan for *question*."""
        prompt = self._prompt_template.replace("{question}", question)
        messages = [{"role": "user", "content": prompt}]

        last_raw = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.chat(
                    messages=messages, tools=None, temperature=0.0,
                )
                raw = response["message"].get("content", "")
                last_raw = raw
                plan = self._parse_response(raw)
                plan.raw_llm_output = raw
                plan.parse_retries = attempt
                logger.info(
                    "Planned '%s' -> %s (%d tasks, attempt %d)",
                    question[:60],
                    plan.question_type,
                    len(plan.tasks),
                    attempt,
                )
                return plan
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Planner parse error (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                continue

        logger.warning("Planner fallback to single task for: %s", question[:80])
        return self._single_task_fallback(question, last_raw)

    def _parse_response(self, raw: str) -> SagePlan:
        raw_clean = raw.strip()
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```$", "", raw_clean)
        raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL)
        raw_clean = raw_clean.strip()

        data = json.loads(raw_clean)

        q_type = data["question_type"]
        if q_type not in ("single", "bridge", "comparison"):
            raise ValueError(f"Unknown question_type: {q_type}")

        tasks = []
        for t in data.get("tasks", []):
            tasks.append(
                RetrievalTask(
                    id=int(t["id"]),
                    query=str(t["query"]),
                    search_method=str(t.get("search_method", "keyword")),
                    goal=str(t.get("goal", "")),
                    depends_on=[int(d) for d in t.get("depends_on", [])],
                )
            )

        if not tasks:
            raise ValueError("No tasks in plan")

        return SagePlan(
            question_type=q_type,
            tasks=tasks,
            expected_answer_type=data.get("expected_answer_type", "entity"),
        )

    @staticmethod
    def _single_task_fallback(question: str, raw_output: str) -> SagePlan:
        return SagePlan(
            question_type="single",
            tasks=[
                RetrievalTask(
                    id=0,
                    query=question,
                    search_method="keyword",
                    goal=question,
                )
            ],
            expected_answer_type="entity",
            raw_llm_output=raw_output,
            parse_retries=-1,
        )
