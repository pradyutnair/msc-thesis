"""Question decomposer: classify type and emit sub-questions with dependency DAG."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.types import DecompositionPlan, SubQuestion

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "decomposer.txt"


class Decomposer:
    """Decompose a multi-hop question into sub-questions with a dependency DAG.

    Single LLM call with structured JSON output. Falls back to single_hop
    if JSON parsing fails after ``max_retries`` attempts.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_retries: int = 2,
        prompt_path: str | Path | None = None,
        use_nothink: bool = False,
    ):
        self.llm = llm_client
        self.max_retries = max_retries
        self.use_nothink = use_nothink

        path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._prompt_template = path.read_text(encoding="utf-8")

    async def decompose(self, question: str) -> DecompositionPlan:
        """Decompose *question* into a :class:`DecompositionPlan`.

        Returns a single-hop fallback if parsing fails after retries.
        """
        prompt = self._prompt_template.replace("{question}", question)
        if self.use_nothink:
            prompt = f"/nothink\n{prompt}"
        messages = [{"role": "user", "content": prompt}]

        last_raw = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
                raw = response["message"].get("content", "")
                last_raw = raw
                plan = self._parse_response(raw)
                plan.raw_llm_output = raw
                plan.parse_retries = attempt
                logger.info(
                    "Decomposed '%s' → %s (%d sub-Qs, attempt %d)",
                    question[:60], plan.question_type,
                    len(plan.sub_questions), attempt,
                )
                return plan
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "Decomposer parse error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
                continue

        logger.warning("Decomposer fallback to single_hop for: %s", question[:80])
        return self._single_hop_fallback(question, last_raw)

    async def decompose_with_evidence(
        self,
        question: str,
        scout_chunks: list[dict],
        scout_answer: str = "",
        osprey_prompt_path: str | Path | None = None,
    ) -> DecompositionPlan:
        """Evidence-aware decomposition for OSPREY Phase 2.

        Tells the decomposer what Phase 1 already found so it can generate
        better sub-questions targeting only the remaining information gaps.

        Falls back to standard :meth:`decompose` if no osprey_prompt_path is
        given or if parsing fails after all retries.
        """
        if not osprey_prompt_path:
            logger.debug("No OSPREY decomposer prompt; falling back to standard decompose")
            return await self.decompose(question)

        # Build scout evidence block (top-5 chunks, 500 chars each)
        lines = []
        for chunk in scout_chunks[:5]:
            cid = chunk.get("id", "?")
            text = chunk.get("text", "")[:500]
            lines.append(f"[Chunk {cid}]:\n{text}")
        scout_evidence = "\n\n---\n\n".join(lines) if lines else "None collected yet."

        path = Path(osprey_prompt_path)
        prompt_template = path.read_text(encoding="utf-8")

        prompt = (
            prompt_template
            .replace("{question}", question)
            .replace("{scout_evidence}", scout_evidence)
            .replace("{scout_answer}", scout_answer or "No preliminary answer yet.")
        )
        if self.use_nothink:
            prompt = f"/nothink\n{prompt}"
        messages = [{"role": "user", "content": prompt}]

        last_raw = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.llm.chat(messages=messages, tools=None, temperature=0.0)
                raw = response["message"].get("content", "")
                last_raw = raw
                plan = self._parse_response(raw)
                plan.raw_llm_output = raw
                plan.parse_retries = attempt
                logger.info(
                    "OSPREY decomposed '%s' → %s (%d sub-Qs, attempt %d)",
                    question[:60], plan.question_type,
                    len(plan.sub_questions), attempt,
                )
                return plan
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "OSPREY decomposer parse error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
                continue

        logger.warning(
            "OSPREY decomposer fallback to standard for: %s", question[:80]
        )
        return await self.decompose(question)

    def _parse_response(self, raw: str) -> DecompositionPlan:
        """Extract JSON from the LLM response and validate."""
        raw_clean = raw.strip()
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw_clean)
        raw_clean = re.sub(r"\s*```$", "", raw_clean)
        raw_clean = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL)
        raw_clean = raw_clean.strip()

        data = json.loads(raw_clean)

        q_type = data["question_type"]
        if q_type not in ("comparison", "bridge", "single_hop"):
            raise ValueError(f"Unknown question_type: {q_type}")

        sub_questions = []
        for sq in data.get("sub_questions", []):
            sub_questions.append(
                SubQuestion(
                    index=int(sq["index"]),
                    text=str(sq["text"]),
                    search_hints=list(sq.get("search_hints", [])),
                    depends_on=[int(d) for d in sq.get("depends_on", [])],
                    placeholder=sq.get("placeholder"),
                )
            )

        edges = [(int(e[0]), int(e[1])) for e in data.get("dependency_edges", [])]
        self._validate_dag(sub_questions, edges)

        return DecompositionPlan(
            question_type=q_type,
            sub_questions=sub_questions,
            dependency_edges=edges,
        )

    @staticmethod
    def _validate_dag(sub_questions: list[SubQuestion], edges: list[tuple[int, int]]) -> None:
        """Validate edges form a DAG via Kahn's algorithm."""
        indices = {sq.index for sq in sub_questions}
        adj: dict[int, list[int]] = {i: [] for i in indices}
        in_degree: dict[int, int] = {i: 0 for i in indices}

        for src, tgt in edges:
            if src not in indices or tgt not in indices:
                raise ValueError(f"Edge ({src}, {tgt}) references unknown index")
            adj[src].append(tgt)
            in_degree[tgt] += 1

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
            raise ValueError("Dependency edges contain a cycle")

    @staticmethod
    def _single_hop_fallback(question: str, raw_output: str) -> DecompositionPlan:
        return DecompositionPlan(
            question_type="single_hop",
            sub_questions=[SubQuestion(index=0, text=question, search_hints=[], depends_on=[], placeholder=None)],
            dependency_edges=[],
            raw_llm_output=raw_output,
            parse_retries=-1,
        )
