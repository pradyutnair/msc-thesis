"""M7 Agents: CORAL architecture — three agents only.

- DecomposerAgent: breaks question into typed sub-questions
- WorkerAgent: ONE unified worker (evidence pool + new retrieval + extraction)
- SynthesizerAgent: combines sub-answers into final answer
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from multi_agent.m7.blackboard import (
    Blackboard,
    DecompositionPlan,
    SubQuestion,
)
from multi_agent.m7.llm_client import VllmChatClient, strip_reasoning

logger = logging.getLogger(__name__)

# -- Prompt loading ----------------------------------------------------

_PROMPT_CACHE: dict[str, str] = {}


def _load_prompt(path: str | Path) -> str:
    path = str(path)
    if path not in _PROMPT_CACHE:
        with open(path, "r", encoding="utf-8") as f:
            _PROMPT_CACHE[path] = f.read()
    return _PROMPT_CACHE[path]


# =====================================================================
# DecomposerAgent
# =====================================================================


class DecomposerAgent:
    """Decomposes a question into typed sub-questions with search queries.

    Uses M6's decomposer_v29.txt prompt. Handles Qwen3 <think> tags and
    both JSON array and object response formats.
    """

    def __init__(self, llm: VllmChatClient, prompt_path: str | Path):
        self.llm = llm
        self.prompt_path = str(prompt_path)

    def decompose(self, question: str) -> DecompositionPlan:
        """One LLM call -> DecompositionPlan."""
        prompt_template = _load_prompt(self.prompt_path)
        prompt = prompt_template.replace("{question}", question)
        response = self.llm.generate(
            prompt, max_new_tokens=1024, temperature=0.1, enable_thinking=True,
        )
        # Strip Qwen3 <think> tags before JSON parsing
        response = strip_reasoning(response)

        try:
            cleaned = re.sub(
                r"```json\s*|\s*```", "", response.strip(), flags=re.DOTALL,
            )

            # Try object format first: {"question_type": ..., "sub_questions": ...}
            obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if obj_match:
                data = json.loads(obj_match.group())
                if "sub_questions" in data:
                    return self._parse_plan(data, question)

            # Try array format: [{"index": 0, ...}, ...]
            arr_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
            if arr_match:
                arr = json.loads(arr_match.group())
                if isinstance(arr, list) and arr:
                    data = {
                        "question_type": "bridge",
                        "expected_answer": "unknown",
                        "sub_questions": arr,
                    }
                    return self._parse_plan(data, question)

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Decomposer JSON parse failed: %s", exc)

        # Fallback: single sub-question = original question
        return DecompositionPlan(
            question_type="single_hop",
            expected_answer="unknown",
            sub_questions=[
                SubQuestion(
                    index=0,
                    text=question,
                    search_queries=[question],
                )
            ],
        )

    def _parse_plan(
        self, data: dict[str, Any], question: str,
    ) -> DecompositionPlan:
        """Parse decomposer JSON output into a DecompositionPlan."""
        question_type = data.get("question_type", "single_hop")
        expected_answer = data.get("expected_answer", "unknown")
        dep_edges = data.get("dependency_edges", [])

        sqs: list[SubQuestion] = []
        for sq_data in data.get("sub_questions", []):
            sq = SubQuestion(
                index=int(sq_data.get("index", len(sqs))),
                text=str(sq_data.get("text", question)),
                search_queries=sq_data.get("search_queries", []),
                search_hints=sq_data.get("search_hints", []),
                known_entities=sq_data.get("known_entities", []),
                unknown_entities=sq_data.get("unknown_entities", []),
                depends_on=[int(d) for d in sq_data.get("depends_on", [])],
            )
            sqs.append(sq)

        if not sqs:
            sqs = [SubQuestion(index=0, text=question, search_queries=[question])]

        return DecompositionPlan(
            question_type=question_type,
            expected_answer=expected_answer,
            sub_questions=sqs,
            dependency_edges=dep_edges,
        )


# =====================================================================
# WorkerAgent
# =====================================================================


class WorkerAgent:
    """Unified CORAL worker: evidence pool search + new retrieval + extraction.

    ONE worker type (no separate analyzer/rewriter/verifier). Makes ONE
    LLM call per sub-question. Handles:
    - Evidence pool documents (from predecessor workers)
    - Newly retrieved documents (from corpus)
    - Predecessor context for collaborative verification
    """

    def __init__(self, llm: VllmChatClient, prompt_path: str | Path):
        self.llm = llm
        self.prompt_path = str(prompt_path)

    def extract(
        self,
        sub_question: str,
        answer_type: str,
        evidence_pool_docs: list[dict[str, Any]],
        new_docs: list[dict[str, Any]],
        predecessor_context: str = "",
        verification_instruction: str = "",
    ) -> dict[str, Any]:
        """One LLM call -> {answer, predecessor_correction, source}."""
        prompt_template = _load_prompt(self.prompt_path)

        prompt = prompt_template.replace("{sub_question}", sub_question)
        prompt = prompt.replace(
            "{answer_type}", answer_type or "a short factual phrase",
        )
        prompt = prompt.replace(
            "{evidence_pool_docs}",
            self._format_pool_docs(evidence_pool_docs),
        )
        prompt = prompt.replace(
            "{new_docs}",
            self._format_new_docs(new_docs),
        )
        prompt = prompt.replace(
            "{predecessor_context}",
            predecessor_context or "No previous worker findings.",
        )
        prompt = prompt.replace(
            "{verification_instruction}", verification_instruction,
        )

        response = self.llm.generate(
            prompt, max_new_tokens=512, temperature=0.1,
        )
        response = strip_reasoning(response)

        # Find the last JSON object in the response (skip any in thinking text)
        try:
            # Try to find all JSON-like blocks and parse the last valid one
            result = None
            for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response):
                try:
                    candidate = json.loads(m.group())
                    if "answer" in candidate:
                        result = candidate
                except (json.JSONDecodeError, ValueError):
                    continue
            if result is None:
                # Fallback: greedy match
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    result = json.loads(match.group())
            if result:
                result.setdefault("answer", "")
                result.setdefault("predecessor_correction", "")
                result.setdefault("source", "unknown")
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: treat raw response as the answer
        answer = response.strip().split("\n")[0].strip()
        return {
            "answer": answer[:200] if answer else "unknown",
            "predecessor_correction": "",
            "source": "fallback",
        }

    @staticmethod
    def _format_pool_docs(docs: list[dict[str, Any]]) -> str:
        """Format evidence pool documents with worker source annotation."""
        if not docs:
            return "No evidence from other workers yet."
        formatted = []
        for i, doc in enumerate(docs[:10], 1):
            title = doc.get("title") or f"Document {i}"
            content = doc.get("content", "")
            doc_id = doc.get("doc_id", i)
            source_worker = doc.get("source_worker", "?")
            formatted.append(
                f"[From Worker {source_worker}'s retrieval | "
                f"ID: {doc_id}]\n"
                f"Title: {title}\n{content}"
            )
        return "\n\n".join(formatted)

    @staticmethod
    def _format_new_docs(docs: list[dict[str, Any]]) -> str:
        """Format newly retrieved corpus documents."""
        if not docs:
            return "No new documents retrieved."
        formatted = []
        for i, doc in enumerate(docs[:15], 1):
            title = doc.get("title") or f"Document {i}"
            content = doc.get("content", "")
            doc_id = doc.get("doc_id", i)
            neighbor_tag = " [neighbor]" if doc.get("is_neighbor") else ""
            formatted.append(
                f"[Doc{i} - ID: {doc_id}{neighbor_tag}]\n"
                f"Title: {title}\n{content}"
            )
        return "\n\n".join(formatted)


# =====================================================================
# SynthesizerAgent
# =====================================================================


class SynthesizerAgent:
    """Combines sub-answers into a final answer.

    Uses M6's synthesizer_v29.txt prompt. Builds evidence blocks and
    entity registry from the blackboard state.
    """

    def __init__(self, llm: VllmChatClient, prompt_path: str | Path):
        self.llm = llm
        self.prompt_path = str(prompt_path)

    def synthesize(self, blackboard: Blackboard) -> str:
        """One LLM call -> final answer string."""
        ctx = blackboard.get_synthesis_context()
        prompt_template = _load_prompt(self.prompt_path)

        # Build evidence blocks
        evidence_blocks = []
        for sq_info in ctx.get("sub_questions", []):
            idx = sq_info["index"]
            text = sq_info["text"]
            answer = sq_info.get("answer") or "No answer"
            status = sq_info["status"]
            evidence_blocks.append(
                f"Sub-question {idx}: \"{text}\"\n"
                f"  Status: {status}\n"
                f"  Answer: {answer}"
            )

        entity_registry = "\n".join(
            f"  answer_{idx} = {ans}"
            for idx, ans in ctx.get("answers", {}).items()
        ) or "  (none)"

        prompt = prompt_template.replace("{question}", ctx["question"])
        prompt = prompt.replace(
            "{expected_answer}", ctx.get("expected_answer", "unknown"),
        )
        prompt = prompt.replace(
            "{evidence_blocks}", "\n\n".join(evidence_blocks),
        )
        prompt = prompt.replace("{entity_registry}", entity_registry)

        response = self.llm.generate(
            prompt, max_new_tokens=256, temperature=0.1,
        )
        response = strip_reasoning(response)

        # Extract just the answer (strip any preamble)
        answer = response.strip()
        for line in answer.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove common prefixes
            for prefix in (
                "Based on",
                "The answer is",
                "Final answer:",
                "FINAL ANSWER:",
                "Answer:",
            ):
                if line.lower().startswith(prefix.lower()):
                    line = line[len(prefix):].strip().strip(":").strip()
                    break
            if line:
                return line

        return answer.split("\n")[0].strip() if answer else ""
