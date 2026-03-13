"""PlannerAgent: unified decompose → monitor → synthesize lifecycle.

A single stateful agent that replaces the separate DecomposerAgent and
SynthesizerAgent. Transitions through phases:

  decompose → monitor → (synthesize | re-decompose) → done

Each phase decision is made autonomously based on blackboard state,
making the planner genuinely agentic rather than a one-shot call.
"""

from __future__ import annotations

import json
import asyncio
import functools
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
_SYNTHESIZE_PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesizer.txt"
_CONSISTENCY_PROMPT_PATH = Path(__file__).parent / "prompts" / "synthesizer_consistency.txt"


# ── Answer normalization (shared with synthesizer_agent.py) ───────────

def _infer_expected_answer_type(question: str) -> str:
    q = (question or "").strip().lower()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had|can|could|would|should)\b", q):
        return "yes_no"
    if q.startswith("when ") or "what year" in q or "what date" in q:
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith("who "):
        return "person"
    if "how many" in q or "how much" in q or q.startswith("what age "):
        return "number"
    return "entity"


_DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}


def _normalize_answer(answer: str, question: str) -> str:
    text = (answer or "").strip()
    text = re.sub(r"^\s*(final answer\s*:|answer\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the answer is|answer is)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")

    expected_type = _infer_expected_answer_type(question)
    if expected_type == "yes_no":
        lowered = text.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"

    # Convert digit-only answers to words for "how many" questions
    if expected_type == "number" and text in _DIGIT_TO_WORD:
        text = _DIGIT_TO_WORD[text]

    for pattern in [
        r"^(.*?)\s+was\s+born\s+first\.?$",
        r"^(.*?)\s+was\s+(?:produced|released|published|created|formed|founded)\s+first\.?$",
        r"^(.*?)\s+(?:died|passed away)\s+(?:first|earlier|before)\.?$",
        r"^(.*?)\s+is\s+(?:the\s+)?(?:older|younger|taller|shorter|bigger|smaller)\.?$",
        r"^(.*?)\s+is\s+the\s+answer\.?$",
        r"^The\s+answer\s+is\s+(.+?)\.?$",
    ]:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            break

    # Strip parenthetical annotations: "Egremont (market town)" → "Egremont"
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()
    text = re.sub(r"\s*[\.,;:!?]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Strip trailing noise words from comparison answers
    text = re.sub(
        r"\s+(?:theme|genre|style|type|form|category)$",
        "", text, flags=re.IGNORECASE,
    )

    if len(text) > 60:
        m = re.match(r"^(.{3,50}?)[.,;]", text)
        if m:
            text = m.group(1).strip()
    return text


def _is_refusal(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if lowered in ("none", "n/a", "unknown", "null", ""):
        return True
    patterns = [
        "cannot be determined", "insufficient information", "not mentioned",
        "no evidence", "unable to determine", "not enough information",
        "unknown", "cannot determine", "no information",
        "not found in the provided", "the question is invalid",
        "the question is asking", "not applicable",
        "no answer", "i don't know", "i cannot",
    ]
    return any(p in lowered for p in patterns)


class PlannerAgent(AutonomousAgent):
    """Unified decompose → monitor → synthesize agent.

    Phases:
      - decompose: break question into sub-question DAG
      - monitor: watch blackboard, decide re-decompose or synthesize
      - synthesize: aggregate evidence into final answer
      - done: terminal
    """

    def __init__(
        self,
        llm_client: LLMClient,
        decompose_prompt_path: str | Path | None = None,
        synthesize_prompt_path: str | Path | None = None,
        consistency_prompt_path: str | Path | None = None,
        max_parse_retries: int = 3,
        max_redecompositions: int = 1,
        enable_consistency_check: bool = True,
    ):
        super().__init__(agent_id="planner", agent_type="planner")
        self.llm = llm_client
        self.max_parse_retries = max_parse_retries
        self.max_redecompositions = max_redecompositions
        self.enable_consistency_check = enable_consistency_check

        self._phase = "decompose"
        self._decompose_count = 0
        self._question_type: str = "unknown"  # Set during decomposition

        d_path = Path(decompose_prompt_path) if decompose_prompt_path else _DECOMPOSE_PROMPT_PATH
        self._decompose_template = d_path.read_text(encoding="utf-8")

        s_path = Path(synthesize_prompt_path) if synthesize_prompt_path else _SYNTHESIZE_PROMPT_PATH
        self._synthesize_template = s_path.read_text(encoding="utf-8")

        c_path = Path(consistency_prompt_path) if consistency_prompt_path else _CONSISTENCY_PROMPT_PATH
        self._consistency_template = c_path.read_text(encoding="utf-8") if c_path.exists() else None

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
        if self._phase == "decompose":
            return await blackboard.read_for_decomposer()
        elif self._phase == "synthesize":
            return await blackboard.read_for_synthesizer()
        else:
            return await blackboard.read_for_decomposer()

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
                # All sub-questions complete and no re-decomposition needed.
                # Transition to synthesize so we can set allow_synthesis flag,
                # then let SynthesizerAgent handle the actual synthesis.
                self._phase = "signal_synthesis"
                return True

            # Wait for ALL sub-questions to complete before synthesizing.
            # Partial synthesis was removed — it caused correctness issues
            # (e.g., "Are both X and Y Z?" needs all answers, not just one).
            return False

        if self._phase == "synthesize":
            return True

        return False

    async def act(self, observation: dict[str, Any], blackboard: Blackboard) -> int:
        if self._phase == "decompose":
            tokens = await self._decompose(observation, blackboard)
            self._phase = "monitor"
            return tokens

        if self._phase == "signal_synthesis":
            # Set allow_synthesis flag so the SynthesizerAgent can fire.
            # This prevents the race where re-decomposition and synthesis
            # both trigger on the same "all terminal" observation.
            blackboard.allow_synthesis = True
            self._phase = "done"
            logger.info("Planner: signaled allow_synthesis, handing off to SynthesizerAgent")
            return 0

        if self._phase == "synthesize":
            # Re-observe: should_act may have transitioned from monitor,
            # but the observation was fetched with read_for_decomposer
            # which lacks verified_evidence needed for synthesis.
            observation = await blackboard.read_for_synthesizer()
            tokens = await self._synthesize(observation, blackboard)
            self._phase = "done"
            return tokens

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
                "Planner: %sdecomposed '%s' → %d sub-questions",
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

    # ── Synthesize ────────────────────────────────────────────────────

    @staticmethod
    def _find_terminal_sq(sub_questions: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Find the terminal (leaf) sub-question in a bridge chain.

        The terminal SQ is the one that no other SQ depends on —
        its answer is the final answer for the original question.
        """
        all_ids = {sq["id"] for sq in sub_questions}
        depended_on: set[int] = set()
        for sq in sub_questions:
            for dep_id in sq.get("dependencies", []):
                depended_on.add(dep_id)
        leaf_ids = all_ids - depended_on
        # Pick the highest-ID leaf (usually the last hop)
        if leaf_ids:
            leaf_id = max(leaf_ids)
            for sq in sub_questions:
                if sq["id"] == leaf_id:
                    return sq
        return None

    async def _synthesize(
        self, observation: dict[str, Any], blackboard: Blackboard,
    ) -> int:
        question = observation["question"]
        sub_questions = observation["sub_questions"]
        verified_evidence = observation["verified_evidence"]
        entity_registry = observation["entity_registry"]

        evidence_blocks = self._build_evidence_blocks(sub_questions, verified_evidence, entity_registry)
        entity_str = "\n".join(f"- {k} = {v}" for k, v in entity_registry.items()) if entity_registry else "None"
        expected_answer = getattr(blackboard, "expected_answer", "") or ""

        # Simple reasoning prompt — no template, no overrides. Planner reasons with thinking ON.
        prompt_parts = [
            f"Question: {question}",
            "",
            f"Sub-question answers:\n{evidence_blocks}",
            "",
            f"Entity registry (authoritative worker answers):\n{entity_str}",
        ]
        if expected_answer:
            prompt_parts.append(f"\nThe answer should be: {expected_answer}")
        prompt_parts.append("""
The entity registry contains verified answers from workers who searched the knowledge base.
Trust these answers — workers already verified them against retrieved evidence.
If evidence sections show "(no evidence)", the worker found the answer in initial context and confirmed it.

Using the sub-question answers and entity registry, answer the original question.
Reply with ONLY the final answer — a short entity, name, number, date, or yes/no. Nothing else.""")

        messages = [{"role": "user", "content": "\n".join(prompt_parts)}]

        total_tokens = 0
        try:
            response = await self._async_chat(messages=messages, tools=None, temperature=0.0)
            raw = response["message"].get("content", "")
            total_tokens += int(response.get("cost", 0.0) * 1_000_000)
        except Exception as exc:
            logger.error("Planner synthesis LLM error: %s", exc)
            answer = await blackboard.salvage_answer()
            answer = _normalize_answer(answer, question)
            await blackboard.set_final_answer(answer)
            await blackboard.terminate("SYNTHESIZED_FALLBACK")
            return 0

        answer = self._extract_answer(raw)
        if not answer or _is_refusal(answer):
            answer = await blackboard.salvage_answer()
            logger.info("Planner: synthesis empty/refusal, salvaged: '%s'", answer[:80])

        answer = _normalize_answer(answer, question)

        await blackboard.set_final_answer(answer)
        await blackboard.terminate("SYNTHESIZED")
        logger.info("Planner: final answer '%s'", answer[:80])
        return total_tokens

    def _correct_comparison_answer(
        self,
        answer: str,
        question: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
    ) -> str:
        """Programmatic fallback for comparison questions.

        Handles 3 patterns that Qwen3-8B consistently gets wrong:
        1. "Who was born first / formed earlier" → smaller year = first
        2. "Are both X and Y Z?" → if both sub-answers confirm Z, answer yes
        3. "Which has more X" → larger number wins
        """
        q = question.lower().strip()

        # Pattern 1: "who/which was X first/earlier" — compare years
        # Only use programmatic comparison when we have exactly 2 entities
        # with parseable years — otherwise trust the LLM synthesizer
        is_first = any(p in q for p in (
            "born first", "formed first", "formed earlier", "founded first",
            "released first", "published first", "created first", "came first",
            "which was founded", "in between",
        ))
        if is_first:
            q_entities = self._extract_entities_from_question(question)
            if len(q_entities) == 2:
                # Verify both sub-answers have parseable years before overriding
                year_count = 0
                for sq in sub_questions:
                    val = entity_registry.get(f"answer_{sq['id']}", "")
                    if self._extract_year(val) is not None:
                        year_count += 1
                if year_count >= 2:
                    return self._compare_by_date(
                        question, sub_questions, entity_registry, pick="smallest",
                    )
            # Fall through to LLM synthesizer answer

        # Pattern 2: "Are both X and Y Z?" — check if both sub-answers confirm
        is_both = re.match(r"^(are|were|do|does|did|is)\s+.+\s+both\b", q)
        if is_both:
            return self._check_both(
                answer, sub_questions, entity_registry, question,
            )

        # Pattern 3: "Which has more" — compare numbers
        is_more = any(p in q for p in (
            "has more", "have more", "which is longer", "which is larger",
            "which is bigger", "more acts", "more episodes", "more seasons",
        ))
        if is_more:
            return self._compare_by_date(
                question, sub_questions, entity_registry, pick="largest",
            )

        return answer

    @staticmethod
    def _extract_year(text: str) -> int | None:
        """Extract the first 4-digit year or last integer from text."""
        m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", text)
        if m:
            return int(m.group(1))
        m = re.search(r"\b(\d+)\b", text)
        return int(m.group(1)) if m else None

    def _compare_by_date(
        self,
        question: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
        pick: str,
    ) -> str:
        """Pick the entity whose sub-answer has the smallest/largest numeric value."""
        # Extract entity names from the original question
        q_entities = self._extract_entities_from_question(question)

        pairs: list[tuple[str, int]] = []
        for i, sq in enumerate(sub_questions):
            sq_id = sq["id"]
            val = entity_registry.get(f"answer_{sq_id}", "")
            year = self._extract_year(val)
            if year is None:
                continue
            # Map SQ to entity: prefer entity from question, fall back to known_entities
            entity = q_entities[i] if i < len(q_entities) else ""
            if not entity:
                for known in sq.get("known_entities", []):
                    entity = known
                    break
            if not entity:
                entity = sq.get("text", "")
            pairs.append((entity, year))

        if len(pairs) < 2:
            return ""

        if pick == "smallest":
            pairs.sort(key=lambda x: x[1])
        else:
            pairs.sort(key=lambda x: -x[1])

        return pairs[0][0]

    @staticmethod
    def _extract_entities_from_question(question: str) -> list[str]:
        """Extract entity names from comparison questions.

        Handles: "Who was born first, X or Y?", "Was X or Y born first?",
                 "Between X and Y, ...", "born first out of X and Y"
        """
        q = question.strip().rstrip("?").strip()
        # Pattern: "..., X or Y"
        m = re.search(r",\s*(.+?)\s+or\s+(.+?)$", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "Was/Is X or Y ..."  (no comma)
        m = re.match(r"(?:was|is|were|are|did|has)\s+(.+?)\s+or\s+(.+?)\s+(?:born|died|formed|founded|established|released|created|published)", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "... out of X and Y"
        m = re.search(r"out of\s+(.+?)\s+and\s+(.+?)$", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "Between X and Y, ..."
        m = re.match(r"(?:between|in between)\s+(.+?)\s+and\s+(.+?),", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "X and Y" for "Are X and Y both Z?"
        m = re.match(r"(?:are|were|is|do|does)\s+(.+?)\s+and\s+(.+?)\s+both\b", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        return []

    @staticmethod
    def _check_both(
        current_answer: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
        question: str,
    ) -> str:
        """For 'are both X and Y Z?' questions, check if both sub-answers confirm Z."""
        _refusals = {"unknown", "no evidence", "not mentioned", "not found", "n/a", "none"}
        non_refusal_count = 0
        denial_count = 0

        for sq in sub_questions:
            sq_id = sq["id"]
            val = entity_registry.get(f"answer_{sq_id}", "").strip().lower()
            if not val or val in _refusals or any(r in val for r in _refusals):
                continue
            # Only count explicit "no" denial, not words starting with "no"
            if val in ("no", "no."):
                denial_count += 1
            elif val.startswith("no,") or val.startswith("no "):
                denial_count += 1
            else:
                non_refusal_count += 1

        if non_refusal_count >= 2 and denial_count == 0:
            return "yes"
        if denial_count > 0:
            return "no"
        return current_answer

    def _correct_bridge_answer(
        self,
        answer: str,
        question: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
    ) -> str:
        """If the synthesizer returned an intermediate entity, use the leaf's answer.

        For bridge questions like "What county is X's birthplace in?":
          SQ-0: "Where was X born?" → "Springfield"
          SQ-1: "What county is Springfield in?" → "Sangamon County"
        If the synthesizer returns "Springfield" (intermediate), correct to "Sangamon County".
        """
        terminal_sq = self._find_terminal_sq(sub_questions)
        if terminal_sq is None:
            return answer

        terminal_id = terminal_sq["id"]
        terminal_answer = entity_registry.get(
            f"answer_{terminal_id}", terminal_sq.get("answer", ""),
        )

        # If the synthesizer already returned the terminal answer, no correction needed
        if not terminal_answer or _is_refusal(terminal_answer):
            return answer
        if answer.lower().strip() == _normalize_answer(terminal_answer, question).lower().strip():
            return answer

        # Check if the synthesizer returned a non-terminal (intermediate) entity
        intermediate_values = set()
        for sq in sub_questions:
            sq_id = sq["id"]
            if sq_id == terminal_id:
                continue
            val = entity_registry.get(f"answer_{sq_id}", sq.get("answer", ""))
            if val:
                intermediate_values.add(val.lower().strip())

        answer_lower = answer.lower().strip()
        if answer_lower in intermediate_values:
            corrected = _normalize_answer(terminal_answer, question)
            logger.info(
                "Planner: bridge correction '%s' → '%s' (was intermediate entity)",
                answer[:40], corrected[:40],
            )
            return corrected

        return answer

    @staticmethod
    def _build_evidence_blocks(
        sub_questions: list[dict[str, Any]],
        verified_evidence: list[dict[str, Any]],
        entity_registry: dict[str, str],
    ) -> str:
        ev_by_sq: dict[int, list[dict]] = {}
        for ev in verified_evidence:
            ev_by_sq.setdefault(ev["sub_question_id"], []).append(ev)

        blocks: list[str] = []
        for sq in sub_questions:
            sq_id = sq["id"]
            answer = sq.get("answer", "(no answer)")
            resolved_answer = entity_registry.get(f"answer_{sq_id}", answer)
            ev_texts = [f"  [{ev['source_chunk_id']}] {ev['content'][:500]}" for ev in ev_by_sq.get(sq_id, [])]
            evidence_str = "\n".join(ev_texts) if ev_texts else "  (no evidence)"
            blocks.append(
                f"### Sub-Question {sq_id}: {sq['text']}\n"
                f"**Status**: {sq['status']}\n"
                f"**Answer**: {resolved_answer}\n"
                f"**Evidence**:\n{evidence_str}\n"
            )
        return "\n".join(blocks)

    @staticmethod
    def _extract_answer(raw: str) -> str:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        match = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        return lines[-1] if lines else raw.strip()

    async def _consistency_check(
        self, question: str, answer: str, evidence_blocks: str,
    ) -> tuple[str, int]:
        prompt = self._consistency_template.format(
            question=question, answer=answer, evidence_blocks=evidence_blocks,
        )
        try:
            response = await self._async_chat(
                messages=[{"role": "user", "content": prompt}], tools=None, temperature=0.0,
            )
            raw = response["message"].get("content", "")
            tokens = int(response.get("cost", 0.0) * 1_000_000)
        except Exception as exc:
            logger.error("Planner consistency check error: %s", exc)
            return answer, 0

        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        revised = self._extract_answer(raw) if "FINAL ANSWER" in raw.upper() else raw.strip()
        if len(revised) > 100:
            lines = [l.strip() for l in revised.split("\n") if l.strip()]
            revised = lines[-1] if lines else revised
        revised = _normalize_answer(revised, question)
        if revised and revised.lower() != answer.lower():
            logger.info("Planner consistency revised: '%s' → '%s'", answer[:40], revised[:40])
            return revised, tokens
        return answer, tokens
