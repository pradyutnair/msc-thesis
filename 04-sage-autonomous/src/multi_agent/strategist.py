"""Strategist agent: plans, reviews, verifies, and generates final answers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from arag.core.llm import LLMClient
from multi_agent.blackboard import Blackboard, Hop

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _strip_llm_wrappers(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _loads_json_with_repair(raw: str) -> dict[str, Any]:
    text = _strip_llm_wrappers(raw or "")
    if not text:
        raise ValueError("empty JSON payload")
    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        cleaned = re.sub(r"[\x00-\x1f\x7f]", "", candidate).strip()
        if not cleaned:
            continue
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = cleaned.replace("\\'", "'")
            repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", repaired)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    raise ValueError("unable to parse JSON")


def _infer_expected_answer_type(question: str) -> str:
    q = (question or "").strip().lower()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had)\b", q):
        return "yes_no"
    if q.startswith("when ") or "what year" in q or "what date" in q:
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith("who "):
        return "person"
    if q.startswith("how many ") or q.startswith("how much "):
        return "number"
    return "entity"


def _normalize_final_answer(answer: str, expected_answer_type: str) -> str:
    text = (answer or "").strip()
    text = re.sub(r"^\s*(final answer\s*:|answer\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*[\.,;:!?]+$", "", text)
    text = re.sub(r"^(the answer is|answer is)\s+", "", text, flags=re.IGNORECASE).strip()
    if expected_answer_type == "yes_no":
        lowered = text.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"
    # Strip verbose refusal-like content
    _verbose_starts = [
        r"no\s+(?:document|evidence|information|chunk)",
        r"not\s+(?:mentioned|found|available|provided)",
        r"(?:the\s+)?documents?\s+(?:do not|don.t|does not|doesn.t)",
        r"(?:there\s+is\s+)?(?:no|insufficient)\s+(?:evidence|information)",
        r"(?:based on|according to)\s+(?:the\s+)?(?:provided|available|retrieved)",
    ]
    for pat in _verbose_starts:
        if re.match(pat, text, re.IGNORECASE):
            text = ""
            break
    return text


class Strategist:
    """Non-ReAct strategist: makes focused single-call decisions at key points."""

    def __init__(
        self,
        llm_client: LLMClient,
        plan_prompt: str | None = None,
        review_prompt: str | None = None,
        verify_prompt: str | None = None,
        answer_prompt: str | None = None,
        verbose: bool = False,
    ):
        self.llm = llm_client
        self.verbose = verbose

        self.plan_prompt = plan_prompt or (_PROMPTS_DIR / "strategist_plan.txt").read_text(encoding="utf-8")
        self.review_prompt = review_prompt or (_PROMPTS_DIR / "strategist_review.txt").read_text(encoding="utf-8")
        self.verify_prompt = verify_prompt or (_PROMPTS_DIR / "strategist_verify.txt").read_text(encoding="utf-8")
        self.answer_prompt = answer_prompt or (_PROMPTS_DIR / "sage_v3_answer.txt").read_text(encoding="utf-8")

    @staticmethod
    def _render(template: str, **kwargs: Any) -> str:
        rendered = template
        for key, value in kwargs.items():
            rendered = rendered.replace("{" + key + "}", str(value))
        return rendered

    async def _call_llm(self, prompt: str) -> str:
        response = await self.llm.async_chat(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        return _strip_llm_wrappers((response.get("message") or {}).get("content", ""))

    async def plan(self, question: str, blackboard: Blackboard) -> None:
        """Phase 1: Analyze question and create hop chain."""
        prompt = self._render(self.plan_prompt, question=question)
        raw = await self._call_llm(prompt)

        try:
            data = _loads_json_with_repair(raw)
        except ValueError:
            logger.warning("Failed to parse plan JSON, falling back to single hop")
            data = {
                "question_type": "single_hop",
                "expected_answer_type": _infer_expected_answer_type(question),
                "hops": [{"question": question, "depends_on": []}],
                "strategy_notes": "Fallback: single hop",
            }

        blackboard.question_type = str(data.get("question_type", "single_hop"))
        blackboard.expected_answer_type = str(data.get("expected_answer_type", "entity"))
        blackboard.strategy_notes = str(data.get("strategy_notes", ""))

        hops_raw = data.get("hops", [])
        if not isinstance(hops_raw, list) or not hops_raw:
            hops_raw = [{"question": question, "depends_on": []}]

        for i, h in enumerate(hops_raw):
            q = str(h.get("question", question))
            deps = h.get("depends_on", [])
            if not isinstance(deps, list):
                deps = []
            deps = [int(d) for d in deps if isinstance(d, (int, float)) and int(d) < i]
            blackboard.hop_chain.append(Hop(id=i, question=q, depends_on=deps))

        # Resolve placeholders for hops with no dependencies
        for hop in blackboard.hop_chain:
            hop.resolved_question = blackboard.resolve_placeholders(hop)

        if self.verbose:
            logger.info("Plan: %s hops, type=%s", len(blackboard.hop_chain), blackboard.question_type)
            for h in blackboard.hop_chain:
                logger.info("  Hop %d: %s (deps=%s)", h.id, h.question, h.depends_on)

    async def review(self, blackboard: Blackboard) -> dict[str, Any]:
        """Phase 2: Review progress and decide next action."""
        prompt = self._render(
            self.review_prompt,
            question=blackboard.question,
            blackboard_state=blackboard.get_state_summary(),
        )
        raw = await self._call_llm(prompt)

        try:
            data = _loads_json_with_repair(raw)
        except ValueError:
            # Default: if all resolved, synthesize; else investigate more
            if blackboard.all_hops_resolved():
                return {"mode": "synthesize", "reasoning": "All hops resolved", "revisions": []}
            return {"mode": "investigate_more", "reasoning": "Parse error fallback", "revisions": []}

        mode = str(data.get("mode", "investigate_more")).strip().lower()
        if mode not in {"synthesize", "investigate_more", "revise", "verify"}:
            mode = "investigate_more"

        revisions = data.get("revisions", [])
        if not isinstance(revisions, list):
            revisions = []

        return {
            "mode": mode,
            "reasoning": str(data.get("reasoning", "")),
            "revisions": revisions,
        }

    async def verify(self, blackboard: Blackboard) -> dict[str, Any]:
        """Phase 2.5: Verify evidence chain before final answer."""
        prompt = self._render(
            self.verify_prompt,
            question=blackboard.question,
            expected_answer_type=blackboard.expected_answer_type,
            evidence_summary=blackboard.get_evidence_summary(),
        )
        raw = await self._call_llm(prompt)

        try:
            data = _loads_json_with_repair(raw)
        except ValueError:
            return {"approved": True, "reasoning": "Parse error, approving by default",
                    "issues": [], "weak_hops": [], "suggested_actions": []}

        return {
            "approved": bool(data.get("approved", True)),
            "reasoning": str(data.get("reasoning", "")),
            "issues": data.get("issues", []),
            "weak_hops": data.get("weak_hops", []),
            "suggested_actions": data.get("suggested_actions", []),
        }


    def _fallback_comparison_answer(self, blackboard: Blackboard) -> str:
        """For comparison questions, extract candidate entities from the question and pick one."""
        q = blackboard.question
        # Try to extract "X or Y" pattern
        m = re.search(r',\s*(.+?)\s+or\s+(.+?)\s*\?', q)
        if m:
            option_a = m.group(1).strip().strip('"\'')
            option_b = m.group(2).strip().strip('"\'')
            # Check if any hop evidence mentions either option
            for hop in blackboard.hop_chain:
                if hop.answer:
                    ans_lower = hop.answer.lower()
                    if option_a.lower() in ans_lower:
                        return option_a
                    if option_b.lower() in ans_lower:
                        return option_b
            # Default to first option as best guess
            return option_a
        # For yes/no comparisons
        if blackboard.expected_answer_type == "yes_no":
            # Check if hop answers suggest same or different
            answers = [h.answer for h in blackboard.hop_chain if h.answer and h.status == "resolved"]
            if len(answers) >= 2:
                if answers[0].lower().strip() == answers[1].lower().strip():
                    return "yes"
                return "no"
        return ""

    async def generate_answer(self, blackboard: Blackboard) -> str:
        """Phase 3: Generate final answer from blackboard evidence."""
        # Build reasoning summary from hop chain
        reasoning_parts = []
        for hop in blackboard.hop_chain:
            q = hop.resolved_question or hop.question
            if hop.answer:
                reasoning_parts.append(f"Hop {hop.id}: {q} -> {hop.answer}")
            else:
                reasoning_parts.append(f"Hop {hop.id}: {q} -> [unresolved]")

        # Build knowledge outline: include hop answers prominently + entity KB + evidence
        knowledge_parts = []

        # Hop chain answers first (most important for the answer generator)
        for hop in blackboard.hop_chain:
            if hop.answer:
                q = hop.resolved_question or hop.question
                knowledge_parts.append(f"Hop {hop.id} ({q}): {hop.answer}")

        # Entity KB
        for key, info in blackboard.entity_kb.items():
            facts_str = "; ".join(info.facts[:5])
            if facts_str:
                knowledge_parts.append(f"{info.name}: {facts_str}")

        # Hop evidence text
        for hop in blackboard.hop_chain:
            if hop.evidence:
                for ev in hop.evidence[:3]:
                    text = str(ev.get("text", ""))
                    if text:
                        knowledge_parts.append(f"[Hop {hop.id} evidence]: {text[:500]}")

        prompt = self._render(
            self.answer_prompt,
            question=blackboard.question,
            expected_answer_type=blackboard.expected_answer_type,
            reasoning="\n".join(reasoning_parts),
            knowledge_outline="\n".join(knowledge_parts) if knowledge_parts else "No knowledge collected.",
        )
        raw = await self._call_llm(prompt)

        # Extract FINAL ANSWER from the response
        answer = ""
        m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
        if m:
            answer = m.group(1).strip()

        # If no FINAL ANSWER line found, try to extract from the full text
        if not answer:
            answer = raw.strip()

        answer = _normalize_final_answer(answer, blackboard.expected_answer_type)

        # Filter out refusal/unknown answers
        _refusals = {"unknown", "unresolved", "n/a", "none", "not found", "no evidence",
                      "no role", "no composer found", "no answer", "not available"}
        ans_lower = answer.lower().strip()
        if ans_lower in _refusals or ans_lower.startswith("no ") and len(ans_lower) < 30:
            answer = ""

        # Fallback: if answer is still empty, use the last resolved hop's answer
        if not answer:
            for hop in reversed(blackboard.hop_chain):
                if hop.answer and hop.status == "resolved":
                    candidate = _normalize_final_answer(hop.answer, blackboard.expected_answer_type)
                    if candidate.lower().strip() not in _refusals:
                        answer = candidate
                        break

        # For comparison questions with empty answer, try to pick one of the options from the question
        if not answer and blackboard.question_type == "comparison":
            answer = self._fallback_comparison_answer(blackboard)

        return answer
