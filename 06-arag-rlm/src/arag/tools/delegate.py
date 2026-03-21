"""RLM-style delegate tool with Parent Evidence Passing (PEP).

When delegating, the parent passes its already-retrieved chunk texts to the
child agent so the child does not start from scratch.  This eliminates
redundant retrieval and enables cross-hop information flow.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING, Union

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.config import Config
    from arag.core.context import AgentContext

MIN_CHILD_BUDGET_TOKENS = 2000


def _preview_evidence(text: str, max_chars: int = 1200) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _normalize_run_result(run_out: Any) -> Tuple[str, str]:
    if isinstance(run_out, tuple) and len(run_out) >= 2:
        return str(run_out[0]), _preview_evidence(str(run_out[1]))
    if isinstance(run_out, dict):
        ans = str(run_out.get("answer", ""))
        ev = run_out.get("evidence_preview", run_out.get("evidence", ""))
        return ans, _preview_evidence(str(ev) if ev is not None else "")
    ans = getattr(run_out, "answer", None)
    if ans is not None:
        ev = getattr(run_out, "evidence_preview", None) or getattr(run_out, "evidence", None)
        return str(ans), _preview_evidence(str(ev) if ev is not None else "")
    return str(run_out), ""


class DelegateTool(BaseTool):
    """Spawns a child agent with Parent Evidence Passing (PEP)."""

    def __init__(
        self,
        agent_factory: Callable[..., Any],
        config: Union["Config", Dict[str, Any]],
        remaining_budget_fn: Callable[["AgentContext"], int],
        get_parent_evidence_fn: Callable[["AgentContext"], str] | None = None,
    ):
        self._agent_factory = agent_factory
        self._config = config
        self._remaining_budget_fn = remaining_budget_fn
        self._get_parent_evidence_fn = get_parent_evidence_fn
        self._delegated_questions: List[str] = []

    def _cfg(self, dotted: str, default: Any) -> Any:
        if hasattr(self._config, "get"):
            return self._config.get(dotted, default)
        return default

    @property
    def name(self) -> str:
        return "delegate"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "delegate",
                "description": (
                    "Delegate a focused sub-question to a child agent. The child "
                    "receives your already-retrieved evidence as context so it does "
                    "not start from scratch. Use for multi-hop questions where you "
                    "need an intermediate fact.\n\n"
                    "IMPORTANT: Ask specific factual questions, NOT identity questions.\n"
                    "GOOD: 'Where was Erik Hort born?', 'When did Acornsoft close?'\n"
                    "BAD: 'Who is Erik Hort?', 'What is Acornsoft?'"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sub_question": {
                            "type": "string",
                            "description": (
                                "A specific factual sub-question (e.g., 'Where was X born?', "
                                "'When did Y die?'). Do NOT ask identity questions like 'Who is X?'."
                            ),
                        },
                        "expected_answer_type": {
                            "type": "string",
                            "description": (
                                "What form the answer should take (e.g. 'place name', 'date', "
                                "'person name', 'yes/no', 'number')."
                            ),
                        },
                        "main_question": {
                            "type": "string",
                            "description": "The original user question for global context.",
                        },
                    },
                    "required": ["sub_question", "expected_answer_type", "main_question"],
                },
            },
        }

    def _is_duplicate(self, sub_question: str) -> bool:
        normalized = sub_question.strip().lower()
        for prev in self._delegated_questions:
            if prev == normalized:
                return True
        return False

    def execute(
        self,
        context: "AgentContext",
        sub_question: str,
        expected_answer_type: str,
        main_question: str,
    ) -> Tuple[str, Dict[str, Any]]:
        max_children = int(self._cfg("agent.delegate.max_children", 4))
        child_fraction = float(self._cfg("agent.delegate.child_budget_fraction", 0.35))
        min_child = int(self._cfg("agent.delegate.min_child_budget_tokens", MIN_CHILD_BUDGET_TOKENS))

        if context.depth > 0:
            return (
                "Error: delegation is only allowed at the root agent (depth 0). "
                "Use search/read tools to answer without delegating.",
                {"error": "depth_limit", "depth": context.depth},
            )

        if context.delegation_count >= max_children:
            return (
                f"Error: delegation limit reached ({max_children} child calls). "
                "Answer using existing evidence or search tools directly.",
                {"error": "max_children", "delegation_count": context.delegation_count},
            )

        # Dedup: refuse to delegate the same question twice
        if self._is_duplicate(sub_question):
            return (
                f"Error: you already delegated this question. "
                "Try searching directly or rephrase the sub-question.",
                {"error": "duplicate_delegation", "sub_question": sub_question},
            )

        remaining = int(self._remaining_budget_fn(context))
        child_budget = int(remaining * child_fraction)

        if child_budget < min_child:
            return (
                f"Error: insufficient budget for delegation. "
                "Search directly instead.",
                {"error": "child_budget_too_small", "remaining_budget": remaining},
            )

        # Track this delegation
        self._delegated_questions.append(sub_question.strip().lower())
        context.delegation_count += 1
        child_depth = context.depth + 1

        # Gather parent evidence (PEP)
        parent_evidence = ""
        if self._get_parent_evidence_fn is not None:
            parent_evidence = self._get_parent_evidence_fn(context)

        try:
            child = self._agent_factory(
                sub_question,
                expected_answer_type,
                main_question,
                child_depth,
                child_budget,
                parent_evidence,
            )
            run_out = child.run(sub_question)
        except Exception as e:
            context.delegation_count -= 1
            return f"Error: child agent failed: {e}", {"error": str(e)}

        answer, evidence_preview = _normalize_run_result(run_out)
        text = f"Answer: {answer}\nEvidence: {evidence_preview}"
        log: Dict[str, Any] = {
            "delegated": True,
            "child_depth": child_depth,
            "child_budget": child_budget,
            "delegation_count": context.delegation_count,
            "parent_evidence_chars": len(parent_evidence),
        }
        return text, log
