"""Finish tool for explicit final answer submission."""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class FinishTool(BaseTool):
    """Submit the final answer explicitly."""

    @property
    def name(self) -> str:
        return "finish"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "finish",
                "description": (
                    "Submit your final answer. Use this when you have enough evidence. "
                    "Keep the answer short and do not call any other tools after this."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Your concise final answer.",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence between 0.0 and 1.0.",
                        },
                        "supporting_chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Chunk IDs that support the answer, if known.",
                        },
                    },
                    "required": ["answer"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        answer: str = "",
        confidence: float = 1.0,
        supporting_chunk_ids: List[str] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        cleaned = str(answer or "").strip()
        context.final_answer = cleaned
        return cleaned, {
            "retrieved_tokens": 0,
            "confidence": confidence,
            "supporting_chunk_ids": supporting_chunk_ids or [],
            "is_finish": True,
        }
