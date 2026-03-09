"""Structured per-agent memory for M6 workers (inspired by AgentFlow)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_RESULT_DISPLAY_CHARS = 800


@dataclass
class MemoryAction:
    """A single tool execution recorded in memory."""

    step: int
    tool_name: str
    arguments: dict[str, Any]
    result: str


class Memory:
    """Accumulates tool actions and results for a single sub-question solve.

    Unlike raw conversation history, Memory provides a structured, queryable
    record that can be formatted for verification prompts.
    """

    def __init__(self) -> None:
        self.actions: list[MemoryAction] = []

    def add_action(
        self,
        step: int,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
    ) -> None:
        self.actions.append(MemoryAction(step, tool_name, arguments, result))

    def format_for_prompt(self) -> str:
        """Format memory for inclusion in LLM prompts."""
        if not self.actions:
            return "No actions taken yet."
        lines: list[str] = []
        for a in self.actions:
            args_str = ", ".join(f"{k}={v!r}" for k, v in a.arguments.items())
            lines.append(f"Step {a.step}: {a.tool_name}({args_str})")
            lines.append(f"  → {a.result[:_RESULT_DISPLAY_CHARS]}")
        return "\n".join(lines)

    def has_read_evidence(self) -> bool:
        """True if at least one read_chunk call returned content."""
        return any(
            a.tool_name == "read_chunk" and a.result and "(already read)" not in a.result
            for a in self.actions
        )

    def get_read_chunks(self) -> list[tuple[str, str]]:
        """Return (chunk_id, content) pairs from read_chunk calls."""
        chunks: list[tuple[str, str]] = []
        for a in self.actions:
            if a.tool_name != "read_chunk":
                continue
            if not a.result or "(already read)" in a.result:
                continue
            chunk_ids = a.arguments.get(
                "chunk_ids", a.arguments.get("chunk_id", []),
            )
            if isinstance(chunk_ids, (str, int)):
                chunk_ids = [str(chunk_ids)]
            for cid in chunk_ids:
                chunks.append((str(cid), a.result))
        return chunks

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize for pipeline results."""
        return [
            {
                "step": a.step,
                "tool_name": a.tool_name,
                "arguments": a.arguments,
                "result_preview": a.result[:200],
            }
            for a in self.actions
        ]
