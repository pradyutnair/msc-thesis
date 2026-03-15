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

    def get_all_evidence(self) -> list[tuple[str, str]]:
        """Return (chunk_id, content) from ALL tool calls that retrieved text.

        Captures: read_chunk, search_and_read, keyword_search, semantic_search.
        """
        import re
        chunks: list[tuple[str, str]] = []
        seen_cids: set[str] = set()

        for a in self.actions:
            if not a.result:
                continue

            # Extract chunk IDs from any tool result
            chunk_ids = re.findall(r"Chunk ID: (\S+)", a.result)
            chunk_ids = [cid.rstrip(",.;:") for cid in chunk_ids]

            if a.tool_name == "read_chunk":
                # Parse the formatted read_chunk output
                current_cid = None
                content_lines = []
                in_content = False
                for line in a.result.split("\n"):
                    if line.startswith("[Chunk "):
                        if current_cid and content_lines:
                            content = "\n".join(content_lines).strip()
                            if content and current_cid not in seen_cids:
                                chunks.append((current_cid, content))
                                seen_cids.add(current_cid)
                        cid_match = re.search(r"\[Chunk (\S+?)\]", line)
                        current_cid = cid_match.group(1) if cid_match else None
                        content_lines = []
                        in_content = False
                    elif line.startswith("-" * 10):
                        in_content = True
                    elif line.startswith("=" * 10):
                        in_content = False
                    elif in_content:
                        content_lines.append(line)
                # Last chunk
                if current_cid and content_lines:
                    content = "\n".join(content_lines).strip()
                    if content and current_cid not in seen_cids:
                        chunks.append((current_cid, content))
                        seen_cids.add(current_cid)

            elif a.tool_name == "search_and_read":
                # search_and_read returns read_chunk formatted output
                current_cid = None
                content_lines = []
                in_content = False
                for line in a.result.split("\n"):
                    if line.startswith("[Chunk "):
                        if current_cid and content_lines:
                            content = "\n".join(content_lines).strip()
                            if content and current_cid not in seen_cids:
                                chunks.append((current_cid, content))
                                seen_cids.add(current_cid)
                        cid_match = re.search(r"\[Chunk (\S+?)\]", line)
                        current_cid = cid_match.group(1) if cid_match else None
                        content_lines = []
                        in_content = False
                    elif line.startswith("-" * 10):
                        in_content = True
                    elif line.startswith("=" * 10):
                        in_content = False
                    elif in_content:
                        content_lines.append(line)
                if current_cid and content_lines:
                    content = "\n".join(content_lines).strip()
                    if content and current_cid not in seen_cids:
                        chunks.append((current_cid, content))
                        seen_cids.add(current_cid)

            elif a.tool_name in ("keyword_search", "semantic_search"):
                # These return snippets with chunk IDs — store as lightweight evidence
                for cid in chunk_ids:
                    if cid not in seen_cids:
                        chunks.append((cid, a.result[:1000]))
                        seen_cids.add(cid)

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
