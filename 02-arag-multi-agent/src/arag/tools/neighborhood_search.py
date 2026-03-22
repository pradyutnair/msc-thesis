"""Neighborhood Search tool — explore chunks adjacent to a found chunk.

Exploits the fact that Wikipedia articles are split into consecutive chunks.
If chunk N is relevant, chunks N-2..N+2 often contain related information
that keyword search alone would miss.
"""

from typing import Any, Dict, List, Tuple, TYPE_CHECKING
from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext


class NeighborhoodSearchTool(BaseTool):
    """Read chunks adjacent to a given chunk ID to explore evidence topology."""

    def __init__(self, chunks_file: str = "data/chunks.json"):
        import json
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    @property
    def name(self) -> str:
        return "neighborhood_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "neighborhood_search",
                "description": (
                    "Read chunks adjacent to a given chunk ID. "
                    "Wikipedia articles are split into consecutive chunks, so "
                    "nearby chunks often contain related information. "
                    "Use this AFTER finding a relevant chunk to discover "
                    "additional evidence that keyword search might miss."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "The chunk ID to explore around.",
                        },
                        "radius": {
                            "type": "integer",
                            "description": "How many chunks before/after to read. Default 2.",
                        },
                    },
                    "required": ["chunk_id"],
                },
            },
        }

    def execute(
        self,
        context: "AgentContext",
        chunk_id: str = "",
        radius: int = 2,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            cid = int(chunk_id)
        except (ValueError, TypeError):
            return "Invalid chunk_id", {"retrieved_tokens": 0}

        radius = min(radius, 3)  # cap at 3
        results = []
        for offset in range(-radius, radius + 1):
            idx = cid + offset
            if 0 <= idx < len(self.chunks):
                text = self.chunks[idx]
                if ":" in text[:6]:
                    text = text.split(":", 1)[1].strip()
                results.append(f"[Chunk {idx}]\n{text[:500]}")

        if not results:
            return "No neighboring chunks found", {"retrieved_tokens": 0}

        output = "\n" + "=" * 40 + "\n"
        output = output.join(results)
        return output, {"retrieved_tokens": len(output) // 4}
