"""Agent execution context for ARAG."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""
    tool_name: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMUsageLog:
    """Log entry for an LLM interaction."""

    phase: str
    input_tokens: int
    output_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Context manager for agent execution state."""

    def __init__(self):
        # Retrieval statistics
        self.total_retrieved_tokens: int = 0
        self.retrieval_logs: List[RetrievalLog] = []

        # LLM usage statistics
        self.total_llm_tokens: int = 0
        self.llm_usage_logs: List[LLMUsageLog] = []

        # State management
        self.read_chunk_ids: Set[str] = set()
        self.search_history: List[Dict[str, Any]] = []

        # RLM recursion: depth in the delegate tree; delegations spawned at this node
        self.depth: int = 0
        self.delegation_count: int = 0
        self.final_answer: str = ""

    def add_retrieval_log(
        self,
        tool_name: str,
        tokens: int,
        metadata: Dict[str, Any] = None,
    ):
        """Add a retrieval log entry."""
        log = RetrievalLog(
            tool_name=tool_name,
            tokens=tokens,
            metadata=metadata or {},
        )
        self.retrieval_logs.append(log)
        self.total_retrieved_tokens += tokens

    def add_llm_usage(
        self,
        phase: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Dict[str, Any] = None,
    ):
        """Add an LLM usage log entry."""
        log = LLMUsageLog(
            phase=phase,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata or {},
        )
        self.llm_usage_logs.append(log)
        self.total_llm_tokens += input_tokens + output_tokens

    def mark_chunk_as_read(self, chunk_id: str):
        """Mark chunk as read."""
        self.read_chunk_ids.add(str(chunk_id))

    def is_chunk_read(self, chunk_id: str) -> bool:
        """Check if chunk has been read."""
        return str(chunk_id) in self.read_chunk_ids

    # Aliases for backward compatibility
    def add_read_chunk(self, chunk_id: str, content: str = None):
        """Alias for mark_chunk_as_read."""
        self.mark_chunk_as_read(chunk_id)

    def has_read_chunk(self, chunk_id: str) -> bool:
        """Alias for is_chunk_read."""
        return self.is_chunk_read(chunk_id)

    def get_read_chunk(self, chunk_id: str):
        """Check if chunk was read (returns None, content not stored)."""
        return None if not self.is_chunk_read(chunk_id) else ""

    def reset(self):
        """Reset context for new query."""
        self.retrieval_logs = []
        self.llm_usage_logs = []
        self.read_chunk_ids = set()
        self.search_history = []
        self.total_retrieved_tokens = 0
        self.total_llm_tokens = 0
        self.depth = 0
        self.delegation_count = 0
        self.final_answer = ""

    def get_summary(self) -> Dict[str, Any]:
        """Get context summary."""
        return {
            "total_retrieved_tokens": self.total_retrieved_tokens,
            "total_llm_tokens": self.total_llm_tokens,
            "retrieval_logs": [
                {
                    "tool_name": log.tool_name,
                    "tokens": log.tokens,
                    "metadata": log.metadata,
                }
                for log in self.retrieval_logs
            ],
            "llm_usage_logs": [
                {
                    "phase": log.phase,
                    "input_tokens": log.input_tokens,
                    "output_tokens": log.output_tokens,
                    "metadata": log.metadata,
                }
                for log in self.llm_usage_logs
            ],
            "chunks_read_count": len(self.read_chunk_ids),
            "chunks_read_ids": list(self.read_chunk_ids),
            "depth": self.depth,
            "delegation_count": self.delegation_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export context as dictionary."""
        return self.get_summary()
