"""
Agentic Memory System for the Multi-Agentic RAG Framework.

This module implements a hierarchical memory system that allows agents to
store, retrieve, and learn from past interactions. The memory system includes
short-term, long-term, and episodic memory components.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import json


class AgenticMemory:
    """
    Hierarchical memory system for agentic RAG.
    
    This class implements a multi-layered memory architecture inspired by
    human memory systems, including:
    - Working Memory: Immediate, short-term storage
    - Short-Term Memory: Recent interactions and context
    - Long-Term Memory: Persistent knowledge and learned patterns
    - Episodic Memory: Specific past experiences and their outcomes
    """
    
    def __init__(
        self,
        working_memory_size: int = 10,
        short_term_memory_size: int = 100,
        long_term_memory_path: Optional[str] = None
    ):
        """
        Initialize the agentic memory system.
        
        Args:
            working_memory_size: Maximum size of working memory.
            short_term_memory_size: Maximum size of short-term memory.
            long_term_memory_path: Path to persist long-term memory.
        """
        self.working_memory = deque(maxlen=working_memory_size)
        self.short_term_memory = deque(maxlen=short_term_memory_size)
        self.long_term_memory = []
        self.episodic_memory = []
        self.long_term_memory_path = long_term_memory_path
        
        # Load long-term memory if path is provided
        if long_term_memory_path:
            self._load_long_term_memory()
    
    def add(self, entry: Dict[str, Any], memory_type: str = "working") -> None:
        """
        Add an entry to the specified memory type.
        
        Args:
            entry: The memory entry to add.
            memory_type: Type of memory ("working", "short_term", "long_term", "episodic").
        """
        # Add timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
        
        if memory_type == "working":
            self.working_memory.append(entry)
        elif memory_type == "short_term":
            self.short_term_memory.append(entry)
        elif memory_type == "long_term":
            self.long_term_memory.append(entry)
            if self.long_term_memory_path:
                self._save_long_term_memory()
        elif memory_type == "episodic":
            self.episodic_memory.append(entry)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    def retrieve(
        self,
        query: Optional[str] = None,
        memory_type: str = "all",
        limit: int = 10,
        time_range: Optional[timedelta] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve entries from memory.
        
        Args:
            query: Optional query to filter entries.
            memory_type: Type of memory to retrieve from.
            limit: Maximum number of entries to return.
            time_range: Optional time range to filter entries.
            
        Returns:
            List of memory entries.
        """
        # Collect entries from specified memory types
        entries = []
        
        if memory_type in ["all", "working"]:
            entries.extend(list(self.working_memory))
        if memory_type in ["all", "short_term"]:
            entries.extend(list(self.short_term_memory))
        if memory_type in ["all", "long_term"]:
            entries.extend(self.long_term_memory)
        if memory_type in ["all", "episodic"]:
            entries.extend(self.episodic_memory)
        
        # Filter by time range if specified
        if time_range:
            cutoff_time = datetime.now() - time_range
            entries = [
                e for e in entries
                if datetime.fromisoformat(e.get("timestamp", "")) > cutoff_time
            ]
        
        # Filter by query if specified
        if query:
            entries = self._filter_by_query(entries, query)
        
        # Sort by timestamp (most recent first)
        entries.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        return entries[:limit]
    
    def consolidate(self) -> None:
        """
        Consolidate memories by moving important short-term memories to long-term.
        
        This method implements a simple consolidation strategy where frequently
        accessed or high-importance memories are promoted to long-term storage.
        """
        # Find high-importance entries in short-term memory
        for entry in list(self.short_term_memory):
            importance = entry.get("importance", 0.5)
            if importance > 0.7:
                self.long_term_memory.append(entry)
        
        # Save consolidated long-term memory
        if self.long_term_memory_path:
            self._save_long_term_memory()
    
    def forget(self, memory_type: str = "working", criteria: Optional[Dict] = None) -> int:
        """
        Remove entries from memory based on criteria.
        
        Args:
            memory_type: Type of memory to forget from.
            criteria: Optional criteria for selective forgetting.
            
        Returns:
            Number of entries removed.
        """
        count = 0
        
        if memory_type == "working":
            if criteria:
                original_len = len(self.working_memory)
                self.working_memory = deque(
                    [e for e in self.working_memory if not self._matches_criteria(e, criteria)],
                    maxlen=self.working_memory.maxlen
                )
                count = original_len - len(self.working_memory)
            else:
                count = len(self.working_memory)
                self.working_memory.clear()
        
        elif memory_type == "short_term":
            if criteria:
                original_len = len(self.short_term_memory)
                self.short_term_memory = deque(
                    [e for e in self.short_term_memory if not self._matches_criteria(e, criteria)],
                    maxlen=self.short_term_memory.maxlen
                )
                count = original_len - len(self.short_term_memory)
            else:
                count = len(self.short_term_memory)
                self.short_term_memory.clear()
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary containing memory statistics.
        """
        return {
            "working_memory_size": len(self.working_memory),
            "short_term_memory_size": len(self.short_term_memory),
            "long_term_memory_size": len(self.long_term_memory),
            "episodic_memory_size": len(self.episodic_memory),
            "total_entries": (
                len(self.working_memory) +
                len(self.short_term_memory) +
                len(self.long_term_memory) +
                len(self.episodic_memory)
            )
        }
    
    def _filter_by_query(self, entries: List[Dict], query: str) -> List[Dict]:
        """
        Filter entries by query relevance.
        
        Args:
            entries: List of memory entries.
            query: Query string.
            
        Returns:
            Filtered list of entries.
        """
        # Simple keyword matching
        # In a real implementation, this would use semantic similarity
        query_lower = query.lower()
        filtered = []
        
        for entry in entries:
            entry_text = json.dumps(entry).lower()
            if query_lower in entry_text:
                filtered.append(entry)
        
        return filtered
    
    def _matches_criteria(self, entry: Dict, criteria: Dict) -> bool:
        """
        Check if an entry matches the given criteria.
        
        Args:
            entry: Memory entry.
            criteria: Criteria dictionary.
            
        Returns:
            True if entry matches criteria.
        """
        for key, value in criteria.items():
            if entry.get(key) != value:
                return False
        return True
    
    def _save_long_term_memory(self) -> None:
        """Save long-term memory to disk."""
        if self.long_term_memory_path:
            with open(self.long_term_memory_path, 'w') as f:
                json.dump(self.long_term_memory, f, indent=2)
    
    def _load_long_term_memory(self) -> None:
        """Load long-term memory from disk."""
        try:
            if self.long_term_memory_path:
                with open(self.long_term_memory_path, 'r') as f:
                    self.long_term_memory = json.load(f)
        except FileNotFoundError:
            self.long_term_memory = []
