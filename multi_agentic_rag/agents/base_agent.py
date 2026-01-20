"""
Base Agent class for the Multi-Agentic RAG Framework.

This module provides the foundational structure for all agents in the system.
Each agent is designed to be autonomous, with its own memory, tools, and reasoning capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agentic RAG system.
    
    Attributes:
        name (str): The unique identifier for the agent.
        role (str): The specific role or function of the agent.
        memory (Optional[object]): The agent's memory system for storing past interactions.
        tools (List[object]): A list of tools the agent can use.
        llm (object): The language model used by the agent for reasoning.
    """
    
    def __init__(
        self,
        name: str,
        role: str,
        llm: Any,
        memory: Optional[Any] = None,
        tools: Optional[List[Any]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: The unique name of the agent.
            role: The role or function of the agent.
            llm: The language model for reasoning.
            memory: Optional memory system for the agent.
            tools: Optional list of tools the agent can use.
        """
        self.name = name
        self.role = role
        self.llm = llm
        self.memory = memory or []
        self.tools = tools or []
        self.created_at = datetime.now()
        
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task.
        
        This method must be implemented by all subclasses to define
        the specific behavior of each agent type.
        
        Args:
            task: A dictionary containing the task information.
            
        Returns:
            A dictionary containing the execution results.
        """
        pass
    
    def add_to_memory(self, entry: Dict[str, Any]) -> None:
        """
        Add an entry to the agent's memory.
        
        Args:
            entry: A dictionary containing the memory entry.
        """
        entry["timestamp"] = datetime.now().isoformat()
        if isinstance(self.memory, list):
            self.memory.append(entry)
        else:
            self.memory.add(entry)
    
    def get_memory(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve entries from the agent's memory.
        
        Args:
            query: Optional query to filter memory entries.
            limit: Maximum number of entries to return.
            
        Returns:
            A list of memory entries.
        """
        if isinstance(self.memory, list):
            return self.memory[-limit:]
        else:
            return self.memory.retrieve(query, limit)
    
    def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Use a specific tool from the agent's toolkit.
        
        Args:
            tool_name: The name of the tool to use.
            **kwargs: Arguments to pass to the tool.
            
        Returns:
            The result of the tool execution.
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.execute(**kwargs)
        raise ValueError(f"Tool '{tool_name}' not found in agent's toolkit.")
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', role='{self.role}')"
