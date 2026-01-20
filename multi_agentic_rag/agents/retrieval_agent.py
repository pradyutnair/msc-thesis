"""
Retrieval Agent for the Multi-Agentic RAG Framework.

This agent is responsible for retrieving relevant information from various
data sources based on the given query or sub-query.
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent


class RetrievalAgent(BaseAgent):
    """
    Agent specialized in retrieving relevant information from data sources.
    
    This agent can work with multiple retrieval strategies (vector search,
    graph search, web search) and can dynamically select the best strategy
    based on the query characteristics.
    """
    
    def __init__(
        self,
        name: str,
        llm: Any,
        retrieval_methods: List[Any],
        memory: Any = None,
        tools: List[Any] = None
    ):
        """
        Initialize the Retrieval Agent.
        
        Args:
            name: The unique name of the agent.
            llm: The language model for reasoning.
            retrieval_methods: List of retrieval methods (vector, graph, web, etc.).
            memory: Optional memory system for the agent.
            tools: Optional list of tools the agent can use.
        """
        super().__init__(
            name=name,
            role="Information Retrieval",
            llm=llm,
            memory=memory,
            tools=tools
        )
        self.retrieval_methods = retrieval_methods
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the retrieval task.
        
        Args:
            task: A dictionary containing:
                - query (str): The query to retrieve information for.
                - retrieval_strategy (str, optional): Specific strategy to use.
                - top_k (int, optional): Number of results to retrieve.
                - filters (Dict, optional): Additional filters for retrieval.
                
        Returns:
            A dictionary containing:
                - documents (List[Dict]): Retrieved documents with metadata.
                - retrieval_method (str): The method used for retrieval.
                - relevance_scores (List[float]): Relevance scores for each document.
        """
        query = task.get("query", "")
        strategy = task.get("retrieval_strategy", "auto")
        top_k = task.get("top_k", 5)
        filters = task.get("filters", {})
        
        # Select retrieval strategy
        if strategy == "auto":
            selected_method = self._select_retrieval_method(query)
        else:
            selected_method = self._get_method_by_name(strategy)
        
        # Perform retrieval
        documents = selected_method.retrieve(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        # Extract relevance scores
        relevance_scores = [doc.get("score", 0.0) for doc in documents]
        
        result = {
            "documents": documents,
            "retrieval_method": selected_method.name,
            "relevance_scores": relevance_scores,
            "query": query,
            "num_results": len(documents)
        }
        
        # Add to memory
        self.add_to_memory({
            "task": "retrieval",
            "query": query,
            "method": selected_method.name,
            "num_results": len(documents)
        })
        
        return result
    
    def _select_retrieval_method(self, query: str) -> Any:
        """
        Dynamically select the best retrieval method based on query characteristics.
        
        Args:
            query: The input query.
            
        Returns:
            The selected retrieval method.
        """
        # Simple heuristic-based selection
        # In a real implementation, this would use the LLM or learned policy
        
        query_lower = query.lower()
        
        # Check for graph-related queries
        if any(keyword in query_lower for keyword in ["relationship", "connected", "link", "network"]):
            for method in self.retrieval_methods:
                if "graph" in method.name.lower():
                    return method
        
        # Check for web-related queries
        if any(keyword in query_lower for keyword in ["latest", "recent", "current", "news"]):
            for method in self.retrieval_methods:
                if "web" in method.name.lower():
                    return method
        
        # Default to vector search
        for method in self.retrieval_methods:
            if "vector" in method.name.lower():
                return method
        
        # Fallback to first available method
        return self.retrieval_methods[0] if self.retrieval_methods else None
    
    def _get_method_by_name(self, name: str) -> Optional[Any]:
        """
        Get a retrieval method by its name.
        
        Args:
            name: The name of the retrieval method.
            
        Returns:
            The retrieval method or None if not found.
        """
        for method in self.retrieval_methods:
            if method.name.lower() == name.lower():
                return method
        return self.retrieval_methods[0] if self.retrieval_methods else None
