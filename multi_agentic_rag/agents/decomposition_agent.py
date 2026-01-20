"""
Decomposition Agent for the Multi-Agentic RAG Framework.

This agent is responsible for analyzing complex queries and breaking them down
into simpler, more manageable sub-queries that can be processed in parallel.
"""

from typing import Dict, List, Any
from .base_agent import BaseAgent


class DecompositionAgent(BaseAgent):
    """
    Agent specialized in decomposing complex queries into sub-queries.
    
    This agent analyzes the input query, identifies its complexity,
    and breaks it down into a series of simpler sub-queries that can
    be processed independently by retrieval agents.
    """
    
    def __init__(self, name: str, llm: Any, memory: Any = None, tools: List[Any] = None):
        """
        Initialize the Decomposition Agent.
        
        Args:
            name: The unique name of the agent.
            llm: The language model for reasoning.
            memory: Optional memory system for the agent.
            tools: Optional list of tools the agent can use.
        """
        super().__init__(
            name=name,
            role="Query Decomposition",
            llm=llm,
            memory=memory,
            tools=tools
        )
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the query decomposition task.
        
        Args:
            task: A dictionary containing:
                - query (str): The complex query to decompose.
                - context (str, optional): Additional context for the query.
                
        Returns:
            A dictionary containing:
                - sub_queries (List[str]): The list of decomposed sub-queries.
                - reasoning (str): The reasoning behind the decomposition.
                - complexity_score (float): A score indicating query complexity.
        """
        query = task.get("query", "")
        context = task.get("context", "")
        
        # Check if query needs decomposition
        complexity_score = self._assess_complexity(query)
        
        if complexity_score < 0.3:
            # Simple query, no decomposition needed
            result = {
                "sub_queries": [query],
                "reasoning": "Query is simple enough to be processed directly.",
                "complexity_score": complexity_score,
                "needs_decomposition": False
            }
        else:
            # Complex query, perform decomposition
            sub_queries = self._decompose_query(query, context)
            result = {
                "sub_queries": sub_queries,
                "reasoning": f"Query decomposed into {len(sub_queries)} sub-queries for parallel processing.",
                "complexity_score": complexity_score,
                "needs_decomposition": True
            }
        
        # Add to memory
        self.add_to_memory({
            "task": "decomposition",
            "query": query,
            "result": result
        })
        
        return result
    
    def _assess_complexity(self, query: str) -> float:
        """
        Assess the complexity of a query.
        
        Args:
            query: The input query.
            
        Returns:
            A complexity score between 0 and 1.
        """
        # Simple heuristic-based complexity assessment
        # In a real implementation, this would use the LLM
        
        complexity_indicators = [
            "and", "or", "compare", "contrast", "multiple", "several",
            "what are the differences", "how does", "explain the relationship"
        ]
        
        query_lower = query.lower()
        score = 0.0
        
        # Check for complexity indicators
        for indicator in complexity_indicators:
            if indicator in query_lower:
                score += 0.2
        
        # Check query length
        if len(query.split()) > 15:
            score += 0.3
        
        # Check for question marks (multiple questions)
        if query.count("?") > 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def _decompose_query(self, query: str, context: str = "") -> List[str]:
        """
        Decompose a complex query into sub-queries.
        
        Args:
            query: The complex query to decompose.
            context: Additional context for the query.
            
        Returns:
            A list of sub-queries.
        """
        # This is a placeholder implementation
        # In a real system, this would use the LLM with a specific prompt
        
        prompt = f"""
        You are a query decomposition expert. Your task is to break down complex queries 
        into simpler, more specific sub-queries that can be answered independently.
        
        Query: {query}
        Context: {context}
        
        Please decompose this query into 2-5 sub-queries. Each sub-query should:
        1. Be self-contained and answerable independently
        2. Contribute to answering the original query
        3. Be specific and focused
        
        Return only the sub-queries, one per line.
        """
        
        # Placeholder: In real implementation, call self.llm with the prompt
        # For now, return a simple decomposition
        sub_queries = [
            f"Sub-query 1 for: {query}",
            f"Sub-query 2 for: {query}"
        ]
        
        return sub_queries
