"""
Orchestrator for the Multi-Agentic RAG Framework.

This module coordinates the execution of multiple agents to answer complex queries.
It implements various orchestration strategies including hierarchical, collaborative,
and sequential execution patterns.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio


class OrchestrationStrategy(Enum):
    """Enumeration of orchestration strategies."""
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"


class MultiAgenticOrchestrator:
    """
    Orchestrator for coordinating multiple agents in the RAG system.
    
    This class manages the workflow of query processing, from decomposition
    through retrieval to synthesis, coordinating multiple agents to work
    together efficiently.
    """
    
    def __init__(
        self,
        decomposition_agent: Any,
        retrieval_agents: List[Any],
        synthesis_agent: Any,
        strategy: OrchestrationStrategy = OrchestrationStrategy.HIERARCHICAL
    ):
        """
        Initialize the orchestrator.
        
        Args:
            decomposition_agent: Agent for query decomposition.
            retrieval_agents: List of retrieval agents.
            synthesis_agent: Agent for information synthesis.
            strategy: Orchestration strategy to use.
        """
        self.decomposition_agent = decomposition_agent
        self.retrieval_agents = retrieval_agents
        self.synthesis_agent = synthesis_agent
        self.strategy = strategy
        self.execution_history = []
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a query through the multi-agent system.
        
        Args:
            query: The input query to process.
            context: Optional context for the query.
            
        Returns:
            A dictionary containing the final answer and metadata.
        """
        context = context or {}
        
        # Step 1: Decompose the query
        decomposition_result = self.decomposition_agent.execute({
            "query": query,
            "context": context.get("additional_context", "")
        })
        
        sub_queries = decomposition_result.get("sub_queries", [query])
        
        # Step 2: Retrieve information for each sub-query
        if self.strategy == OrchestrationStrategy.HIERARCHICAL:
            retrieval_results = self._hierarchical_retrieval(sub_queries)
        elif self.strategy == OrchestrationStrategy.COLLABORATIVE:
            retrieval_results = self._collaborative_retrieval(sub_queries)
        elif self.strategy == OrchestrationStrategy.SEQUENTIAL:
            retrieval_results = self._sequential_retrieval(sub_queries)
        else:  # ADAPTIVE
            retrieval_results = self._adaptive_retrieval(sub_queries, decomposition_result)
        
        # Step 3: Synthesize the final answer
        synthesis_result = self.synthesis_agent.execute({
            "query": query,
            "retrieved_documents": retrieval_results.get("documents", []),
            "sub_query_results": retrieval_results.get("sub_query_results", [])
        })
        
        # Compile final result
        final_result = {
            "query": query,
            "answer": synthesis_result.get("answer", ""),
            "confidence": synthesis_result.get("confidence", 0.0),
            "sources": synthesis_result.get("sources", []),
            "metadata": {
                "decomposition": decomposition_result,
                "retrieval": retrieval_results,
                "synthesis": synthesis_result,
                "strategy": self.strategy.value
            }
        }
        
        # Record execution
        self.execution_history.append({
            "query": query,
            "result": final_result,
            "timestamp": decomposition_result.get("timestamp", "")
        })
        
        return final_result
    
    def _hierarchical_retrieval(self, sub_queries: List[str]) -> Dict[str, Any]:
        """
        Perform hierarchical retrieval with a master-worker pattern.
        
        Args:
            sub_queries: List of sub-queries to process.
            
        Returns:
            Dictionary containing retrieval results.
        """
        all_documents = []
        sub_query_results = []
        
        for sub_query in sub_queries:
            # Assign to the most appropriate retrieval agent
            agent = self._select_best_agent(sub_query)
            
            result = agent.execute({
                "query": sub_query,
                "top_k": 5
            })
            
            all_documents.extend(result.get("documents", []))
            sub_query_results.append({
                "sub_query": sub_query,
                "result": result
            })
        
        return {
            "documents": all_documents,
            "sub_query_results": sub_query_results,
            "strategy": "hierarchical"
        }
    
    def _collaborative_retrieval(self, sub_queries: List[str]) -> Dict[str, Any]:
        """
        Perform collaborative retrieval where agents work in parallel.
        
        Args:
            sub_queries: List of sub-queries to process.
            
        Returns:
            Dictionary containing retrieval results.
        """
        all_documents = []
        sub_query_results = []
        
        # Process all sub-queries in parallel with all agents
        for sub_query in sub_queries:
            agent_results = []
            
            for agent in self.retrieval_agents:
                result = agent.execute({
                    "query": sub_query,
                    "top_k": 3
                })
                agent_results.append(result)
                all_documents.extend(result.get("documents", []))
            
            sub_query_results.append({
                "sub_query": sub_query,
                "agent_results": agent_results
            })
        
        # Remove duplicates based on document content
        unique_documents = self._deduplicate_documents(all_documents)
        
        return {
            "documents": unique_documents,
            "sub_query_results": sub_query_results,
            "strategy": "collaborative"
        }
    
    def _sequential_retrieval(self, sub_queries: List[str]) -> Dict[str, Any]:
        """
        Perform sequential retrieval where each step informs the next.
        
        Args:
            sub_queries: List of sub-queries to process.
            
        Returns:
            Dictionary containing retrieval results.
        """
        all_documents = []
        sub_query_results = []
        context = {}
        
        for i, sub_query in enumerate(sub_queries):
            # Use context from previous retrievals
            agent = self._select_best_agent(sub_query)
            
            result = agent.execute({
                "query": sub_query,
                "top_k": 5,
                "context": context
            })
            
            all_documents.extend(result.get("documents", []))
            sub_query_results.append({
                "sub_query": sub_query,
                "result": result,
                "step": i + 1
            })
            
            # Update context for next iteration
            context[f"step_{i+1}_results"] = result
        
        return {
            "documents": all_documents,
            "sub_query_results": sub_query_results,
            "strategy": "sequential"
        }
    
    def _adaptive_retrieval(
        self,
        sub_queries: List[str],
        decomposition_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform adaptive retrieval based on query complexity and characteristics.
        
        Args:
            sub_queries: List of sub-queries to process.
            decomposition_result: Result from decomposition agent.
            
        Returns:
            Dictionary containing retrieval results.
        """
        complexity = decomposition_result.get("complexity_score", 0.5)
        
        # Choose strategy based on complexity
        if complexity < 0.3:
            return self._sequential_retrieval(sub_queries)
        elif complexity < 0.7:
            return self._hierarchical_retrieval(sub_queries)
        else:
            return self._collaborative_retrieval(sub_queries)
    
    def _select_best_agent(self, query: str) -> Any:
        """
        Select the best retrieval agent for a given query.
        
        Args:
            query: The query to process.
            
        Returns:
            The selected retrieval agent.
        """
        # Simple selection based on agent capabilities
        # In a real implementation, this would use learned policies
        
        if not self.retrieval_agents:
            raise ValueError("No retrieval agents available")
        
        # For now, return the first agent
        # TODO: Implement intelligent agent selection
        return self.retrieval_agents[0]
    
    def _deduplicate_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Remove duplicate documents from the list.
        
        Args:
            documents: List of documents.
            
        Returns:
            List of unique documents.
        """
        seen = set()
        unique = []
        
        for doc in documents:
            # Use content hash as identifier
            content = doc.get("content", "")
            if content and content not in seen:
                seen.add(content)
                unique.append(doc)
        
        return unique
    
    def get_execution_history(self, limit: int = 10) -> List[Dict]:
        """
        Get the execution history.
        
        Args:
            limit: Maximum number of entries to return.
            
        Returns:
            List of execution history entries.
        """
        return self.execution_history[-limit:]
