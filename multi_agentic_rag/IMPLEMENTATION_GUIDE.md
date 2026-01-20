# Implementation Guide for Multi-Agentic RAG Framework

This guide provides detailed instructions for implementing and extending the multi-agentic RAG framework for your thesis research.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Implementing Retrieval Methods](#implementing-retrieval-methods)
3. [Integrating LLMs](#integrating-llms)
4. [Creating Custom Agents](#creating-custom-agents)
5. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
6. [Advanced Topics](#advanced-topics)

## Getting Started

### Installation

First, clone the repository and install the dependencies:

```bash
cd /path/to/msc-thesis/multi_agentic_rag
pip install -r requirements.txt
```

### Basic Usage

Here's a minimal example to get you started:

```python
from multi_agentic_rag.agents import DecompositionAgent, RetrievalAgent, SynthesisAgent
from multi_agentic_rag.orchestrator import MultiAgenticOrchestrator, OrchestrationStrategy
from multi_agentic_rag.memory import AgenticMemory

# Initialize components
memory = AgenticMemory()
decomposition_agent = DecompositionAgent(name="decomposer", llm=your_llm, memory=memory)
retrieval_agent = RetrievalAgent(name="retriever", llm=your_llm, retrieval_methods=[], memory=memory)
synthesis_agent = SynthesisAgent(name="synthesizer", llm=your_llm, memory=memory)

# Create orchestrator
orchestrator = MultiAgenticOrchestrator(
    decomposition_agent=decomposition_agent,
    retrieval_agents=[retrieval_agent],
    synthesis_agent=synthesis_agent,
    strategy=OrchestrationStrategy.HIERARCHICAL
)

# Process a query
result = orchestrator.process_query("Your question here")
print(result["answer"])
```

## Implementing Retrieval Methods

The framework is designed to work with multiple retrieval methods. Here's how to implement your own:

### 1. Vector-Based Retrieval

Create a new file `retrieval/vector_retrieval.py`:

```python
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorRetrieval:
    """Vector-based retrieval using FAISS and sentence transformers."""
    
    def __init__(self, index_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.name = "vector_retrieval"
        self.encoder = SentenceTransformer(embedding_model)
        self.index = faiss.read_index(index_path)
        self.documents = []  # Load your documents here
    
    def retrieve(self, query: str, top_k: int = 5, filters: Dict = None) -> List[Dict[str, Any]]:
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                "content": self.documents[idx]["content"],
                "source": self.documents[idx]["source"],
                "score": float(1 / (1 + distance))  # Convert distance to similarity
            })
        
        return results
```

### 2. Graph-Based Retrieval

Create a new file `retrieval/graph_retrieval.py`:

```python
from typing import List, Dict, Any
import networkx as nx

class GraphRetrieval:
    """Graph-based retrieval using NetworkX."""
    
    def __init__(self, graph_path: str):
        self.name = "graph_retrieval"
        self.graph = nx.read_gpickle(graph_path)
    
    def retrieve(self, query: str, top_k: int = 5, filters: Dict = None) -> List[Dict[str, Any]]:
        # Extract entities from query (use NER or LLM)
        entities = self._extract_entities(query)
        
        # Find relevant subgraph
        relevant_nodes = set()
        for entity in entities:
            if entity in self.graph:
                # Get neighbors within 2 hops
                neighbors = nx.single_source_shortest_path_length(self.graph, entity, cutoff=2)
                relevant_nodes.update(neighbors.keys())
        
        # Format results
        results = []
        for node in list(relevant_nodes)[:top_k]:
            results.append({
                "content": self.graph.nodes[node].get("content", ""),
                "source": f"graph_node_{node}",
                "score": 0.8  # Placeholder score
            })
        
        return results
    
    def _extract_entities(self, query: str) -> List[str]:
        # Placeholder: Use NER or LLM to extract entities
        return []
```

### 3. Web Search Retrieval

Create a new file `retrieval/web_retrieval.py`:

```python
from typing import List, Dict, Any
from duckduckgo_search import DDGS

class WebRetrieval:
    """Web-based retrieval using DuckDuckGo."""
    
    def __init__(self):
        self.name = "web_retrieval"
        self.ddgs = DDGS()
    
    def retrieve(self, query: str, top_k: int = 5, filters: Dict = None) -> List[Dict[str, Any]]:
        # Perform web search
        results = list(self.ddgs.text(query, max_results=top_k))
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.get("body", ""),
                "source": result.get("href", ""),
                "score": 0.7  # Placeholder score
            })
        
        return formatted_results
```

## Integrating LLMs

The framework is designed to work with any LLM. Here are examples for popular providers:

### OpenAI

```python
from openai import OpenAI

class OpenAILLM:
    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
```

### Anthropic Claude

```python
from anthropic import Anthropic

class ClaudeLLM:
    def __init__(self, model: str = "claude-3-opus-20240229"):
        self.client = Anthropic()
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text
```

## Creating Custom Agents

To create a custom agent, inherit from `BaseAgent`:

```python
from multi_agentic_rag.agents.base_agent import BaseAgent
from typing import Dict, Any

class CustomAgent(BaseAgent):
    def __init__(self, name: str, llm: Any, memory: Any = None):
        super().__init__(
            name=name,
            role="Custom Role",
            llm=llm,
            memory=memory
        )
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Implement your custom logic here
        query = task.get("query", "")
        
        # Use the LLM
        prompt = f"Process this query: {query}"
        response = self.llm.generate(prompt)
        
        # Add to memory
        self.add_to_memory({
            "task": "custom_task",
            "query": query,
            "response": response
        })
        
        return {
            "result": response,
            "metadata": {}
        }
```

## Evaluation and Benchmarking

### Setting Up Benchmarks

Create a new file `evaluation/benchmark.py`:

```python
import json
from typing import List, Dict, Any
from tqdm import tqdm

class Benchmark:
    def __init__(self, dataset_path: str):
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
    
    def evaluate(self, orchestrator: Any) -> Dict[str, float]:
        correct = 0
        total = len(self.dataset)
        
        for item in tqdm(self.dataset):
            query = item["question"]
            expected_answer = item["answer"]
            
            result = orchestrator.process_query(query)
            predicted_answer = result["answer"]
            
            if self._check_answer(predicted_answer, expected_answer):
                correct += 1
        
        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    def _check_answer(self, predicted: str, expected: str) -> bool:
        # Implement your answer checking logic
        # This could use exact match, F1 score, or LLM-based evaluation
        return predicted.lower().strip() == expected.lower().strip()
```

### Running Evaluations

```python
from evaluation.benchmark import Benchmark

# Load benchmark
benchmark = Benchmark("path/to/hotpotqa.json")

# Evaluate
results = benchmark.evaluate(orchestrator)
print(f"Accuracy: {results['accuracy']:.2%}")
```

## Advanced Topics

### 1. Implementing ReFrag

```python
class ReFragAgent(BaseAgent):
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        documents = task.get("documents", [])
        query = task.get("query", "")
        
        # Analyze query to determine optimal chunk size
        optimal_chunk_size = self._determine_chunk_size(query)
        
        # Re-chunk documents
        refragged_docs = self._refragment(documents, optimal_chunk_size)
        
        # Reorganize based on semantic relationships
        reorganized_docs = self._reorganize(refragged_docs, query)
        
        return {
            "documents": reorganized_docs,
            "chunk_size": optimal_chunk_size
        }
```

### 2. Implementing Collective Reasoning

```python
class DebateAgent(BaseAgent):
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        query = task.get("query", "")
        documents = task.get("documents", [])
        
        # Generate multiple candidate answers
        candidates = []
        for _ in range(3):
            answer = self._generate_answer(query, documents)
            candidates.append(answer)
        
        # Conduct debate
        final_answer = self._debate(candidates, query, documents)
        
        return {
            "answer": final_answer,
            "candidates": candidates
        }
    
    def _debate(self, candidates: List[str], query: str, documents: List) -> str:
        # Implement debate logic
        # Each candidate argues for itself
        # A moderator decides the winner
        pass
```

### 3. Implementing Memory Consolidation

```python
# In your main loop
for epoch in range(num_epochs):
    # Process queries
    for query in queries:
        result = orchestrator.process_query(query)
        
        # Add to memory with importance score
        memory.add({
            "query": query,
            "result": result,
            "importance": calculate_importance(result)
        }, memory_type="short_term")
    
    # Consolidate memory at the end of each epoch
    memory.consolidate()
```

## Tips and Best Practices

1. **Start Simple**: Begin with a basic implementation and gradually add complexity.
2. **Test Incrementally**: Test each component individually before integrating.
3. **Use Logging**: Add comprehensive logging to track agent behavior.
4. **Monitor Performance**: Track metrics like latency, token usage, and accuracy.
5. **Experiment**: Try different orchestration strategies and agent configurations.
6. **Document**: Keep detailed notes on your experiments and findings.

## Troubleshooting

### Common Issues

**Issue**: Agents not communicating properly
**Solution**: Check that the task dictionaries have the expected keys and formats.

**Issue**: Memory growing too large
**Solution**: Implement periodic consolidation and forgetting mechanisms.

**Issue**: Poor retrieval quality
**Solution**: Experiment with different embedding models and chunk sizes.

## Next Steps

1. Implement the retrieval methods described above
2. Integrate your preferred LLM
3. Set up evaluation on HotpotQA or 2WikiMultiHopQA
4. Experiment with different orchestration strategies
5. Implement one of the advanced research directions

Good luck with your thesis!
