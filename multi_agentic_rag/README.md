# Multi-Agentic RAG Framework

A modular, extensible framework for building state-of-the-art multi-agent Retrieval-Augmented Generation (RAG) systems.

## Overview

This framework provides a comprehensive foundation for developing advanced multi-agentic RAG systems. It combines insights from cutting-edge research papers and implements key architectural patterns including hierarchical agents, collaborative reasoning, agentic memory, and adaptive retrieval strategies.

## Architecture

The framework is built around several core components:

### 1. Agents

**Base Agent** (`agents/base_agent.py`)
- Abstract base class for all agents
- Provides common functionality: memory management, tool use, execution interface

**Decomposition Agent** (`agents/decomposition_agent.py`)
- Analyzes complex queries and breaks them into manageable sub-queries
- Assesses query complexity to determine if decomposition is needed
- Enables parallel processing of sub-queries

**Retrieval Agent** (`agents/retrieval_agent.py`)
- Retrieves relevant information from multiple data sources
- Supports dynamic strategy selection (vector, graph, web search)
- Can work with multiple retrieval methods simultaneously

**Synthesis Agent** (`agents/synthesis_agent.py`)
- Synthesizes information from multiple sources
- Resolves conflicts between sources using consistency voting
- Generates coherent, comprehensive final answers

### 2. Memory System

**Agentic Memory** (`memory/agentic_memory.py`)
- Hierarchical memory architecture inspired by human cognition
- Four memory types:
  - **Working Memory**: Immediate, short-term storage
  - **Short-Term Memory**: Recent interactions and context
  - **Long-Term Memory**: Persistent knowledge and learned patterns
  - **Episodic Memory**: Specific past experiences and outcomes
- Supports memory consolidation and selective forgetting
- Persistent storage for long-term memory

### 3. Orchestrator

**Multi-Agentic Orchestrator** (`orchestrator.py`)
- Coordinates execution of multiple agents
- Implements four orchestration strategies:
  - **Hierarchical**: Master-worker pattern with specialized agents
  - **Collaborative**: Parallel processing with all agents
  - **Sequential**: Step-by-step processing with context propagation
  - **Adaptive**: Dynamic strategy selection based on query complexity
- Tracks execution history for learning and debugging

## Key Features

### Modularity
Each component is designed to be independent and replaceable, allowing for easy experimentation with different approaches.

### Extensibility
The framework provides clear interfaces for adding new:
- Agent types
- Retrieval methods
- Memory systems
- Orchestration strategies

### Research-Driven
Built on insights from recent papers including:
- MA-RAG: Hierarchical multi-agent reasoning
- DecEx-RAG: Process-level supervision with MDP formulation
- HM-RAG: Multi-source retrieval and decision refinement
- GraphRAG: Knowledge graph-based retrieval

### Memory and Learning
Unlike stateless RAG systems, this framework includes sophisticated memory capabilities that enable agents to:
- Learn from past interactions
- Build persistent knowledge bases
- Refine strategies over time

## Installation

```bash
# Clone the repository
cd /path/to/msc-thesis

# Install dependencies (to be added)
pip install -r requirements.txt
```

## Quick Start

```python
from multi_agentic_rag.agents import DecompositionAgent, RetrievalAgent, SynthesisAgent
from multi_agentic_rag.orchestrator import MultiAgenticOrchestrator, OrchestrationStrategy
from multi_agentic_rag.memory import AgenticMemory

# Initialize memory
memory = AgenticMemory(
    working_memory_size=10,
    short_term_memory_size=100,
    long_term_memory_path="./memory.json"
)

# Initialize agents
decomposition_agent = DecompositionAgent(
    name="decomposer",
    llm=your_llm,
    memory=memory
)

retrieval_agent = RetrievalAgent(
    name="retriever",
    llm=your_llm,
    retrieval_methods=[vector_search, graph_search],
    memory=memory
)

synthesis_agent = SynthesisAgent(
    name="synthesizer",
    llm=your_llm,
    memory=memory
)

# Initialize orchestrator
orchestrator = MultiAgenticOrchestrator(
    decomposition_agent=decomposition_agent,
    retrieval_agents=[retrieval_agent],
    synthesis_agent=synthesis_agent,
    strategy=OrchestrationStrategy.ADAPTIVE
)

# Process a query
result = orchestrator.process_query(
    query="What are the key differences between transformer and RNN architectures?",
    context={"domain": "deep_learning"}
)

print(result["answer"])
print(f"Confidence: {result['confidence']}")
print(f"Sources: {result['sources']}")
```

## Research Directions

This framework is designed to support research in several key areas:

### 1. ReFrag (Re-Fragmentation)
- Dynamic reorganization of retrieved information
- Adaptive chunking strategies based on query characteristics
- Multi-perspective information synthesis

### 2. Collective Intelligence
- Advanced collaboration protocols between agents
- Debate and negotiation mechanisms
- Emergent intelligence from agent interactions

### 3. Agentic Memory
- Learning from successes and failures
- Memory consolidation strategies
- Transfer learning across tasks

### 4. Process-Level Supervision
- Fine-grained feedback at each reasoning step
- MDP-based formulation of RAG
- Reinforcement learning for agent policies

## Project Structure

```
multi_agentic_rag/
├── agents/
│   ├── base_agent.py           # Abstract base class
│   ├── decomposition_agent.py  # Query decomposition
│   ├── retrieval_agent.py      # Information retrieval
│   └── synthesis_agent.py      # Answer synthesis
├── retrieval/
│   └── (retrieval methods to be added)
├── memory/
│   └── agentic_memory.py       # Hierarchical memory system
├── tools/
│   └── (tools to be added)
├── config/
│   └── (configuration files to be added)
├── prompts/
│   └── (prompt templates to be added)
├── tests/
│   └── (unit tests to be added)
├── orchestrator.py             # Multi-agent coordination
└── README.md                   # This file
```

## Next Steps

### Immediate Priorities
1. Implement concrete retrieval methods (vector, graph, web)
2. Add LLM integration (OpenAI, Anthropic, local models)
3. Create prompt templates for each agent
4. Develop evaluation framework and benchmarks

### Research Experiments
1. Compare orchestration strategies on multi-hop QA datasets
2. Evaluate memory consolidation strategies
3. Implement and test ReFrag techniques
4. Develop collective reasoning protocols

### Integration Opportunities
1. Integrate GraphRAG for knowledge graph construction
2. Adapt DecEx-RAG's process-level supervision
3. Implement HM-RAG's decision refinement
4. Add support for multimodal data sources

## Contributing

This is a research project for an MSc thesis. Contributions, suggestions, and feedback are welcome!

## License

To be determined.

## References

See the technical report for a comprehensive list of references and related work.
