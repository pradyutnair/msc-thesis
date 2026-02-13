# A-RAG Analysis: Limitations, Exploitable Gaps, and Experiment PRD

**Author:** Pradyut Nair | **Date:** February 13, 2026  
**Context:** MSc Thesis — Multi-Agent RAG for Collaborative Search

---

## 1. A-RAG Paper Summary

**A-RAG** (Du et al., Feb 2026) introduces an agentic RAG framework with three hierarchical retrieval tools exposed directly to the LLM:

| Tool | Granularity | Function |
|------|------------|----------|
| `keyword_search` | Token-level | Exact lexical matching → returns snippets from matching chunks |
| `semantic_search` | Sentence-level | Dense embedding similarity → returns top-k chunk snippets |
| `chunk_read` | Chunk-level (~1000 tokens) | Full content access for a specific chunk ID |

**Agent loop:** ReAct-style (reason → tool call → observe → repeat). Single agent, sequential tool calls (intentionally no parallel calling). Context tracker prevents re-reading chunks.

**Key results (GPT-4o-mini backbone, LLM-Acc):**

| Dataset | Naive RAG | A-RAG (Naive) | A-RAG (Full) |
|---------|-----------|---------------|-------------|
| MuSiQue | 38.6% | 43.8% | **48.3%** |
| HotpotQA | 74.5% | 76.6% | **80.7%** |
| 2WikiMQA | 42.6% | 52.3% | **66.7%** |

**Scaling findings:** Performance improves with model size and test-time compute (more iterations). Hierarchical tools > single tool (ablation shows each tool contributes).

---

## 2. A-RAG Limitations & Exploitable Gaps

### 2.1 Single-Agent Bottleneck (Critical — Directly Exploitable)

**The limitation:** A-RAG uses a single agent with sequential tool calls. The paper *explicitly* avoids parallel tool calling:

> "We intentionally avoid parallel tool calling and other sophisticated designs to facilitate clean observation of how different interface configurations influence agent behavior."

**Why this matters for your thesis:**
- For multi-hop questions, the agent must sequentially: decompose → search hop 1 → read → search hop 2 → read → synthesize. This is inherently serial.
- Your thesis Q1 is *exactly* about parallelism. Multi-agent A-RAG with parallel search paths is the natural extension.
- A-RAG's MuSiQue score (48.3%) still has massive headroom. Your baseline data shows that even with IRCoT, hop-2 recall is only 24.9% — parallel agents searching different hops simultaneously could improve this.

**Exploitable experiment:** Run A-RAG's hierarchical tools but with N parallel agents, each handling a decomposed sub-question. Compare latency and accuracy vs. single-agent A-RAG.

### 2.2 No Inter-Agent Communication or Collaboration

**The limitation:** A-RAG is fundamentally a *single-agent* system. There is no mechanism for:
- Sharing intermediate findings between agents
- Voting or consensus on retrieved evidence
- Specialization (e.g., one agent for keyword search, another for semantic search)

**Why this matters:** Your thesis Q2 is about collaboration strategies. A-RAG proves hierarchical tools help a single agent — but what if *multiple* agents with different retrieval strategies could share their findings?

**Exploitable experiment:** Multi-agent A-RAG where:
- Agent A specializes in keyword search (good for entity-heavy queries)
- Agent B specializes in semantic search (good for paraphrased/abstract queries)
- A synthesis agent merges their findings
- Compare vs. single-agent A-RAG that can use all tools

### 2.3 Closed-Source Model Dependency

**The limitation:** A-RAG's primary results use GPT-4o-mini and GPT-5-mini via API. The paper does not evaluate open-source models comprehensively on all benchmarks. They test model scaling (Qwen3-30B, GPT-4o-mini, GPT-5-mini) but don't provide full results for smaller open models like Qwen2.5-7B.

**Why this matters:** Your setup on Snellius uses Qwen2.5-7B-Instruct. A-RAG's approach might degrade significantly with weaker tool-calling models. This is an opportunity:
- If A-RAG performs poorly with Qwen2.5-7B, multi-agent collaboration could compensate for individual agent weakness.
- "Collective intelligence from weak agents" is a compelling thesis narrative.

**Exploitable experiment:** Implement A-RAG with Qwen2.5-7B, measure the degradation, then show multi-agent versions recover performance.

### 2.4 No Retrieval Quality Analysis

**The limitation:** A-RAG reports only LLM-Acc and Contain-Match-Acc. They do **not** report:
- Retrieval Recall@k, Precision@k, MRR
- Per-hop retrieval analysis
- Supporting facts F1
- Token efficiency per-question (only aggregates)

**Why this matters:** Your baseline work already has rich per-hop retrieval analysis that A-RAG lacks. You can provide *deeper diagnostic analysis* than A-RAG, making your work methodologically stronger.

**Exploitable experiment:** Implement A-RAG's hierarchical tools, then apply your existing per-hop retrieval analysis framework. Show WHERE in the multi-hop chain the hierarchical tools help vs. where multi-agent coordination is still needed.

### 2.5 Context Tracker is Passive, Not Strategic

**The limitation:** A-RAG's context tracker only prevents re-reading chunks. It doesn't:
- Prioritize which chunks to read based on what's already been found
- Track which hops in a multi-hop chain have been satisfied
- Maintain a "retrieval plan" that adapts based on intermediate findings

**Why this matters:** A multi-agent system could have a *coordinator* agent that tracks the state of evidence gathering and strategically dispatches search agents to fill gaps. This is "intelligent orchestration" vs. A-RAG's "blind avoidance of duplicates."

### 2.6 Fixed Iteration Budget, No Adaptive Stopping

**The limitation:** A-RAG uses a fixed max_loops=15 budget. When exceeded, it forces a synthesis. There's no intelligent "confidence-based stopping" — the agent either finds the answer or runs out of budget.

**Why this matters:** In multi-agent systems, agents could vote on confidence. If 3/4 agents agree on an answer, you stop early. If they disagree, you allocate more budget. This directly addresses Q3 (inference-time scaling).

### 2.7 Chunking Strategy is Static

**The limitation:** A-RAG uses fixed ~1000-token chunks with sentence boundary alignment. This is a one-size-fits-all approach. Some questions need fine-grained sentence-level evidence; others need broader document context.

**Why this matters:** Multiple agents could operate at different chunk granularities. One agent works with small chunks (high precision), another with large chunks (high recall), and a synthesis agent combines their findings. This is a form of "retrieval diversity" that A-RAG cannot achieve with its fixed chunking.

### 2.8 Different Corpus & Evaluation Setup

**The limitation:** A-RAG uses LinearRAG's corpus setup (different from FlashRAG's wiki18_100w). They use LLM-as-judge evaluation (LLM-Acc) rather than standard EM/F1 metrics. This makes direct comparison with your FlashRAG baselines non-trivial.

**However, this is also an opportunity:** You can implement the *concept* of hierarchical retrieval tools within FlashRAG's infrastructure and compare fairly against your existing baselines.

---

## 3. Mapping to Your Research Questions

| A-RAG Gap | Your RQ | Intervention |
|-----------|---------|-------------|
| Single agent, sequential search | Q1 (Parallelism) | N parallel search agents with different sub-questions |
| No inter-agent communication | Q2 (Collaboration) | Shared evidence pool + consensus mechanisms |
| Fixed iteration budget | Q3 (Scaling) | Adaptive compute allocation based on agent agreement |
| No retrieval diversity | Q1+Q2 | Agents with different retrieval strategies (keyword vs. semantic) |
| GPT-4o-mini dependency | Q2 | Show weak agents collaborate to match/exceed strong single agent |

---

## 4. Connecting to Your Baseline Results

Your Day 1-5 ablation study revealed critical findings that directly motivate a multi-agent extension:

| Finding | Implication for Multi-Agent Design |
|---------|-----------------------------------|
| Reranker is the **only** component that significantly improves over Standard RAG | Multi-agent system MUST include reranking. Build it into each agent's pipeline. |
| 55.8% of MuSiQue questions have ZERO gold docs with Standard RAG | Single-shot retrieval is fundamentally broken for multi-hop. Need diverse search strategies. |
| IRCoT improves recall (21.4% → 33.3% on MuSiQue) but F1 gain is not significant (p=0.328 on HQA) | Iterative retrieval alone isn't enough — context dilution is real. Multi-agent parallel search avoids accumulating noise. |
| Per-hop recall decays: hop1=33.5%, hop2=11.6%, hop3=6.5%, hop4=3.2% | Each hop needs a dedicated search effort. Parallel agents per hop is the natural solution. |
| Refiners (RECOMP, SC) significantly hurt performance | Don't refine — instead, use targeted retrieval (like A-RAG's keyword search) to get precise evidence. |
| CoT hurts extractive QA (Reranker+CoT < Reranker alone) | Agent reasoning should happen internally (ReAct-style), not in the answer format. |
| Gold context ceiling: MuSiQue 63.9% (2-hop), 60.2% (3-hop), 45.4% (4-hop) | Even with perfect retrieval, reasoning is hard. Multi-agent deliberation may help reasoning too. |

---

## 5. Proposed Experiments

### Experiment A: Agentic RAG Baseline (Single-Agent A-RAG Reproduction)

**Goal:** Reproduce A-RAG's hierarchical tool approach with Qwen2.5-7B on your FlashRAG corpus. This becomes your "single-agent agentic baseline."

**Setup:**
- Model: Qwen2.5-7B-Instruct (vLLM)
- Corpus: wiki18_100w (FlashRAG)
- Tools: keyword_search (BM25 on FlashRAG corpus), semantic_search (E5-base-v2), chunk_read
- Agent loop: ReAct prompting, max 10 iterations
- Datasets: MuSiQue, HotpotQA, 2WikiMQA (same splits as your Days 1-5)
- Metrics: EM, F1, Recall@k, per-hop retrieval recall, token usage, latency

**Expected outcome:** Improvement over Standard RAG (your Day 1 baseline) but likely much weaker than A-RAG's GPT-4o-mini numbers due to model capability gap.

### Experiment B: Multi-Agent Parallel Search (Q1 — Parallelism)

**Goal:** Test whether parallel multi-agent search improves over single-agent agentic RAG.

**Setup:**
- Decomposer agent: Takes the question, outputs N sub-questions (one per hop)
- N Search agents: Each handles one sub-question using hierarchical tools (keyword + semantic + chunk_read)
- Aggregator agent: Merges all retrieved evidence, produces final answer
- Parallel execution: Search agents run concurrently (measure wall-clock time)
- Test with N = {1, 2, 4} agents

**Key metrics:**
- EM/F1 vs single-agent baseline
- Latency improvement (parallel vs sequential)
- Per-hop retrieval recall (does dedicated per-hop search help?)

### Experiment C: Multi-Agent Collaboration Strategies (Q2 — Collective Intelligence)

**Goal:** Compare different collaboration strategies between agents.

**Strategies to test:**
1. **Independent + Merge:** Agents search independently, all evidence pooled, single synthesizer
2. **Sequential handoff:** Agent 1 searches, passes findings to Agent 2 which searches for gaps
3. **Debate:** Two agents independently answer, then critique each other, re-search, converge
4. **Diverse retrieval:** Agent A uses only keyword search, Agent B uses only semantic search, results merged

**Key metric:** Does collaboration EM > max(individual agent EM)?

### Experiment D: Inference-Time Scaling (Q3 — Scaling Laws)

**Goal:** Plot performance vs. compute curves for multi-agent system.

**Variables:**
- Number of agents: 1, 2, 4, 8
- Iteration budget per agent: 3, 5, 10, 15
- Total token budget: fixed, measure how allocation across agents affects performance

**Key metric:** Performance vs. total tokens consumed. Find the diminishing returns point.

---

## 6. PRD: Experiment A — Single-Agent Agentic RAG on FlashRAG

This is the next experiment to implement. It serves as the agentic baseline that all multi-agent experiments (B, C, D) will build upon.

### 6.1 Objective

Implement A-RAG-style hierarchical retrieval tools within the FlashRAG ecosystem and evaluate a single ReAct agent on MuSiQue, HotpotQA, and 2WikiMQA using Qwen2.5-7B-Instruct.

### 6.2 System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AgenticRAGPipeline                │
│                                                     │
│  Input: question (str)                              │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │              ReAct Agent Loop                 │  │
│  │                                               │  │
│  │  1. THINK: Analyze question, plan strategy    │  │
│  │  2. ACT: Call one tool                        │  │
│  │  3. OBSERVE: Process tool output              │  │
│  │  4. Repeat until answer or max_iterations     │  │
│  │                                               │  │
│  │  Tools available:                             │  │
│  │  ┌─────────────┐ ┌──────────────┐            │  │
│  │  │ keyword_    │ │ semantic_    │            │  │
│  │  │ search()    │ │ search()     │            │  │
│  │  └─────┬───────┘ └──────┬───────┘            │  │
│  │        │                │                     │  │
│  │        ▼                ▼                     │  │
│  │  ┌─────────────────────────────┐              │  │
│  │  │      chunk_read()           │              │  │
│  │  │  (full chunk content)       │              │  │
│  │  └─────────────────────────────┘              │  │
│  │                                               │  │
│  │  ┌─────────────────────────────┐              │  │
│  │  │    finish(answer=...)       │              │  │
│  │  │  (terminate and return)     │              │  │
│  │  └─────────────────────────────┘              │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  State tracker:                                     │
│  - read_chunks: set[int]  (chunk IDs already read)  │
│  - retrieved_snippets: list[str]                    │
│  - iteration_count: int                             │
│  - token_usage: int                                 │
│                                                     │
│  Output: prediction (str), metadata (dict)          │
└─────────────────────────────────────────────────────┘
```

### 6.3 Component Specifications

#### 6.3.1 Corpus & Index (Reuse FlashRAG Infrastructure)

**Corpus:** wiki18_100w (already available at FlashRAG standard path)
- ~21M passages, 100-word chunks
- Pre-built E5-base-v2 FAISS index (already built from Days 1-5)

**Additional index needed:** BM25/keyword index over the same corpus
- Use `rank_bm25` library or Pyserini/Anserini
- Index the same wiki18_100w passages
- This enables keyword_search tool

**Implementation:**
```python
# BM25 index construction (one-time)
from rank_bm25 import BM25Okapi
import json

# Load corpus chunks
with open(corpus_path) as f:
    corpus = json.load(f)  # list of {"id": ..., "contents": ...}

tokenized_corpus = [doc["contents"].lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
# Pickle for reuse
```

#### 6.3.2 Tool Definitions

Each tool must be defined as a JSON function schema for the LLM's tool-calling interface, AND have a Python implementation.

**Tool 1: keyword_search**
```python
def keyword_search(keywords: list[str], top_k: int = 5) -> str:
    """
    Search corpus for chunks containing exact keyword matches.
    
    Args:
        keywords: List of keywords to search for (e.g., ["Albert Einstein", "Nobel Prize"])
        top_k: Number of results to return (default 5)
    
    Returns:
        String with matched chunk IDs and snippets (sentences containing keywords).
        Format: "Chunk {id}: {snippet}\n..."
    
    Implementation:
        1. For each chunk, count keyword occurrences weighted by keyword length
        2. Score = sum(count(keyword, chunk) * len(keyword)) for each keyword
        3. Return top-k chunks with highest scores
        4. For each chunk, extract only sentences containing ≥1 keyword as snippet
    """
```

**Tool 2: semantic_search**
```python
def semantic_search(query: str, top_k: int = 5) -> str:
    """
    Search corpus using dense embedding similarity.
    
    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
    
    Returns:
        String with matched chunk IDs and relevant sentences.
        Format: "Chunk {id} (score: {sim:.3f}): {matching_sentence}\n..."
    
    Implementation:
        1. Encode query with E5-base-v2
        2. FAISS similarity search over pre-built index
        3. Map sentence hits back to parent chunks
        4. Return top-k unique chunks with best-matching sentence as snippet
    """
```

**Tool 3: chunk_read**
```python
def chunk_read(chunk_id: int) -> str:
    """
    Read the full content of a specific chunk.
    
    Args:
        chunk_id: The numeric ID of the chunk to read
    
    Returns:
        Full chunk text, OR "This chunk has been read before" if already accessed.
    
    Implementation:
        1. Check if chunk_id in read_chunks set
        2. If yes: return "This chunk has been read before. Try a different chunk."
        3. If no: add to read_chunks, return full chunk content
    """
```

**Tool 4: finish**
```python
def finish(answer: str) -> str:
    """
    Submit the final answer and terminate the agent loop.
    
    Args:
        answer: The final answer to the question. Should be concise and direct.
    
    Implementation:
        Sets a termination flag. The answer is extracted from the argument.
    """
```

#### 6.3.3 ReAct Agent Prompt

```
SYSTEM PROMPT:
You are a research assistant that answers questions by searching a knowledge corpus.
You have access to the following tools:

1. keyword_search(keywords, top_k) - Search for chunks containing specific keywords. 
   Best for finding specific entities, names, dates, or technical terms.
   
2. semantic_search(query, top_k) - Search for semantically similar passages.
   Best for finding information about concepts, relationships, or paraphrased content.
   
3. chunk_read(chunk_id) - Read the full content of a specific chunk after finding 
   it via search. Use this when a snippet looks promising but you need more context.
   
4. finish(answer) - Submit your final answer. Be concise and direct.

Strategy guidelines:
- Start with keyword_search for specific entities mentioned in the question
- Use semantic_search when you need to find information described in different words
- Use chunk_read to get full context when snippets are promising but incomplete
- For multi-hop questions, search for each hop's evidence separately
- When you have enough evidence, call finish() with a concise answer
- Do NOT repeat searches you've already done
- If you cannot find the answer after several attempts, call finish() with your best guess

USER MESSAGE:
Question: {question}

Think step by step. For each step, explain your reasoning, then call exactly one tool.
```

#### 6.3.4 Agent Loop Implementation

```python
class AgenticRAGPipeline:
    def __init__(self, config):
        self.llm = vLLMGenerator(config)  # Qwen2.5-7B-Instruct
        self.keyword_searcher = BM25Searcher(config)
        self.semantic_searcher = E5Searcher(config)  # Reuse FlashRAG's retriever
        self.corpus = load_corpus(config)
        self.max_iterations = config.get("max_iterations", 10)
        self.tools = self._build_tool_schemas()
    
    def run_single(self, question: str) -> dict:
        """Run agent on a single question."""
        # State
        read_chunks = set()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"}
        ]
        metadata = {
            "iterations": 0,
            "tool_calls": [],
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "retrieved_chunk_ids": [],
            "read_chunk_ids": [],
        }
        
        for iteration in range(self.max_iterations):
            metadata["iterations"] += 1
            
            # LLM generates reasoning + tool call
            response = self.llm.generate_with_tools(
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            # Track tokens
            metadata["total_input_tokens"] += response.input_tokens
            metadata["total_output_tokens"] += response.output_tokens
            
            # Check if model wants to call a tool
            if response.tool_calls:
                tool_call = response.tool_calls[0]  # Single tool per iteration
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Execute tool
                if tool_name == "finish":
                    prediction = tool_args["answer"]
                    metadata["tool_calls"].append({"tool": "finish", "args": tool_args})
                    break
                    
                result = self._execute_tool(tool_name, tool_args, read_chunks, metadata)
                
                # Append assistant message + tool result to conversation
                messages.append({"role": "assistant", "content": response.content, 
                                "tool_calls": response.tool_calls})
                messages.append({"role": "tool", "content": result, 
                                "tool_call_id": tool_call.id})
            else:
                # Model generated text without tool call — extract answer
                prediction = self._extract_answer(response.content)
                break
        else:
            # Max iterations reached — force synthesis
            messages.append({"role": "user", "content": 
                "You've reached the maximum number of search iterations. "
                "Based on everything you've found so far, call finish() with your best answer."
            })
            response = self.llm.generate_with_tools(messages=messages, tools=self.tools)
            if response.tool_calls and response.tool_calls[0].function.name == "finish":
                prediction = json.loads(response.tool_calls[0].function.arguments)["answer"]
            else:
                prediction = self._extract_answer(response.content)
        
        return {"prediction": prediction, "metadata": metadata}
    
    def _execute_tool(self, tool_name, tool_args, read_chunks, metadata):
        """Execute a tool and return its string result."""
        metadata["tool_calls"].append({"tool": tool_name, "args": tool_args})
        
        if tool_name == "keyword_search":
            results = self.keyword_searcher.search(
                keywords=tool_args["keywords"], 
                top_k=tool_args.get("top_k", 5)
            )
            metadata["retrieved_chunk_ids"].extend([r["chunk_id"] for r in results])
            return self._format_search_results(results)
            
        elif tool_name == "semantic_search":
            results = self.semantic_searcher.search(
                query=tool_args["query"],
                top_k=tool_args.get("top_k", 5)
            )
            metadata["retrieved_chunk_ids"].extend([r["chunk_id"] for r in results])
            return self._format_search_results(results)
            
        elif tool_name == "chunk_read":
            chunk_id = tool_args["chunk_id"]
            if chunk_id in read_chunks:
                return "This chunk has been read before. Try a different chunk."
            read_chunks.add(chunk_id)
            metadata["read_chunk_ids"].append(chunk_id)
            return self.corpus[chunk_id]["contents"]
```

#### 6.3.5 Tool-Calling with Qwen2.5-7B via vLLM

**Critical implementation note:** Qwen2.5-7B-Instruct supports tool calling via its chat template. With vLLM, you need to use the OpenAI-compatible server or format the prompt correctly.

**Option A (Recommended): vLLM OpenAI-compatible server**
```bash
# Launch vLLM server with tool-calling support
python -m vllm.entrypoints.openai.api_server \
    --model /projects/prjs1800/models/base/qwen2.5-7b-instruct \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9
```

Then use the OpenAI Python client to call the server with tools.

**Option B: Direct vLLM generation with manual prompt formatting**
If tool-calling mode is unreliable, fall back to ReAct-style text generation:
- Format tools as text in the system prompt
- Parse model output for `Action: tool_name(args)` patterns
- This is more robust with smaller models that may struggle with structured tool calls

**Recommendation:** Try Option A first. If Qwen2.5-7B's tool-calling is unreliable (common with 7B models), switch to Option B with text-based ReAct parsing. Record which approach is used.

#### 6.3.6 Evaluation

**Datasets (reuse your existing FlashRAG splits):**
- MuSiQue test set (2,417 questions)
- HotpotQA test set (7,405 questions)  
- 2WikiMQA test set (12,576 questions)

**Metrics to compute (reuse your existing evaluation scripts):**
1. EM, F1 (answer quality — primary)
2. Retrieval Recall@k where k = number of chunks read (use your per-item GT analysis)
3. Per-hop retrieval recall for MuSiQue (reuse Day 2/4 analysis scripts)
4. Contain@k, Precision@k, MRR (reuse your post-hoc analysis)
5. Token usage: total input + output tokens per question (from metadata)
6. Latency: wall-clock seconds per question
7. Tool usage statistics: avg keyword_search calls, avg semantic_search calls, avg chunk_read calls per question

**Statistical tests:** Paired bootstrap (n=10,000) against:
- Standard RAG (Day 1 baseline)
- Reranker (Day 2 — current best)
- IRCoT (Day 4 — best iterative method)

### 6.4 Directory Structure

```
/projects/prjs1800/
├── experiments/
│   └── agentic_rag/
│       ├── configs/
│       │   ├── agentic_rag_musique.yaml
│       │   ├── agentic_rag_hotpotqa.yaml
│       │   └── agentic_rag_2wiki.yaml
│       ├── src/
│       │   ├── agent.py              # AgenticRAGPipeline class
│       │   ├── tools/
│       │   │   ├── __init__.py
│       │   │   ├── keyword_search.py  # BM25-based keyword search
│       │   │   ├── semantic_search.py # E5-based semantic search (wraps FlashRAG retriever)
│       │   │   ├── chunk_read.py      # Chunk reader with dedup tracking
│       │   │   └── tool_schemas.py    # JSON tool schemas for LLM
│       │   ├── react_prompt.py        # System prompt templates
│       │   ├── bm25_index.py          # BM25 index builder
│       │   └── utils.py              # Answer extraction, formatting
│       ├── scripts/
│       │   ├── build_bm25_index.py    # One-time BM25 index construction
│       │   ├── run_agentic_rag.py     # Main experiment runner
│       │   └── evaluate.py            # Evaluation + comparison with baselines
│       └── results/
│           ├── agentic_rag_musique.json
│           ├── agentic_rag_hotpotqa.json
│           └── analysis/
│               ├── per_hop_retrieval.json
│               ├── tool_usage_stats.json
│               └── comparison_with_baselines.md
```

### 6.5 SLURM Job Configuration

```bash
#!/bin/bash
#SBATCH --job-name=agentic_rag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/projects/prjs1800/results/logs/%j_agentic_rag.out

module load 2024
module load Python/3.11.5-GCCcore-13.2.0
source /projects/prjs1800/venvs/flashrag-venv/bin/activate

# Step 1: Build BM25 index (if not already built)
python experiments/agentic_rag/scripts/build_bm25_index.py \
    --corpus /projects/prjs1800/external/FlashRAG/data/wiki18_100w/corpus.jsonl \
    --output /projects/prjs1800/indexes/bm25_wiki18_100w.pkl

# Step 2: Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model /projects/prjs1800/models/base/qwen2.5-7b-instruct \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 &
VLLM_PID=$!
sleep 30  # Wait for server to start

# Step 3: Run experiments
for dataset in musique hotpotqa 2wikimultihopqa; do
    python experiments/agentic_rag/scripts/run_agentic_rag.py \
        --config experiments/agentic_rag/configs/agentic_rag_${dataset}.yaml \
        --output experiments/agentic_rag/results/agentic_rag_${dataset}.json
done

# Step 4: Evaluate
python experiments/agentic_rag/scripts/evaluate.py \
    --results_dir experiments/agentic_rag/results/ \
    --baseline_dir /projects/prjs1800/results/

kill $VLLM_PID
```

### 6.6 Configuration Template

```yaml
# agentic_rag_musique.yaml
model:
  name: "qwen2.5-7b-instruct"
  api_base: "http://localhost:8000/v1"
  api_key: "not-needed"
  temperature: 0.0
  max_tokens: 1024  # Per-iteration generation budget

agent:
  max_iterations: 10
  tool_call_mode: "auto"  # or "react_text" for fallback
  verbose: true  # Log each iteration for analysis

tools:
  keyword_search:
    bm25_index_path: "/projects/prjs1800/indexes/bm25_wiki18_100w.pkl"
    corpus_path: "/projects/prjs1800/external/FlashRAG/data/wiki18_100w/corpus.jsonl"
    default_top_k: 5
    snippet_max_sentences: 3
  semantic_search:
    model_path: "/projects/prjs1800/models/base/e5-base-v2"
    index_path: "/projects/prjs1800/indexes/e5_wiki18_100w.index"
    default_top_k: 5
  chunk_read:
    corpus_path: "/projects/prjs1800/external/FlashRAG/data/wiki18_100w/corpus.jsonl"

dataset:
  name: "musique"
  path: "RUC-NLPIR/FlashRAG_datasets"
  split: "test"

evaluation:
  metrics: ["em", "f1", "recall", "precision", "mrr"]
  gt_supporting_facts: true  # Compute per-hop retrieval metrics
  save_per_item: true  # Save per-question results for analysis
```

### 6.7 Expected Results & Success Criteria

| Metric | Standard RAG (Day 1) | Reranker (Day 2) | IRCoT (Day 4) | Agentic RAG (Expected) |
|--------|---------------------|------------------|---------------|----------------------|
| HotpotQA F1 | 42.01 | 47.42 | 42.46 | 44-50 |
| MuSiQue F1 | 13.03 | 15.52 | 14.29 | 14-18 |
| 2WikiMQA F1 | ~32* | ~35* | — | 33-38 |
| Avg iterations | 1 | 1 | ~3 | 4-7 |
| Token usage | ~2K | ~2K | ~8K | ~6-10K |

*\* Approximate from your retrieval metrics*

**Success criteria:**
- Agentic RAG F1 ≥ Reranker F1 on at least 2/3 datasets (proves tool autonomy helps)
- Per-hop retrieval recall improves on MuSiQue (proves adaptive search helps multi-hop)
- Clear tool usage patterns emerge (e.g., keyword search used more for entity-heavy hops)

**If Agentic RAG underperforms** (likely with Qwen2.5-7B tool calling):
- This is still a useful result — demonstrates that single-agent agentic RAG with weak models fails
- Motivates multi-agent approach: "collaboration compensates for weak individual agents"
- Document failure modes: Does the model fail to call tools? Call wrong tools? Fail to synthesize?

### 6.8 Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Qwen2.5-7B can't reliably do tool calling | Fall back to text-based ReAct parsing (Option B in 6.3.5) |
| BM25 index too large for memory | Use Pyserini's on-disk index instead of in-memory rank_bm25 |
| 10 iterations × 7K questions = slow | Run on 500-question subset first, then scale. Parallelize with batch processing. |
| Token context window overflow | Cap conversation history at 12K tokens, summarize older tool results |
| vLLM server instability | Use direct vLLM generation (offline mode) as fallback |

### 6.9 Post-Experiment Analysis Plan

After running Experiment A, produce:

1. **Master comparison table** (extend your Day 6 table with Agentic RAG row)
2. **Tool usage analysis:** What tools does the agent prefer? How does usage differ across datasets?
3. **Per-hop retrieval analysis:** Does agentic search improve hop-2/3/4 recall vs. Standard RAG and IRCoT?
4. **Failure categorization:** Apply your existing error taxonomy (retrieval miss / partial retrieval / reasoning failure) to Agentic RAG
5. **Qualitative examples:** Show 5-10 trajectories (agent's reasoning + tool calls) for successes and failures
6. **Multi-agent motivation document:** Based on failure modes, articulate which failures multi-agent could fix

This analysis directly feeds into Experiments B, C, D (multi-agent extensions).

---

## 7. Timeline

| Day | Task | Output |
|-----|------|--------|
| Day 1 | Build BM25 index, implement tool functions, test on 10 questions | Working tool implementations |
| Day 2 | Implement ReAct agent loop, test tool calling with Qwen2.5-7B | Working single-agent pipeline |
| Day 3 | Run on MuSiQue subset (500 questions), debug | Initial results + debugging |
| Day 4 | Run full evaluation on all 3 datasets | Full results |
| Day 5 | Analysis: comparison tables, per-hop retrieval, tool usage stats | Analysis report |
| Day 6 | Write multi-agent hypothesis based on failure modes | Multi-agent design doc for Experiment B |