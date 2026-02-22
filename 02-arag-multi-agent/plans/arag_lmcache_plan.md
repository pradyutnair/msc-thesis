# Technical PRD: MA²RAG — Multi-Agent Agentic RAG with Shared Evidence Caching

**Version**: 1.0  
**Author**: Pradyut Nair  
**Date**: 2026-02-21  
**Status**: Implementation-ready  
**Repo target**: `~/Documents/Thesis/02-multi-agent-arag/`

---

## 1. Problem Statement

Single-agent A-RAG (E4: Qwen3-30B-A3B + E5-base-v2) achieves 53.7% mean LLM-Accuracy across HotpotQA, MuSiQue, and 2WikiMultihop — a 7.4pp gap vs. GPT-4o-mini (61.1%). Failure taxonomy on MuSiQue (n=624 failures) reveals three systematic bottlenecks:

| Failure Mode | Count | % | Root Cause |
|---|---:|---:|---|
| Never searched hop-2 | 80 | 12.8% | Agent terminates after hop-1, never formulates hop-2 query |
| Searched but missed | 111 | 17.8% | Single query formulation insufficient to retrieve supporting evidence |
| Retrieved but couldn't synthesize | 367 | 58.8% | Correct evidence retrieved but context too noisy / reasoning fails |
| Decomposition failure | 63 | 10.1% | Query trajectory misaligned with gold decomposition |

**Goal**: Close the accuracy gap by ≥5pp mean LLM-Accuracy while reducing wall-clock latency on independent-hop questions, using only open-weight models on Snellius HPC.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
Input Question Q
       │
       ▼
┌─────────────────────────┐
│   DECOMPOSER MODULE     │  Classify Q → {comparison, bridge, single-hop}
│   (Qwen3-30B-A3B)       │  Emit sub-queries {q1..qN} + dependency DAG
└────────────┬────────────┘
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
┌────────┐┌────────┐┌────────┐
│Agent 1 ││Agent 2 ││Agent 3 │  Each: full A-RAG ReAct loop
│  (q1)  ││  (q2)  ││  (q3)  │  Tools: keyword_search, semantic_search, chunk_read
└───┬────┘└───┬────┘└───┬────┘
    │         │         │
    ▼         ▼         ▼
┌─────────────────────────────┐
│   SHARED EVIDENCE CACHE     │  In-memory dict: doc_id → {text, embedding, kv_hash}
│   (cross-agent read/write)  │  Cache-hit deduplication + optional KV block reuse
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────┐
│   AGGREGATOR MODULE     │  Evidence dedup → conflict resolution → synthesis
│   (Qwen3-30B-A3B)       │  Verify sub-answers against evidence
└────────────┬────────────┘
             │
             ▼
        Final Answer
```

### 2.2 Dependency-Aware Dispatch

Sub-questions have one of three dependency types:

- **Independent** (comparison questions): q1 and q2 execute in parallel, q3 (comparison) waits for both.
- **Sequential** (bridge questions): q1 executes first; q2's query template contains a placeholder filled by q1's answer.
- **Single-hop**: No decomposition, single agent runs A-RAG directly (fallback).

The decomposer outputs a JSON structure:

```json
{
  "type": "bridge",
  "sub_questions": [
    {"id": "q1", "text": "Who directed Inception?", "depends_on": []},
    {"id": "q2", "text": "What year was {q1.answer} born?", "depends_on": ["q1"]}
  ]
}
```

The dispatcher uses `asyncio.gather()` for independent sub-questions and sequential `await` for dependent ones.

---

## 3. Module Specifications

### 3.1 Decomposer Module

**File**: `src/decomposer.py`

**Input**: Original question Q (str), question metadata (dataset, qid)

**Output**: `DecompositionResult` dataclass:
```python
@dataclass
class SubQuestion:
    id: str                    # "q1", "q2", ...
    text: str                  # The sub-question text (may contain {qN.answer} templates)
    depends_on: list[str]      # IDs of sub-questions this depends on
    hop_type: str              # "entity_lookup", "attribute_lookup", "comparison", "filter"

@dataclass
class DecompositionResult:
    question_type: str         # "bridge", "comparison", "single_hop"
    sub_questions: list[SubQuestion]
    dependency_graph: dict[str, list[str]]  # adjacency list, topologically sortable
    original_question: str
```

**Implementation strategy**:
1. **Phase 1 (prompt-based)**: Few-shot decomposition prompt with 5 examples per question type (bridge, comparison) drawn from MuSiQue gold decompositions. Use Qwen3-30B-A3B with structured JSON output (constrained decoding via vLLM `guided_json`).
2. **Phase 2 (RL-trained)**: Fine-tune decomposer with GRPO using "Gain Beyond RAG" reward: `R = Acc(decomposed) - Acc(single_agent)`. Training data: 2,400 samples from MuSiQue + HotpotQA training sets (following s3 paper methodology). This is a stretch goal.

**Decomposition prompt template** (`prompts/decomposer.txt`):
```
You are a question decomposition expert. Given a multi-hop question, break it into
the minimal set of sub-questions needed to answer it. Output valid JSON.

Rules:
1. Each sub-question should be answerable with a single retrieval step.
2. If sub-question B needs the answer to A, write B's text with {A.id.answer} placeholder.
3. Minimize the number of sub-questions (typically 2-3 for multi-hop).
4. Classify the question type: "bridge" (chained entity lookup), "comparison" (parallel
   attribute comparison), or "single_hop" (no decomposition needed).

Examples:
Q: "What government position is held by the former combatant commander who oversaw
    Operation Iraqi Freedom?"
Output: {
  "question_type": "bridge",
  "sub_questions": [
    {"id": "q1", "text": "Who was the combatant commander who oversaw Operation Iraqi Freedom?", "depends_on": [], "hop_type": "entity_lookup"},
    {"id": "q2", "text": "What government position is held by {q1.answer}?", "depends_on": ["q1"], "hop_type": "attribute_lookup"}
  ]
}

Q: "Which film has a higher budget, Inception or The Dark Knight?"
Output: {
  "question_type": "comparison",
  "sub_questions": [
    {"id": "q1", "text": "What is the budget of Inception?", "depends_on": [], "hop_type": "attribute_lookup"},
    {"id": "q2", "text": "What is the budget of The Dark Knight?", "depends_on": [], "hop_type": "attribute_lookup"},
    {"id": "q3", "text": "Which is higher, {q1.answer} or {q2.answer}?", "depends_on": ["q1", "q2"], "hop_type": "comparison"}
  ]
}

Now decompose:
Q: "{question}"
```

**Fallback logic**: If decomposer output fails JSON parsing or produces only 1 sub-question, fall back to single-agent A-RAG (no decomposition). Track fallback rate as a metric.

---

### 3.2 Search Agent Module

**File**: `src/agent.py`

**Each agent is a full A-RAG-style ReAct agent** with the same three tools as the original A-RAG paper. Agents are instantiated per sub-question with independent state.

**Agent constructor**:
```python
class ARAGAgent:
    def __init__(
        self,
        sub_question: SubQuestion,
        evidence_cache: SharedEvidenceCache,
        vllm_client: AsyncOpenAI,
        retriever: HybridRetriever,
        max_iterations: int = 5,
        max_context_tokens: int = 4096,
        pre_loaded_evidence: list[Evidence] | None = None,  # for dependent agents
    ):
        self.sub_question = sub_question
        self.cache = evidence_cache
        self.client = vllm_client
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.trajectory: list[AgentStep] = []
        self.retrieved_docs: set[str] = set()  # chunk_ids already read

        # Pre-load evidence from upstream agents (for bridge dependencies)
        if pre_loaded_evidence:
            for ev in pre_loaded_evidence:
                self.trajectory.append(AgentStep(
                    thought=f"Pre-loaded from upstream: {ev.summary}",
                    action="pre_loaded",
                    observation=ev.text[:500],
                ))
```

**Tool definitions** (matching A-RAG paper exactly):

```python
TOOLS = [
    {
        "name": "keyword_search",
        "description": "Search for documents using exact keyword matching. Returns top-k chunks scored by keyword frequency × character length. Use for entity names, specific terms.",
        "parameters": {
            "query": {"type": "string", "description": "Keywords to search for"},
            "top_k": {"type": "integer", "default": 5, "description": "Number of results"}
        }
    },
    {
        "name": "semantic_search",
        "description": "Search for documents using dense embedding similarity. Returns top-k chunks by cosine similarity. Use for conceptual/fuzzy queries.",
        "parameters": {
            "query": {"type": "string", "description": "Natural language query"},
            "top_k": {"type": "integer", "default": 5, "description": "Number of results"}
        }
    },
    {
        "name": "chunk_read",
        "description": "Read the full content of a specific chunk by ID. Use after search to read promising chunks.",
        "parameters": {
            "chunk_id": {"type": "string", "description": "The chunk ID to read"}
        }
    },
    {
        "name": "finish",
        "description": "Submit your answer to the sub-question. Call this when you have sufficient evidence.",
        "parameters": {
            "answer": {"type": "string", "description": "Your answer to the sub-question"},
            "confidence": {"type": "float", "description": "0.0-1.0 confidence score"},
            "supporting_chunk_ids": {"type": "array", "items": {"type": "string"}}
        }
    }
]
```

**Cache-aware tool execution**:

When an agent calls `semantic_search` or `keyword_search`:
1. Execute the retrieval against the corpus index as normal.
2. For each result chunk_id, check `evidence_cache.has(chunk_id)`.
3. If cached: skip embedding computation, return cached text + metadata. Log as cache hit.
4. If not cached: retrieve from corpus, write to `evidence_cache`. Log as cache miss.

```python
async def execute_tool(self, action: str, params: dict) -> str:
    if action == "keyword_search":
        results = self.retriever.keyword_search(params["query"], params.get("top_k", 5))
        return self._process_search_results(results)

    elif action == "semantic_search":
        results = self.retriever.semantic_search(params["query"], params.get("top_k", 5))
        return self._process_search_results(results)

    elif action == "chunk_read":
        chunk_id = params["chunk_id"]
        if chunk_id in self.retrieved_docs:
            return "[Already read this chunk. Try a different one.]"

        # Check shared cache first
        cached = self.cache.get(chunk_id)
        if cached:
            self.cache.log_hit(chunk_id, self.sub_question.id)
            self.retrieved_docs.add(chunk_id)
            return cached.text

        # Cache miss: read from corpus
        text = self.retriever.read_chunk(chunk_id)
        self.cache.put(chunk_id, Evidence(
            chunk_id=chunk_id,
            text=text,
            source_agent=self.sub_question.id,
            timestamp=time.time(),
        ))
        self.cache.log_miss(chunk_id, self.sub_question.id)
        self.retrieved_docs.add(chunk_id)
        return text

    elif action == "finish":
        return params  # Handled by run loop
```

**ReAct system prompt** (`prompts/agent_system.txt`):
```
You are a focused search agent. Your task is to find the answer to one specific question
by searching a document corpus.

Your question: {sub_question}

You have three search tools:
- keyword_search(query, top_k): Exact keyword matching. Best for names, dates, specific terms.
- semantic_search(query, top_k): Semantic similarity search. Best for conceptual queries.
- chunk_read(chunk_id): Read the full text of a chunk found via search.
- finish(answer, confidence, supporting_chunk_ids): Submit your final answer.

Strategy guidelines:
1. Start with keyword_search if the question contains specific entity names.
2. Use semantic_search for broader conceptual queries.
3. Read the most promising chunks with chunk_read before answering.
4. You have a maximum of {max_iterations} tool calls. Be efficient.
5. If you find a clear answer, call finish immediately. Don't over-search.

Think step by step. For each step, output:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [JSON parameters]
```

**Agent run loop**:
```python
async def run(self) -> AgentResult:
    messages = [{"role": "system", "content": self.system_prompt}]

    for i in range(self.max_iterations):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
            extra_body={"guided_regex": REACT_REGEX}  # optional constrained decoding
        )

        text = response.choices[0].message.content
        thought, action, params = parse_react_output(text)

        self.trajectory.append(AgentStep(thought=thought, action=action, params=params))

        if action == "finish":
            return AgentResult(
                sub_question_id=self.sub_question.id,
                answer=params["answer"],
                confidence=params.get("confidence", 0.5),
                supporting_chunks=params.get("supporting_chunk_ids", []),
                trajectory=self.trajectory,
                num_iterations=i + 1,
            )

        observation = await self.execute_tool(action, params)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    # Max iterations reached — force answer from current context
    return self._force_finish(messages)
```

---

### 3.3 Shared Evidence Cache

**File**: `src/evidence_cache.py`

Thread-safe in-memory cache shared across all concurrent agents for a single question.

```python
import threading
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Evidence:
    chunk_id: str
    text: str
    embedding: Optional[list[float]] = None  # E5 embedding, if computed
    source_agent: str = ""                    # which agent first retrieved this
    timestamp: float = 0.0
    kv_block_hash: Optional[str] = None       # vLLM prefix cache hash (Phase 3)

@dataclass
class CacheStats:
    total_lookups: int = 0
    hits: int = 0
    misses: int = 0
    unique_docs: int = 0
    hit_log: list[dict] = field(default_factory=list)  # [{chunk_id, requesting_agent, source_agent}]

class SharedEvidenceCache:
    def __init__(self):
        self._store: dict[str, Evidence] = {}
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, chunk_id: str) -> Optional[Evidence]:
        with self._lock:
            self.stats.total_lookups += 1
            if chunk_id in self._store:
                self.stats.hits += 1
                return self._store[chunk_id]
            self.stats.misses += 1
            return None

    def put(self, chunk_id: str, evidence: Evidence) -> None:
        with self._lock:
            if chunk_id not in self._store:
                self.stats.unique_docs += 1
            self._store[chunk_id] = evidence

    def get_all(self) -> list[Evidence]:
        """Return all cached evidence (for aggregator pre-loading)."""
        with self._lock:
            return list(self._store.values())

    def log_hit(self, chunk_id: str, requesting_agent: str):
        with self._lock:
            self.stats.hit_log.append({
                "chunk_id": chunk_id,
                "requesting_agent": requesting_agent,
                "source_agent": self._store[chunk_id].source_agent,
                "type": "hit",
            })

    def log_miss(self, chunk_id: str, requesting_agent: str):
        with self._lock:
            self.stats.hit_log.append({
                "chunk_id": chunk_id,
                "requesting_agent": requesting_agent,
                "type": "miss",
            })

    def get_hit_rate(self) -> float:
        if self.stats.total_lookups == 0:
            return 0.0
        return self.stats.hits / self.stats.total_lookups

    def summary(self) -> dict:
        return {
            "total_lookups": self.stats.total_lookups,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.get_hit_rate(),
            "unique_docs_cached": self.stats.unique_docs,
        }
```

---

### 3.4 Dispatcher Module

**File**: `src/dispatcher.py`

Orchestrates parallel/sequential execution based on the dependency graph.

```python
import asyncio
from collections import defaultdict

class AgentDispatcher:
    def __init__(
        self,
        vllm_client: AsyncOpenAI,
        retriever: HybridRetriever,
        max_agent_iterations: int = 5,
    ):
        self.client = vllm_client
        self.retriever = retriever
        self.max_agent_iterations = max_agent_iterations

    async def dispatch(
        self,
        decomposition: DecompositionResult,
    ) -> list[AgentResult]:
        cache = SharedEvidenceCache()
        results: dict[str, AgentResult] = {}

        # Topological sort of dependency graph
        levels = self._topological_levels(decomposition)

        for level in levels:
            # All sub-questions in this level can run in parallel
            tasks = []
            for sq in level:
                # Gather evidence from completed dependencies
                pre_loaded = []
                for dep_id in sq.depends_on:
                    dep_result = results[dep_id]
                    # Resolve template placeholders
                    sq.text = sq.text.replace(
                        f"{{{dep_id}.answer}}",
                        dep_result.answer
                    )
                    # Pre-load supporting evidence
                    for chunk_id in dep_result.supporting_chunks:
                        cached_ev = cache.get(chunk_id)
                        if cached_ev:
                            pre_loaded.append(cached_ev)

                agent = ARAGAgent(
                    sub_question=sq,
                    evidence_cache=cache,
                    vllm_client=self.client,
                    retriever=self.retriever,
                    max_iterations=self.max_agent_iterations,
                    pre_loaded_evidence=pre_loaded if pre_loaded else None,
                )
                tasks.append(agent.run())

            # Execute all agents in this level concurrently
            level_results = await asyncio.gather(*tasks, return_exceptions=True)

            for sq, result in zip(level, level_results):
                if isinstance(result, Exception):
                    results[sq.id] = AgentResult(
                        sub_question_id=sq.id,
                        answer="[AGENT_ERROR]",
                        confidence=0.0,
                        supporting_chunks=[],
                        trajectory=[],
                        num_iterations=0,
                        error=str(result),
                    )
                else:
                    results[sq.id] = result

        return list(results.values()), cache

    def _topological_levels(self, decomp: DecompositionResult) -> list[list[SubQuestion]]:
        """Group sub-questions into levels for parallel execution."""
        in_degree = {sq.id: len(sq.depends_on) for sq in decomp.sub_questions}
        sq_map = {sq.id: sq for sq in decomp.sub_questions}
        dependents = defaultdict(list)
        for sq in decomp.sub_questions:
            for dep in sq.depends_on:
                dependents[dep].append(sq.id)

        levels = []
        queue = [sq_map[sid] for sid, deg in in_degree.items() if deg == 0]

        while queue:
            levels.append(queue)
            next_queue = []
            for sq in queue:
                for dep_id in dependents[sq.id]:
                    in_degree[dep_id] -= 1
                    if in_degree[dep_id] == 0:
                        next_queue.append(sq_map[dep_id])
            queue = next_queue

        return levels
```

---

### 3.5 Aggregator Module

**File**: `src/aggregator.py`

Takes sub-answers + all cached evidence and produces the final answer.

```python
class Aggregator:
    def __init__(self, vllm_client: AsyncOpenAI):
        self.client = vllm_client

    async def aggregate(
        self,
        original_question: str,
        decomposition: DecompositionResult,
        agent_results: list[AgentResult],
        evidence_cache: SharedEvidenceCache,
    ) -> FinalResult:
        # Build aggregation context
        sub_answers_text = "\n".join([
            f"Sub-question ({r.sub_question_id}): {self._get_sq_text(r.sub_question_id, decomposition)}\n"
            f"Answer: {r.answer} (confidence: {r.confidence:.2f})\n"
            f"Supporting evidence IDs: {r.supporting_chunks}"
            for r in agent_results
        ])

        # Deduplicated evidence summary (top chunks by frequency across agents)
        all_evidence = evidence_cache.get_all()
        evidence_text = "\n---\n".join([
            f"[{ev.chunk_id}]: {ev.text[:500]}"
            for ev in all_evidence[:15]  # Cap at 15 unique chunks
        ])

        prompt = AGGREGATOR_PROMPT.format(
            original_question=original_question,
            sub_answers=sub_answers_text,
            evidence=evidence_text,
        )

        response = await self.client.chat.completions.create(
            model="Qwen/Qwen3-30B-A3B",
            messages=[
                {"role": "system", "content": "You synthesize sub-answers into a final answer. Be concise and precise. Output only the final answer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )

        answer = strip_think_tags(response.choices[0].message.content)
        return FinalResult(
            answer=answer,
            sub_results=agent_results,
            cache_stats=evidence_cache.summary(),
        )
```

**Aggregator prompt** (`prompts/aggregator.txt`):
```
Original question: {original_question}

Sub-answers from search agents:
{sub_answers}

Supporting evidence (deduplicated):
{evidence}

Instructions:
1. Verify each sub-answer against its supporting evidence.
2. If sub-answers conflict, prefer the one with higher confidence and better evidence.
3. Synthesize the sub-answers into a single, concise final answer.
4. If the question asks for a comparison, explicitly state which entity satisfies the condition.
5. Output ONLY the final answer, nothing else.

Final answer:
```

---

### 3.6 Hybrid Retriever

**File**: `src/retriever.py`

Wraps the existing ARAG chunked index (E5-base-v2 embeddings + BM25 keyword index).

```python
import numpy as np
from pathlib import Path

class HybridRetriever:
    def __init__(
        self,
        corpus_dir: str,        # /projects/prjs1800/external/arag/data/{dataset}/
        embedding_model: str,   # "intfloat/e5-base-v2"
        device: str = "cuda",
    ):
        self.corpus = self._load_corpus(corpus_dir)
        self.embeddings = self._load_embeddings(corpus_dir)
        self.bm25_index = self._build_bm25(self.corpus)
        self.embedding_model = self._load_embedding_model(embedding_model, device)

    def keyword_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """BM25-style keyword search matching A-RAG's keyword_search tool."""
        scores = self.bm25_index.get_scores(query.split())
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            SearchResult(
                chunk_id=self.corpus[i]["chunk_id"],
                text=self.corpus[i]["text"][:200],  # Preview only
                score=float(scores[i]),
            )
            for i in top_indices if scores[i] > 0
        ]

    def semantic_search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Dense retrieval matching A-RAG's semantic_search tool."""
        query_emb = self.embedding_model.encode(query)
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            SearchResult(
                chunk_id=self.corpus[i]["chunk_id"],
                text=self.corpus[i]["text"][:200],
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    def read_chunk(self, chunk_id: str) -> str:
        """Full text retrieval by chunk ID."""
        return self.chunk_map[chunk_id]["text"]
```

---

## 4. Infrastructure & Serving

### 4.1 vLLM Server Configuration

**File**: `scripts/start_vllm.sh`

```bash
#!/bin/bash
#SBATCH --job-name=vllm-qwen3-30b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # Single H100 80GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

module load cuda/12.4

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.92 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 64 \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --disable-log-requests
```

Key settings:
- `--enable-prefix-caching`: vLLM's automatic prefix caching (APC) uses content-addressed hashing to reuse KV blocks across requests with shared prefixes. Since all agents share the same system prompt, the system prompt KV cache is computed once and reused across all concurrent agent calls.
- `--max-num-seqs 64`: Allows up to 64 concurrent sequences (enough for 3-4 agents × multiple tool calls).
- `--max-model-len 16384`: Agent contexts stay small (~4K tokens each); 16K accommodates aggregator context.

### 4.2 Embedding Server

**File**: `scripts/start_embedder.sh`

```bash
#!/bin/bash
#SBATCH --job-name=e5-embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # A100 40GB is sufficient
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

python -m vllm.entrypoints.openai.api_server \
    --model intfloat/e5-base-v2 \
    --task embed \
    --port 8001 \
    --dtype float16
```

Alternatively, load E5 embeddings in-process if pre-computed ARAG index is used (no live embedding needed for cached corpus).

### 4.3 Client Configuration

**File**: `src/config.py`

```python
from dataclasses import dataclass

@dataclass
class MA2RAGConfig:
    # Model serving
    vllm_base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-30B-A3B"

    # Decomposer
    decomposer_max_tokens: int = 512
    decomposer_temperature: float = 0.0

    # Agent
    max_agents: int = 4                 # Max parallel agents
    agent_max_iterations: int = 5       # Max ReAct loops per agent
    agent_max_context_tokens: int = 4096
    agent_temperature: float = 0.0

    # Retriever
    corpus_dir: str = "/projects/prjs1800/external/arag/data/{dataset}/"
    embedding_model: str = "intfloat/e5-base-v2"
    search_top_k: int = 5

    # Aggregator
    aggregator_max_tokens: int = 128
    aggregator_max_evidence_chunks: int = 15

    # Evaluation
    judge_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    judge_base_url: str = "http://localhost:8002/v1"
    datasets: list[str] = ("hotpotqa", "musique", "2wikimultihop")
    samples_per_dataset: int = 1000

    # Logging
    save_trajectories: bool = True
    output_dir: str = "results/"
```

---

## 5. Evaluation Framework

### 5.1 Metrics

| Metric | Definition | Source |
|---|---|---|
| **LLM-Accuracy** | DeepSeek-R1 judge binary correctness (0/1) | Primary. Matches E1-E4 evaluation protocol |
| **Contain-Accuracy** | Answer string containment check | Secondary. Quick sanity check |
| **EM** | Exact match after normalization | Standard. For paper comparisons |
| **F1** | Token-level F1 | Standard. For paper comparisons |
| **Wall-clock Latency** | Time from question input to final answer (seconds) | Efficiency contribution |
| **Total LLM Tokens** | Sum of prompt + completion tokens across all agents | Compute cost control |
| **Cache Hit Rate** | `cache.hits / cache.total_lookups` | Cache contribution |
| **Retrieval Recall** | Fraction of gold support paragraphs found in cached evidence | Retrieval quality |

### 5.2 Experiment Matrix

| ID | Config | Description | Purpose |
|---|---|---|---|
| **E4** | Single A-RAG (existing) | Qwen3-30B + E5, single agent | Baseline (already have results) |
| **M1** | MA²RAG, no cache | Decompose → parallel agents → aggregate, independent caches | Ablation: multi-agent without sharing |
| **M2** | MA²RAG, doc cache | Same + shared document cache | Core system |
| **M3** | MA²RAG, doc+KV cache | Same + vLLM prefix cache exploitation | Stretch: REFRAG-style optimization |
| **M4** | Single A-RAG, 2× iterations | Single agent with max_iterations=10 | Equal-compute control |
| **B0** | Non-agentic baseline (existing) | Top-5 retrieve → generate | Floor baseline (already have results) |

### 5.3 Evaluation Runner

**File**: `scripts/run_evaluation.py`

```python
async def evaluate_dataset(config: MA2RAGConfig, dataset: str):
    questions = load_questions(config.corpus_dir.format(dataset=dataset), config.samples_per_dataset)
    vllm_client = AsyncOpenAI(base_url=config.vllm_base_url, api_key="dummy")
    retriever = HybridRetriever(config.corpus_dir.format(dataset=dataset), config.embedding_model)
    decomposer = Decomposer(vllm_client, config)
    dispatcher = AgentDispatcher(vllm_client, retriever, config.agent_max_iterations)
    aggregator = Aggregator(vllm_client)

    predictions = []
    for q in tqdm(questions):
        t0 = time.time()

        # Step 1: Decompose
        decomp = await decomposer.decompose(q["question"])

        # Step 2: Dispatch agents
        agent_results, cache = await dispatcher.dispatch(decomp)

        # Step 3: Aggregate
        final = await aggregator.aggregate(q["question"], decomp, agent_results, cache)

        latency = time.time() - t0
        predictions.append({
            "qid": q["qid"],
            "question": q["question"],
            "prediction": final.answer,
            "gold_answer": q["answer"],
            "decomposition": asdict(decomp),
            "sub_results": [asdict(r) for r in agent_results],
            "cache_stats": cache.summary(),
            "latency_seconds": latency,
            "total_agent_iterations": sum(r.num_iterations for r in agent_results),
        })

    # Save predictions
    save_predictions(predictions, config.output_dir, dataset)

    # Run judges
    llm_acc = await run_deepseek_judge(predictions, config)
    em_f1 = compute_em_f1(predictions)
    contain_acc = compute_contain_accuracy(predictions)

    return {
        "dataset": dataset,
        "llm_accuracy": llm_acc,
        "contain_accuracy": contain_acc,
        "em": em_f1["em"],
        "f1": em_f1["f1"],
        "mean_latency": np.mean([p["latency_seconds"] for p in predictions]),
        "mean_cache_hit_rate": np.mean([p["cache_stats"]["hit_rate"] for p in predictions]),
        "mean_agent_iterations": np.mean([p["total_agent_iterations"] for p in predictions]),
    }
```

---

## 6. File Structure

```
02-multi-agent-arag/
├── src/
│   ├── __init__.py
│   ├── config.py                  # MA2RAGConfig dataclass
│   ├── decomposer.py             # Question decomposition
│   ├── agent.py                  # A-RAG ReAct agent
│   ├── evidence_cache.py         # SharedEvidenceCache
│   ├── dispatcher.py             # Dependency-aware parallel dispatch
│   ├── aggregator.py             # Evidence dedup + synthesis
│   ├── retriever.py              # HybridRetriever (BM25 + E5 dense)
│   ├── utils.py                  # ReAct parsing, think-tag stripping, answer normalization
│   └── types.py                  # All dataclasses (SubQuestion, AgentResult, Evidence, etc.)
├── prompts/
│   ├── decomposer.txt            # Few-shot decomposition prompt
│   ├── agent_system.txt          # Agent ReAct system prompt
│   └── aggregator.txt            # Aggregation prompt
├── configs/
│   ├── m1_no_cache.yaml
│   ├── m2_doc_cache.yaml
│   ├── m3_kv_cache.yaml
│   └── m4_single_2x.yaml
├── scripts/
│   ├── start_vllm.sh             # vLLM server launch (Snellius SBATCH)
│   ├── start_embedder.sh         # Embedding server launch
│   ├── run_evaluation.py         # Main evaluation entry point
│   ├── run_single_question.py    # Debug: run one question end-to-end
│   ├── analyze_cache_stats.py    # Cache hit rate analysis
│   ├── failure_taxonomy.py       # Extended failure taxonomy (reuse E4 script)
│   └── compare_experiments.py    # Generate comparison tables
├── jobs/
│   ├── submit_m1.sh
│   ├── submit_m2.sh
│   ├── submit_m3.sh
│   └── submit_m4.sh
├── tests/
│   ├── test_decomposer.py        # Unit tests: decomposition parsing, fallback
│   ├── test_dispatcher.py        # Unit tests: topological sort, parallel dispatch
│   ├── test_cache.py             # Unit tests: thread-safety, hit/miss logging
│   └── test_integration.py       # End-to-end: one question through full pipeline
├── analysis/
│   └── (generated at eval time)
├── results/
│   └── (generated at eval time)
└── README.md
```

---

## 7. Implementation Phases

### Phase 1: Core Pipeline (Week 1-2)

**Goal**: End-to-end pipeline running on a single question, no caching optimization.

1. Implement `types.py` with all dataclasses.
2. Implement `decomposer.py` with few-shot prompt (no RL).
3. Port `retriever.py` from existing ARAG reproduction code — reuse the same corpus loader and index.
4. Implement `agent.py` with ReAct loop and tool execution.
5. Implement `evidence_cache.py` (document-level cache only).
6. Implement `dispatcher.py` with topological sort and `asyncio.gather`.
7. Implement `aggregator.py` with evidence dedup prompt.
8. Test with `run_single_question.py` on 5 hand-picked questions (2 bridge, 2 comparison, 1 single-hop).

**Validation gate**: Pipeline produces sensible answers for all 5 test questions. Decomposer correctly identifies question type and dependency structure.

### Phase 2: Full Evaluation (Week 3-4)

**Goal**: Run M1 and M2 experiments on all three datasets (1000 questions each).

1. Implement `run_evaluation.py` with async batch processing.
2. Implement judge integration (reuse DeepSeek-R1 judge from E4).
3. Implement EM/F1 computation (reuse `04_eval_em_f1.py` from ARAG baselines).
4. Run M1 (no cache) and M2 (shared doc cache) on all datasets.
5. Run M4 (single agent 2× iterations) for equal-compute ablation.
6. Implement `compare_experiments.py` to generate comparison tables vs. E4 and B0.

**Validation gate**: M2 achieves ≥55% mean LLM-Accuracy (improvement over E4's 53.7%). Cache hit rate is measurable and non-trivial (>5% for bridge questions, >15% for comparison questions).

### Phase 3: KV Cache Optimization (Week 5-6, stretch)

**Goal**: Exploit vLLM prefix caching for cross-agent KV reuse.

1. Instrument vLLM to log prefix cache hit rates per request.
2. Structure agent prompts so shared system prompt + retrieved document text forms a common prefix across agents reading the same documents.
3. Measure TTFT (time to first token) reduction from prefix cache hits.
4. If significant: implement explicit document-text prefix ordering in agent context to maximize prefix sharing.
5. Run M3 experiment and compare latency vs. M2.

**Validation gate**: Measurable TTFT reduction (>10%) on questions where agents share retrieved documents.

### Phase 4: Analysis & Writing (Week 7-8)

1. Extended failure taxonomy on MA²RAG failures (same methodology as E4 MuSiQue taxonomy).
2. Cache hit rate analysis by question type (bridge vs. comparison).
3. Latency breakdown: decomposition time + parallel agent time + aggregation time.
4. Ablation analysis: contribution of each component.
5. Write paper/thesis chapter.

---

## 8. Key Design Decisions & Rationale

### 8.1 Why not LangGraph / DSPy?

- **LangGraph**: Adds heavyweight state machine abstraction. Our DAG dispatch is simple enough (~50 lines of asyncio). LangGraph's overhead and opinionated structure would complicate debugging agent trajectories and measuring cache performance.
- **DSPy**: Good for prompt optimization but its compilation approach doesn't align with our ReAct agent loop. We need fine-grained control over tool execution and cache integration at each step. DSPy modules abstract this away.
- **Decision**: Build directly on `asyncio` + `openai` client library. Minimal dependencies, maximum control, easier to debug and instrument.

### 8.2 Why Qwen3-30B-A3B?

- Best accuracy in our single-agent experiments (E4: 53.7% mean).
- MoE architecture (3B active params of 30B total) means lower per-token compute than a dense 30B, critical when running 3-4 agents concurrently.
- Fits on a single H100 80GB with vLLM.
- Strong instruction-following and tool-use capability.

### 8.3 Why not train the decomposer first?

Prompt-based decomposition is the 80/20 solution. MuSiQue gold decompositions show that most questions decompose into 2-3 sub-questions with clear dependency structure. Few-shot prompting with 5 examples per question type should handle >80% of cases correctly. RL training is the stretch goal for Phase 3+ if prompt-based decomposition proves to be a bottleneck.

### 8.4 Cache granularity: document-level vs. KV-level

- **Document-level (Phase 1-2)**: Simple Python dict, no infra changes. Prevents redundant `chunk_read` calls and redundant embedding computations. Expected benefit: reduced agent iterations and lower retrieval latency.
- **KV-level (Phase 3)**: Exploit vLLM's built-in prefix caching. If two agents read the same document, the second agent's prefill for that document text is a cache hit. Requires careful prompt structuring to maximize prefix overlap. Expected benefit: lower TTFT, higher throughput.

### 8.5 How the aggregator avoids the "retrieved but couldn't synthesize" problem

The aggregator receives:
1. Clean sub-answers (not raw retrieved passages).
2. Deduplicated evidence (not the full messy context each agent saw).
3. Confidence scores to weight conflicting sub-answers.

This means the aggregator's context is much smaller and more structured than a single agent's growing context window. The hypothesis: synthesis accuracy improves because the aggregator reasons over distilled sub-answers rather than raw multi-hop evidence chains.

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Decomposer produces bad sub-questions | Cascading failure: wrong sub-questions → wrong retrievals → wrong answer | Fallback to single-agent A-RAG when decomposition fails JSON parsing or produces single sub-question. Track fallback rate. |
| Concurrent vLLM requests cause OOM | Pipeline crashes mid-evaluation | Set `--max-num-seqs 64` conservatively. If OOM, reduce to 32 and serialize dependent agents. |
| Cache hit rate too low to matter | C2 contribution weakened | Even 5% hit rate on bridge questions is publishable if paired with latency analysis. Focus story on parallel dispatch (C1) if cache is weak. |
| Aggregator hallucinates despite correct sub-answers | Accuracy doesn't improve over single-agent | Add evidence verification step: aggregator must cite specific chunk_ids. If uncited, flag as low-confidence. |
| Wall-clock latency doesn't improve (decomposition overhead) | C1 contribution weakened | Measure and report decomposition time separately. For comparison questions (independent hops), parallel benefit should dominate. |

---

## 10. Dependencies

```
# Core
openai>=1.40.0          # AsyncOpenAI client for vLLM
asyncio                 # Built-in
numpy>=1.24.0
tqdm>=4.65.0
pyyaml>=6.0

# Retrieval
rank_bm25>=0.2.2        # BM25 keyword search
sentence-transformers>=2.7.0  # E5 embedding (if live encoding needed)
faiss-cpu>=1.7.4        # Dense retrieval index (or faiss-gpu)

# Evaluation
rouge-score>=0.1.2      # F1 computation
nltk>=3.8               # Tokenization for EM/F1

# Already available from ARAG reproduction
# - vLLM server (Snellius module)
# - DeepSeek-R1-Distill-Qwen-32B judge (separate vLLM instance)
# - ARAG chunked corpus + precomputed E5 embeddings
```

---

## 11. Success Criteria

| Criterion | Target | Measurement |
|---|---|---|
| Mean LLM-Accuracy (M2) | ≥ 56% (≥ 2.3pp over E4) | DeepSeek-R1 judge on 3×1000 questions |
| Mean LLM-Accuracy (M2) on MuSiQue specifically | ≥ 40% (≥ 2.4pp over E4) | DeepSeek-R1 judge on 1000 MuSiQue questions |
| Cache hit rate on comparison questions | ≥ 15% | SharedEvidenceCache stats |
| Cache hit rate on bridge questions | ≥ 5% | SharedEvidenceCache stats |
| Wall-clock latency on comparison Qs | ≤ 0.7× single-agent | Time measurement |
| "Never searched hop-2" failure rate | ≤ 5% (down from 12.8%) | Failure taxonomy analysis |
| Decomposer fallback rate | ≤ 15% | Tracked in predictions |
| M2 > M1 (cache helps) | Statistically significant | Paired t-test on per-question accuracy |
| M2 > M4 (architecture > compute) | M2 mean > M4 mean | Equal total LLM tokens comparison |