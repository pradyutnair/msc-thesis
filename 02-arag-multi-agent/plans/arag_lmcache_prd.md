# MA²RAG: Multi-Agent Agentic RAG with Shared Caching

## Context

Single-agent A-RAG (E4, Qwen3-30B-A3B) achieves 53.7% mean LLM-Accuracy across HotpotQA/MuSiQue/2WikiMultihop — 7.4pp below the paper's GPT-4o-mini baseline (61.1%). MuSiQue failure taxonomy (n=624) reveals three addressable bottlenecks:

| Failure Mode | Rate | Targeted By |
|---|---|---|
| Retrieved but couldn't synthesize | 58.8% | Aggregator (structured synthesis) |
| Searched but missed | 17.8% | Parallel agents (redundant search) |
| Never searched hop-2 | 12.8% | Decomposer (explicit sub-questions) |
| Decomposition failure | 10.1% | Decomposer (classification + DAG) |

**Goal**: Close the gap by ≥5pp (target ≥58% mean) via multi-agent decompose-search-aggregate architecture with LMCache efficiency gains. All work modifies only `/projects/prjs1800/msc-thesis/02-arag-multi-agent/`.

---

## Architecture

```
Question Q
    │
    ▼
┌─────────────┐
│ DECOMPOSER  │  Classify type (comparison/bridge/single-hop)
│ (1 LLM call)│  Emit sub-queries + dependency DAG
└──────┬──────┘
       │
  ┌────┴────┐  topological wave dispatch
  ▼         ▼
┌───────┐ ┌───────┐   Wave 1: independent sub-Qs (asyncio.gather)
│Agent 1│ │Agent 2│   Each agent = full ReAct A-RAG loop
│(ReAct)│ │(ReAct)│   Tools: keyword_search, semantic_search, chunk_read
└───┬───┘ └───┬───┘   Write evidence → SharedEvidenceCache
    │         │
    ▼         ▼
┌───────────────────┐
│  EVIDENCE CACHE   │  doc_id → {text, embedding, score, source_agent}
│  (in-memory dict) │  + LMCache KV prefix sharing via vLLM
└────────┬──────────┘
         │
    ┌────┘  Wave 2: dependent sub-Qs resolve placeholders
    ▼
┌───────┐
│Agent 3│  Bridge hop-2 agent (uses hop-1 answer)
│(ReAct)│  Reads cache before searching
└───┬───┘
    │
    ▼
┌─────────────┐
│ AGGREGATOR  │  3-phase: evidence assembly → CoT synthesis → self-verify
│ (1-2 calls) │  Structured prompt targeting synthesis failures
└──────┬──────┘
       │
       ▼
  Final Answer
```

---

## Components to Build

### 1. Types (`src/multi_agent/types.py`)

```python
@dataclass
class SubQuestion:
    index: int
    text: str
    search_hints: list[str]      # entity names for seeding search
    depends_on: list[int]        # prerequisite sub-question indices
    placeholder: str | None      # "[answer_0]" for bridge dependencies

@dataclass
class DecompositionPlan:
    question_type: Literal["comparison", "bridge", "single_hop"]
    sub_questions: list[SubQuestion]
    dependency_edges: list[tuple[int, int]]

@dataclass
class AgentResult:
    sub_question_index: int
    answer: str
    evidence_doc_ids: list[str]
    trajectory: list[dict]
    loops: int
    total_tokens: int

@dataclass
class CachedDocument:
    doc_id: str
    text: str
    embedding: np.ndarray | None
    source_agent: int
    retrieval_score: float

@dataclass
class PipelineResult:
    question: str
    decomposition: DecompositionPlan
    agent_results: dict[int, AgentResult]
    final_answer: str
    cache_analytics: dict
    total_tokens: int
    wall_clock_seconds: float
```

### 2. Decomposer (`src/multi_agent/decomposer.py`)

Single LLM call with structured JSON output. Uses `/nothink` mode for Qwen3 to avoid unnecessary reasoning tokens.

**Prompt strategy**: System prompt defines the 3 question types + JSON schema. Zero-shot initially; add 3 few-shot examples (one per type) if quality is poor on pilot.

**Key decisions**:
- Classify question type FIRST (comparison/bridge/single_hop) — this determines dispatch strategy
- Use `[answer_N]` placeholders in bridge sub-questions
- Include `search_hints` (entity names) to seed retrieval
- Fallback: if JSON parsing fails after 2 retries, treat as single_hop (run original question through one agent)
- Single-hop questions bypass decomposition entirely (no overhead)

### 3. Dispatcher (`src/multi_agent/dispatcher.py`)

Dependency-aware async execution using topological sort → wave grouping → `asyncio.gather` per wave.

**Execution by type**:
- **Comparison**: Wave 1 runs all entity sub-Qs in parallel → Aggregator handles comparison
- **Bridge**: Wave 1 (hop-1), resolve placeholder, Wave 2 (hop-2), ... sequential waves
- **Single-hop**: 1 agent, 1 wave — equivalent to baseline

**Agent creation**: Each agent is `BaseAgent` (reused as-is from `src/arag/agent/base.py`) with:
- Scoped system prompt: focuses agent on its specific sub-question
- Reduced `max_loops=5` (sub-questions are simpler than full multi-hop Qs)
- Evidence cache write-through: `chunk_read` tool writes to shared cache
- Evidence cache read-through: agent checks cache before searching

### 4. Shared Evidence Cache (`src/multi_agent/evidence_cache.py`)

In-memory `dict[str, CachedDocument]` with `asyncio.Lock`. Lives per-question (created fresh, discarded after).

**Operations**:
- `put(doc)` → store; returns False if duplicate (hit logged)
- `get_relevant(query_embedding, top_k)` → cosine similarity over cached docs
- `get_all_evidence()` → sorted by score for aggregator
- `compute_analytics()` → hit rate, cross-agent reuse count, unique docs

**Cross-agent reuse flow**: Agent 1 retrieves chunk about "Einstein born in Ulm" → writes to cache → Agent 2 (searching for "population of [answer_0]") reads cache → finds Ulm mention → directly searches "population of Ulm" instead of re-discovering birthplace.

### 5. Aggregator (`src/multi_agent/aggregator.py`)

The most critical new component — targets the 58.8% synthesis failure.

**3-phase synthesis**:
1. **Evidence Assembly**: Deduplicate docs, group by sub-question, build structured evidence blocks
2. **Chain-of-Thought Synthesis**: Structured prompt with 4 steps:
   - Verify each sub-answer against its evidence
   - Check for contradictions between sub-answers
   - Chain verified sub-answers to answer original question
   - State final answer as concise phrase
3. **Self-Verification** (optional, for ablation): Second LLM call checks answer against evidence

**Why this beats single-agent synthesis**:
- Clean context (structured evidence blocks vs. noisy ReAct trajectory)
- Explicit verification step
- No "lost in the middle" effect (evidence adjacent to sub-questions)
- Fresh attention (no serial reasoning decay)

### 6. Pipeline Orchestrator (`src/multi_agent/pipeline.py`)

End-to-end: `question → decompose → dispatch → aggregate → result`

Wraps everything with timing, token counting, and error handling. Exports `PipelineResult` with full provenance for evaluation.

### 7. Modified Tools

**`chunk_read.py`**: Add optional `evidence_cache` parameter. After reading a chunk, write it to the shared cache. Backwards-compatible (cache=None → original behavior).

**`search_agent.py`** (new wrapper around `BaseAgent`): Before issuing search, check cache for relevant docs. Inject cached evidence into agent context as "Previously retrieved by another agent: ...".

### 8. Multi-Agent Batch Runner (`scripts/multi_agent_runner.py`)

Async-based runner replacing `ThreadPoolExecutor` for the multi-agent path. Uses `asyncio.Semaphore` to limit concurrent vLLM requests (prevent OOM). Checkpoint resume same as existing `batch_runner.py`.

---

## LMCache Integration

### Integration Points (ordered by impact)

**1. Shared System Prompt Prefix (automatic with vLLM)**
All search agents share identical system prompt (~1000 tokens). vLLM's `--enable-prefix-caching` reuses KV for this prefix across all requests. Free performance.

**2. Cross-Agent Evidence KV Reuse (via LMCache)**
When Agent 1 processes chunk text and Agent 2 later includes the same chunk, LMCache detects matching token sequences via content-addressed hashing and reuses KV blocks.

```python
# vLLM server launch with LMCache
lmcache_config = LMCacheEngineConfig(
    chunk_size=256,
    local_cpu=True,
    max_local_cpu_size=10.0,  # 10GB CPU buffer
    enable_blending=False,     # Start without, enable if hit rate is high
    cache_policy="LRU"
)
```

**3. Aggregator Evidence Reuse**
Aggregator prompt contains evidence passages already processed by search agents. Their KV caches may still be in LMCache → aggregator gets TTFT speedup.

**4. Batch-Level Cross-Question Reuse**
Consecutive questions often retrieve overlapping documents. LRU keeps recent chunk KV caches.

### Measurement
- KV cache hit rate (tokens served from cache / total tokens)
- TTFT with vs. without LMCache
- GPU memory peak comparison

---

## Experiment Matrix

### Primary Experiments

| ID | Config | Changes From | Tests | Hypothesis |
|---|---|---|---|---|
| **E4** | Single-agent A-RAG (existing) | — | Baseline | Mean 53.7% (known) |
| **M1** | Multi-agent, no cache | +decomposer +dispatcher +aggregator | Architecture value | M1 > E4 by 3-5pp |
| **M2** | Multi-agent + doc cache | +evidence_cache on M1 | Evidence sharing | M2 > M1 by 1-2pp |
| **M3** | Multi-agent + doc + KV cache | +LMCache on M2 | Inference efficiency | M3 ≈ M2 accuracy, 20-30% faster |
| **M4** | Single-agent 2× iterations | E4 with max_loops=6 | Equal-compute control | M1 > M4 (architecture > compute) |

### Ablation Experiments

| ID | Config | Isolates |
|---|---|---|
| **A1** | M1 without decomposer (full Q to each agent) | Decomposition value |
| **A2** | M1 without aggregator (use last agent's answer) | Aggregation value |
| **A3** | M1 sequential dispatch (no parallelism) | Parallelism value (latency) |
| **A4** | M2 without aggregator self-verify phase | Verification value |

### Scaling Experiments (if time permits)

| ID | Config | Tests |
|---|---|---|
| **S1** | M1 with Qwen3-8B (smaller model) | Does multi-agent compensate for model size? |
| **S2** | M1 with oracle decomposition (MuSiQue gold sub-Qs) | Upper bound on decomposition quality |

### Metrics Per Experiment

| Category | Metrics |
|---|---|
| **Accuracy** | LLM-Accuracy (DeepSeek-R1 judge), EM, Token F1 |
| **Per-hop** | Hop-k Recall (MuSiQue only, using gold paragraphs) |
| **Efficiency** | Tokens/question, tokens/correct answer, wall-clock latency (p50, p90) |
| **Cache** | Doc cache hit rate, KV cache hit rate, cross-agent reuse count |
| **Decomposition** | Sub-Q count distribution, DAG correctness (vs MuSiQue gold) |
| **Failure** | Extended taxonomy: DE (decomposition error), RM-k (retrieval miss hop-k), ASF (aggregation synthesis failure), AH (aggregation hallucination) |

### Statistical Rigor
- 1000 questions/dataset × 3 datasets = 3000 total (sufficient for 3pp effect at p<0.05)
- 3 seeds per config (capture inference non-determinism)
- Paired bootstrap test (10k resamples) for pairwise accuracy comparisons
- 95% BCa confidence intervals on deltas
- Compute budget equalization: M1-M3 total tokens capped at 2× E4 median

---

## File Structure (new/modified files only)

```
02-arag-multi-agent/
├── src/
│   ├── arag/
│   │   ├── agent/base.py              # EXISTING (reuse as-is)
│   │   ├── tools/chunk_read.py        # MODIFY: add cache write-through
│   │   └── ...                        # EXISTING (no changes)
│   │
│   └── multi_agent/                   # NEW package
│       ├── __init__.py
│       ├── types.py                   # Dataclasses
│       ├── decomposer.py             # Question decomposition
│       ├── dispatcher.py             # Dependency-aware async dispatch
│       ├── evidence_cache.py         # Shared doc cache
│       ├── aggregator.py             # 3-phase synthesis
│       ├── search_agent.py           # BaseAgent wrapper with cache integration
│       ├── pipeline.py               # End-to-end orchestrator
│       └── prompts/
│           ├── decomposer.txt        # Decomposer system prompt
│           ├── search_agent.txt      # Scoped search agent prompt
│           └── aggregator.txt        # Structured synthesis prompt
│
├── scripts/
│   ├── multi_agent_runner.py          # NEW: async batch runner
│   ├── run_experiment.py              # NEW: CLI entry point for experiment matrix
│   └── analyze_results.py            # NEW: tables, figures, failure analysis
│
├── configs/
│   ├── base.yaml                      # NEW: shared defaults
│   ├── m1_multi_agent.yaml
│   ├── m2_doc_cache.yaml
│   ├── m3_kv_cache.yaml
│   ├── m4_single_2x.yaml
│   └── ablations/                     # A1-A4 configs
│
├── jobs/                              # SLURM .job files
│   ├── vllm_server.job               # vLLM + optional LMCache
│   ├── m1_hotpotqa.job               # One job per (experiment, dataset)
│   ├── m1_musique.job
│   ├── m1_2wiki.job
│   ├── m2_hotpotqa.job
│   └── ...
│
└── results/
    └── {experiment}/{dataset}/
        ├── predictions.jsonl
        ├── decompositions.jsonl
        ├── cache_analytics.json
        └── eval_summary.json
```

---

## Implementation Sequence

### Phase 1: Core Pipeline (Days 1-4)
1. **Day 1**: `types.py` + `decomposer.py` + test on 50 questions per dataset (validate JSON output quality)
2. **Day 2**: `search_agent.py` (BaseAgent wrapper) + `dispatcher.py` (topological sort + asyncio.gather)
3. **Day 3**: `evidence_cache.py` + `aggregator.py` (3-phase synthesis)
4. **Day 4**: `pipeline.py` + `multi_agent_runner.py` + end-to-end test on 100 questions

### Phase 2: Experiments M1 + M4 (Days 5-7)
5. **Day 5**: Create configs + SLURM jobs, run M1 on all 3 datasets (parallel jobs)
6. **Day 6**: Run M4 (single-agent 2×), evaluate M1 with DeepSeek-R1 judge
7. **Day 7**: If M1 < E4 → diagnose (test decomposer + aggregator in isolation), iterate prompts

### Phase 3: Cache + LMCache (Days 8-10)
8. **Day 8**: Wire evidence cache into chunk_read + search_agent, run M2
9. **Day 9**: Configure LMCache + vLLM, verify prefix caching works, run M3
10. **Day 10**: Evaluate M2/M3, compute cache metrics

### Phase 4: Ablations + Analysis (Days 11-14)
11. **Day 11-12**: Run A1-A4 ablations
12. **Day 13**: Run S1 (8B model) and S2 (oracle decomposition) if time permits
13. **Day 14**: Failure taxonomy, statistical tests, generate figures

---

## Key Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Decomposer JSON parsing fails with Qwen3-30B-A3B | Medium | High | Retry 2×, fallback to single-agent; validate on 50-Q pilot first |
| Multi-agent overhead wipes out accuracy gains (M1 ≤ E4) | Medium | High | 100-Q pilot before full run; test aggregator with gold sub-answers to isolate component |
| vLLM OOM with concurrent agents (2-3 parallel requests) | Medium | Medium | Set `max_num_seqs` limit; profile with 1/2/3 concurrent; fallback to sequential |
| LMCache + vLLM integration unstable | Medium-High | Low | LMCache is secondary; fall back to vLLM built-in `--enable-prefix-caching`; M1/M2 are primary contributions |
| Bridge questions show no improvement (serial bottleneck) | Medium | Medium | Aggregator still helps synthesis; report results stratified by question type |

---

## Verification Plan

1. **Unit tests**: Decomposer JSON parsing, dispatcher wave ordering, cache put/get/analytics
2. **Integration test**: 10 questions end-to-end, verify PipelineResult has all fields
3. **Pilot run**: 100 questions per dataset on M1, check M1 > E4 before committing to full runs
4. **Full evaluation**: DeepSeek-R1 judge on all predictions, failure taxonomy, statistical tests
5. **Cache verification**: Log cache hit/miss events, verify cross-agent reuse on comparison questions
6. **LMCache verification**: Compare TTFT with/without LMCache on same questions, verify accuracy is identical (bitwise)

---

## Publication Positioning

**Title direction**: "MA²RAG: Multi-Agent Agentic Retrieval-Augmented Generation with Shared Evidence Caching for Multi-Hop Question Answering"

**Novel contributions**:
1. Failure-taxonomy-driven architecture: each component targets a specific, quantified failure mode
2. Dependency-aware sub-question dispatch with parallel execution
3. Structured aggregation with self-verification targeting the synthesis bottleneck
4. LMCache KV sharing for multi-agent inference efficiency

**Key figures for paper**:
- Architecture diagram (Figure 1)
- Accuracy vs. compute Pareto frontier: multi-agent curve dominates single-agent (Figure 2)
- Per-hop recall heatmap on MuSiQue (Figure 3)
- Failure migration Sankey: E4 failures → M2 outcomes (Figure 4)
- Latency CDF: M2 vs M3 showing KV cache speedup (Figure 5)
