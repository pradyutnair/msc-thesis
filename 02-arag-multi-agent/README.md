# MA²RAG — Multi-Agent Decomposed Retrieval, Holistic Reasoning

> **MSc Thesis Experiments** · University of Amsterdam · MultIX Lab · Spring 2026  
> Built on top of [A-RAG](https://arxiv.org/abs/2602.03442) (Du et al., 2026)

This repository contains the full experimental pipeline for **MA²RAG**: a multi-agent extension of agentic RAG that decomposes complex questions across specialised retrieval agents, then synthesises a final answer over the pooled evidence.

---

## Architecture Overview

### Baseline: E4 — Single-Agent A-RAG

```mermaid
flowchart LR
    Q([Question]) --> A

    subgraph A["Search Agent (ReAct loop)"]
        direction TB
        T1[keyword_search]
        T2[semantic_search]
        T3[read_chunk]
    end

    A <-->|iterative tool calls| C[(Wikipedia\nCorpus)]
    A --> Ans([Answer])
```

---

### M1 — Sub-Answer Aggregation *(deprecated)*

Agents answer their sub-questions; the aggregator reconciles compressed sub-answers.

```mermaid
flowchart TD
    Q([Question]) --> D[Decomposer]

    D -->|SQ-0: bridge entity?| A0
    D -->|SQ-1: final answer?| A1

    subgraph A0["Agent 0 (ReAct)"]
        direction LR
        S0[search] --> R0[read] --> F0[finish + answer]
    end

    subgraph A1["Agent 1 (ReAct)"]
        direction LR
        S1[search] --> R1[read] --> F1[finish + answer]
    end

    A0 -->|"sub-answer₀ + chunks"| Agg
    A1 -->|"sub-answer₁ + chunks"| Agg

    subgraph Agg["Aggregator (M1)"]
        direction TB
        SA["Sub-Answer Summary\n(compressed intermediaries)"]
        Pool["Unified Evidence Pool"]
        SA --> Syn
        Pool --> Syn
        Syn[Synthesis LLM call]
    end

    Agg --> Ans([Answer])

    style SA fill:#ffcccc,stroke:#cc0000
    style Agg fill:#fff3e0
```

**Problem**: Compressed sub-answers lose information. If Agent 0 answers incorrectly, the aggregator inherits that error.

---

### M2 — DRHR: Decomposed Retrieval, Holistic Reasoning ✅

Agents are **pure retrieval workers** — their sub-answers are ignored. The synthesiser reasons directly over the raw evidence pool.

```mermaid
flowchart TD
    Q([Question]) --> D[Decomposer]

    D -->|SQ-0| A0
    D -->|SQ-1| A1

    subgraph A0["Agent 0 — Retrieval Worker"]
        direction LR
        S0[search] --> R0[read chunks] --> F0[finish ·  signal only]
    end

    subgraph A1["Agent 1 — Retrieval Worker"]
        direction LR
        S1[search] --> R1[read chunks] --> F1[finish · signal only]
    end

    A0 -->|"chunks only  ✓"| Pool
    A1 -->|"chunks only  ✓"| Pool

    subgraph Pool["Structured Evidence Pool"]
        direction TB
        Comp["comparison  →  per-entity sections\n[ENTITY SECTION SQ-0] … [ENTITY SECTION SQ-1]"]
        Bridge["bridge  →  flat pool ordered by chain step\n[Doc | SQ-0] … [Doc | SQ-1]"]
    end

    Pool --> Syn["Holistic Synthesiser\n(single LLM call, max_tokens=512)"]
    Syn --> Ans([Answer])

    style Pool fill:#e8f5e9,stroke:#388e3c
    style Syn fill:#e3f2fd,stroke:#1565c0
```

**Key insight**: Decomposition guides *retrieval*, not *reasoning*. Holistic synthesis over structured raw evidence avoids the information compression bottleneck.

---

## Dispatch & Wave Strategy

For **bridge** questions, agents run in sequential waves so each wave can use prior evidence. For **comparison** questions, agents run in parallel (one per entity).

```mermaid
flowchart LR
    subgraph Bridge["Bridge Question (sequential waves)"]
        direction TB
        W1["Wave 1 · Agent 0\nfinds bridge entity"] --> W2["Wave 2 · Agent 1\nfinds final answer"]
    end

    subgraph Comp["Comparison Question (parallel)"]
        direction LR
        C0["Agent 0 · Entity A"] & C1["Agent 1 · Entity B"]
    end

    subgraph Single["Single-hop (bypass)"]
        direction LR
        AG0["Agent 0 · direct answer\n(no synthesis call)"]
    end
```

---

## Results

### LLM Accuracy (%) — Qwen3-30B-A3B + E5-base-v2 + DeepSeek-R1 judge

```mermaid
xychart-beta
    title "HotpotQA — LLM Accuracy (%)"
    x-axis ["E4 (single)", "M1", "M1v3", "M1v5", "M1v8", "M2 DRHR"]
    y-axis "Accuracy (%)" 0 --> 75
    bar [66.5, 44.7, 49.9, 55.1, 54.9, 63.4]
```

```mermaid
xychart-beta
    title "2WikiMultiHop — LLM Accuracy (%)"
    x-axis ["E4 (single)", "M1", "M1v3", "M1v5", "M1v8", "M2 DRHR"]
    y-axis "Accuracy (%)" 0 --> 65
    bar [56.9, 25.4, 28.2, 32.1, 32.3, 34.2]
```

```mermaid
xychart-beta
    title "MuSiQue — LLM Accuracy (%)"
    x-axis ["E4 (single)", "M1", "M1v3", "M1v5", "M1v8", "M2 DRHR"]
    y-axis "Accuracy (%)" 0 --> 45
    bar [37.6, 24.9, 27.4, 29.6, 30.3, 27.2]
```

### Full Results Table

| Version | HotpotQA | 2WikiMH | MuSiQue | **Mean** | Key Change |
|---------|:--------:|:-------:|:-------:|:--------:|------------|
| E4 *(single-agent baseline)* | 66.5 | 56.9 | 37.6 | **53.7** | Single ReAct agent, full iterative search |
| M1 | 44.7 | 25.4 | 24.9 | 31.7 | Multi-agent baseline; sub-answer aggregation |
| M1v3 | 49.9 | 28.2 | 27.4 | 35.2 | Full 1000-sample runs |
| M1v5 | 55.1 | 32.1 | 29.6 | 38.9 | Fix `finish()` leak; decisive aggregator; single-hop bypass |
| M1v6 | 55.9 | 27.7 | 30.3 | 38.0 | Unified evidence pool (regression on 2Wiki) |
| M1v7 | 51.5 | 23.6 | 29.6 | 34.9 | Per-entity sections (broken comparison instruction) |
| M1v8 | 54.9 | 32.3 | 30.3 | 39.2 | Fixed comparison; disabled self-verify |
| **M2 DRHR** | **63.4** | **34.2** | **27.2** | **41.6** | Holistic synthesis over raw evidence pool |

### M2 DRHR — Accuracy by Question Type

| Dataset | bridge | comparison | single_hop | n |
|---------|:------:|:----------:|:----------:|---|
| HotpotQA | 63.4% | 64.3% | 60.6% | 1000 |
| 2WikiMultiHop | **21.0%** | 48.6% | 22.2% | 1000 |
| MuSiQue | 27.4% | 33.3% | 19.2% | 1000 |

**Key finding**: 2Wiki bridge accuracy (21.0%) is the dominant failure mode — bridge agents retrieve without knowledge of what earlier agents found.

### Gaps to E4 Closed by M2

```mermaid
xychart-beta
    title "Gap to E4 Remaining After M1v8 vs M2 DRHR (pp)"
    x-axis ["HotpotQA", "2WikiMH", "MuSiQue"]
    y-axis "Gap to E4 (pp)" 0 --> 30
    bar [11.6, 24.6, 7.3]
    line [3.1, 22.7, 10.4]
```
*Bars = M1v8 gap to E4; Line = M2 DRHR gap to E4*

---

## M1 → M2: What Changed

| Component | M1 | M2 DRHR |
|-----------|----|----|
| Agent role | Answer sub-question | Retrieve evidence only |
| `finish()` answer | Used by aggregator | Ignored (fallback only) |
| Aggregator input | Sub-answer summary + chunks | Chunks only |
| Synthesis calls | 1 main + optional self-verify | 1 holistic call |
| Self-verify | Enabled (comparison) | Removed |
| Comparison pool | Flat merged | Per-entity labeled sections |
| Token budget | 6k (shared with sub-answers) | 5.5k tiktoken (evidence only) |
| Synthesis `max_tokens` | 4096 (config default) | 512 (short answers only) |

---

## Known Issues & Fixes Applied

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `VLLMValidationError` in synthesis | tiktoken underestimates Qwen token count ~2.3× | `max_tokens=512` in synthesis call |
| `VLLMValidationError` in search agents | Agent accumulates large read_chunk history | `_safe_max_tokens()` in `base.py` — dynamic cap using 2.5× ratio |
| `FINAL ANSWER:` prefix in 11% bridge/comparison predictions | LLM outputs prefix on separate line; `_extract_final_answer` regex edge case | Defensive strip in `aggregate()` |
| `finish()` written as plain text (~40% of calls) | Qwen3 instruction-following quirk | 5-pattern regex in `base.py` `_extract_finish_answer()` |
| Stale predictions skipped on re-run | Runner checkpoint logic | Separate output dirs per version (`results/m2_drhr/`) |

---

## Project Structure

```
02-arag-multi-agent/
├── src/
│   ├── arag/
│   │   ├── agent/base.py          # ReAct loop + _safe_max_tokens fix
│   │   ├── core/llm.py            # OpenAI-compatible LLM client
│   │   └── tools/                 # keyword_search, semantic_search, read_chunk
│   └── multi_agent/
│       ├── aggregator.py          # M2 DRHR holistic synthesiser
│       ├── decomposer.py          # Question type + sub-question generation
│       ├── dispatcher.py          # Wave-based agent dispatch
│       ├── search_agent.py        # Retrieval worker wrapper
│       ├── evidence_cache.py      # Cross-agent dedup cache
│       └── prompts/
│           ├── aggregator.txt     # M2: no sub-answer section
│           ├── search_agent.txt   # M2: retrieval worker framing
│           └── decomposer.txt
├── configs/
│   ├── m2_drhr.yaml               # Current best config
│   └── m1v8.yaml                  # Best M1 config
├── jobs/                          # SLURM job files (Snellius H100)
├── experiments/
│   └── m1_results_summary.md      # Full results log
└── results/
    ├── m2_drhr/{hotpotqa,2wikimultihop,musique}/
    └── m1v8/{hotpotqa,2wikimultihop,musique}/
```

---

## Running Experiments

```bash
# On Snellius — submit all three datasets in parallel
cd /projects/prjs1800/msc-thesis/02-arag-multi-agent
for DATASET in hotpotqa 2wikimultihop musique; do
  RID=$(sbatch --parsable jobs/m2_drhr_${DATASET}.job)
  EID=$(sbatch --parsable --dependency=afterok:${RID} jobs/eval_m2_drhr_${DATASET}.job)
  echo "${DATASET}: runner=${RID}, eval=${EID}"
done
```

---

## Citation

```bibtex
@misc{nair2026ma2rag,
  title  = {MA²RAG: Multi-Agent Decomposed Retrieval with Holistic Reasoning},
  author = {Pradyut Nair},
  year   = {2026},
  note   = {MSc Thesis, University of Amsterdam}
}
```

Built on [A-RAG](https://arxiv.org/abs/2602.03442) by Du et al. (2026).
