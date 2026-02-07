# Day 4: Iterative Retrieval — IRCoT & FLARE

## Overview

Day 4 evaluates whether iterative retrieval strategies can overcome the retrieval ceiling
observed in Days 1-3. We test two principled approaches:

1. **IRCoT** (Trivedi et al., ACL 2023): Interleaves retrieval with chain-of-thought reasoning.
   The model generates intermediate thoughts that serve as reformulated queries for subsequent
   retrieval rounds (up to 5 iterations).

2. **FLARE** (Jiang et al., EMNLP 2023): Forward-Looking Active REtrieval. Generates text
   token-by-token and triggers retrieval only when token-level confidence drops below a
   threshold (0.2). Reactive/adaptive retrieval.

**Setup**: Qwen2.5-7B-Instruct (vLLM), E5-base-v2 retriever, FAISS flat index (21M wiki18 passages).
Evaluated on full HotpotQA (7,405) and MuSiQue (2,417) dev sets.

---

## Master Comparison Table (Days 1-4)

| Day | Method                    | HQA EM  | HQA F1  | MSQ EM  | MSQ F1  |
|-----|---------------------------|---------|---------|---------|---------|
| 1   | Standard RAG (top-5)      | 0.3164  | 0.4201  | 0.0633  | 0.1303  |
| 2   | + BGE Reranker (top-5)    | 0.3641  | 0.4742  | 0.0770  | 0.1552  |
| 3   | + RECOMP Refiner          | 0.2955  | 0.4002  | 0.0550  | 0.1185  |
| 3   | + SelectiveContext        | 0.2700  | 0.3656  | 0.0501  | 0.1126  |
| **4** | **IRCoT (5 iter)**      | **0.3064** | **0.4246** | **0.0724** | **0.1429** |
| **4** | **FLARE (theta=0.2)**   | **0.1885** | **0.2657** | **0.0393** | **0.1144** |

### Key Observations
- **IRCoT is approximately equal to Day 1 Standard RAG** on HotpotQA (F1: 0.4246 vs 0.4201, delta=+0.45 pp) — negligible gain
- **IRCoT slightly improves** MuSiQue over Day 1 (F1: 0.1429 vs 0.1303, delta=+1.26 pp) — modest gain
- **Both iterative methods underperform Day 2 Reranker** (HQA F1: 0.4742, MSQ F1: 0.1552)
- **FLARE is significantly worse** than all baselines — essentially broken retrieval
- **Day 2 Reranker remains the best configuration** across all 4 days

---

## IRCoT Detailed Analysis

### Iteration Behavior

**HotpotQA:**

| Iterations | Count | Percentage |
|-----------|-------|------------|
| 1         | 1     | 0.0%       |
| 2         | 2,024 | 27.3%      |
| 3         | 3,465 | 46.8%      |
| 4         | 1,407 | 19.0%      |
| 5 (max)   | 508   | 6.9%       |

Average: 3.05 iterations, 11.9 accumulated docs per item.

**MuSiQue:**

| Iterations | Count | Percentage |
|-----------|-------|------------|
| 2         | 549   | 22.7%      |
| 3         | 1,123 | 46.5%      |
| 4         | 491   | 20.3%      |
| 5 (max)   | 254   | 10.5%      |

Average: 3.19 iterations, 13.6 accumulated docs per item.

### Retrieval Recall Progression (IRCoT)

**HotpotQA — Per-round recall (documents retrieved in THAT specific round):**

| Round | n examples | Avg Recall |
|-------|-----------|------------|
| 0     | 7,405     | 50.0%      |
| 1     | 7,404     | 44.0%      |
| 2     | 5,380     | 31.6%      |
| 3     | 1,915     | 30.1%      |
| 4     | 508       | 20.4%      |

**HotpotQA — Accumulated recall (all docs through round N):**

| Round | n examples | Avg Recall | Full Recall (100%) |
|-------|-----------|------------|-------------------|
| 0     | 7,405     | 50.0%      | 23.9%             |
| 1     | 7,404     | 58.8%      | 37.6%             |
| 2     | 5,380     | 67.1%      | 50.2%             |
| 3     | 1,915     | 69.6%      | 53.1%             |
| 4     | 508       | 65.3%      | 46.9%             |

**Critical insight**: Accumulated recall improves from 50.0% to 64.7% overall, and full
recall from 23.9% to 47.3%. But later rounds retrieve *less* relevant docs (diminishing returns).
Items that hit max iterations (round 4-5) actually have *lower* accumulated recall than
those that terminate early — these are the hardest cases.

**MuSiQue — Accumulated recall:**

| Round | n examples | Avg Recall | Full Recall |
|-------|-----------|------------|-------------|
| 0     | 2,417     | 21.4%      | 3.3%        |
| 1     | 2,417     | 28.4%      | 7.5%        |
| 2     | 1,868     | 33.6%      | 11.8%       |
| 3     | 745       | 32.8%      | 9.1%        |
| 4     | 254       | 33.1%      | 7.5%        |

**MuSiQue per-hop recall (accumulated):**

| Hop | Found/Total | Recall |
|-----|------------|--------|
| 1   | 1,118/2,417 | 46.3%  |
| 2   | 601/2,417   | 24.9%  |
| 3   | 176/1,165   | 15.1%  |
| 4   | 29/405      | 7.2%   |

Multi-hop retrieval remains the fundamental bottleneck. Even with iterative retrieval,
later hops are rarely found because:
1. The CoT thoughts are not specific enough to find later-hop evidence
2. The retriever (E5-base-v2) struggles with reformulated queries
3. Error compounds across hops

### Answer Quality by Retrieval Success (IRCoT HotpotQA)

| Retrieval Bucket | n items | Avg F1 |
|-----------------|---------|--------|
| recall = 0       | 1,327   | 0.1024 |
| 0 < recall < 1   | 2,576   | 0.3295 |
| recall = 1       | 3,502   | 0.6166 |

This confirms the strong correlation between retrieval recall and answer quality.
When IRCoT successfully retrieves all evidence, F1 reaches 0.6166 — much higher than
the overall average. The bottleneck is retrieval, not reasoning.

### Answer Quality by Retrieval Success (IRCoT MuSiQue)

| Retrieval Bucket     | n items | Avg F1 |
|---------------------|---------|--------|
| recall = 0           | 971     | 0.0669 |
| 0 < recall < 0.5     | 439     | 0.0887 |
| 0.5 <= recall < 1    | 720     | 0.1732 |
| recall = 1           | 287     | 0.4069 |

Same pattern: full retrieval recall leads to 4069 F1, vs 0.0669 with zero recall.

### Error Categorization (IRCoT HotpotQA)

| Category              | Count | % of Total |
|----------------------|-------|------------|
| Correct (EM=1)       | 2,269 | 30.6%      |
| Reasoning failure     | 1,887 | 25.5%      |
| Partial retrieval     | 1,999 | 27.0%      |
| Total retrieval miss  | 1,250 | 16.9%      |

- 25.5% of items have full retrieval recall but wrong answers (reasoning failure)
- 43.9% have incomplete retrieval (partial + total miss)

### Error Categorization (IRCoT MuSiQue)

| Category              | Count | % of Total |
|----------------------|-------|------------|
| Correct (EM=1)       | 175   | 7.2%       |
| Reasoning failure     | 210   | 8.7%       |
| Partial retrieval     | 1,081 | 44.7%      |
| Total retrieval miss  | 951   | 39.3%      |

MuSiQue is dominated by retrieval failures (84.0% partial + total miss).

### F1 by Question Type (IRCoT HotpotQA)

| Type       | n items | Avg F1 |
|------------|---------|--------|
| Bridge     | 5,918   | 0.3876 |
| Comparison | 1,487   | 0.5717 |

Comparison questions significantly easier — they require matching entities from
two separate paragraphs but have more formulaic answer patterns.

### Item-Level Comparison: IRCoT vs Day 1 Standard RAG

**HotpotQA (n=7,405):**
- Improved: 1,246 (16.8%) — avg improvement: +0.6028 F1
- Degraded: 1,260 (17.0%) — avg degradation: -0.5698 F1
- Unchanged: 4,899 (66.2%)
- **Net: -14 items** (nearly zero-sum)

**MuSiQue (n=2,417):**
- Improved: 343 (14.2%) — avg improvement: +0.4367 F1
- Degraded: 307 (12.7%) — avg degradation: -0.3892 F1
- Unchanged: 1,767 (73.1%)
- **Net: +36 items** (slight positive)

**Interpretation**: IRCoT is a *sidegrade* on HotpotQA. It improves and degrades roughly
equal numbers of items. The iterative retrieval helps some questions but the additional
context from multiple rounds confuses the model on others (context dilution).

---

## FLARE Analysis — Critical Failure Mode

### Retrieval Triggering

FLARE uses token-level confidence to decide when to retrieve. With Qwen2.5-7B-Instruct
and threshold=0.2:

**HotpotQA:**

| Iteration | Model Confident (No Retrieve) | Percentage |
|-----------|------------------------------|------------|
| 0         | 6,358 / 7,405                | 85.9%      |
| 1         | 7,342 / 7,405                | 99.1%      |
| 2         | 7,384 / 7,405                | 99.7%      |
| 3         | 7,394 / 7,405                | 99.9%      |
| 4         | 7,395 / 7,405                | 99.9%      |

**Only 14.4% of items (1,068/7,405) ever trigger ANY retrieval.**

**MuSiQue:**

| Iteration | Model Confident (No Retrieve) | Percentage |
|-----------|------------------------------|------------|
| 0         | 1,996 / 2,417                | 82.6%      |
| 1         | 2,400 / 2,417                | 99.3%      |
| 2         | 2,411 / 2,417                | 99.8%      |
| 3         | 2,414 / 2,417                | 99.9%      |
| 4         | 2,416 / 2,417                | 100.0%     |

**Only 17.7% of items (429/2,417) ever trigger retrieval.**

### Why FLARE Fails: LLM Overconfidence

FLARE's fundamental assumption is that low token-level confidence correlates with
knowledge gaps that retrieval can fill. This assumption **breaks down** with
instruction-tuned LLMs like Qwen2.5-7B:

1. **Overconfident generation**: The model produces tokens with high probability even when
   generating incorrect content. Instruction tuning teaches models to generate fluent,
   confident-sounding text regardless of factual accuracy.

2. **Threshold too low**: With theta=0.2, retrieval only triggers when token probability drops
   below 20%. But Qwen2.5-7B rarely drops this low — it is either correct OR confidently wrong.

3. **Parametric knowledge dominance**: FLARE HotpotQA achieves EM=0.1885 with 99% zero
   retrieval recall — the model is answering purely from parametric knowledge. This explains
   why comparison questions (F1=0.6086) vastly outperform bridge questions (F1=0.1796):
   entity comparisons are often memorized, while bridge reasoning requires external evidence.

### Retrieval Recall (FLARE)

| Dataset    | Avg Recall | Full Recall | Zero Recall |
|------------|-----------|-------------|-------------|
| HotpotQA   | 0.6%      | 0.1%        | 99.0%       |
| MuSiQue    | 0.4%      | 0.0%        | 99.1%       |

### Error Categories (FLARE HotpotQA)

| Category              | Count | Percentage |
|----------------------|-------|------------|
| Total retrieval miss  | 5,953 | 80.4%      |
| Correct (EM=1)       | 1,396 | 18.9%      |
| Partial retrieval     | 51    | 0.7%       |
| Reasoning failure     | 5     | 0.1%       |

18.9% correct answers from purely parametric knowledge. This provides a useful **parametric
knowledge baseline** — the model alone (without retrieval) achieves approximately 18.9% EM on HotpotQA.

### Error Categories (FLARE MuSiQue)

| Category              | Count | Percentage |
|----------------------|-------|------------|
| Total retrieval miss  | 2,301 | 95.2%      |
| Correct (EM=1)       | 95    | 3.9%       |
| Partial retrieval     | 21    | 0.9%       |

---

## Thesis Implications

### 1. Iterative Retrieval Does Not Scale with Qwen2.5-7B

IRCoT improves accumulated retrieval recall (50% to 65% on HotpotQA) but this does NOT
translate to proportional answer improvement. The CoT-generated queries are not precise
enough with a 7B model — a larger LLM might generate better intermediate thoughts.

### 2. FLARE's Confidence-Based Approach is Incompatible with Instruction-Tuned LLMs

This is a key negative result. FLARE was designed for completion-style models where
token probability correlates with factual uncertainty. Instruction-tuned models generate
fluent text with high confidence regardless of factual accuracy, rendering FLARE's
retrieval mechanism inert.

**For the thesis**: This motivates explicit retrieval scheduling (e.g., retrieve every N
tokens, or at decomposed sub-question boundaries) over confidence-based triggering.

### 3. The Retrieval Ceiling is the Binding Constraint

Across all 4 days of experiments:
- When retrieval recall = 100%, IRCoT achieves F1 = 0.6166 (HotpotQA)
- When retrieval recall = 0%, F1 drops to 0.1024
- The gap between "retrieval works" and "retrieval fails" is approximately 6x

The fundamental bottleneck is not the pipeline architecture (sequential, iterative, adaptive)
but the **retriever's ability to find relevant evidence given the query**. This motivates:
- Better query decomposition strategies (Day 5: reasoning pipeline)
- Retriever fine-tuning or hybrid retrieval
- Oracle retrieval experiments to establish upper bounds

### 4. Day 2 Reranker Remains Best

The BGE reranker (Day 2) provides the best cost-benefit ratio:
- Simple single-pass architecture
- +5.41 pp F1 on HotpotQA, +2.49 pp on MuSiQue over standard RAG
- No iterative overhead (3-5x faster than IRCoT)

### 5. Context Dilution is Real

IRCoT accumulates 11.9 docs on average (vs 5 for standard RAG). The extra context
does not help and may hurt — items that run maximum iterations have *lower* performance,
suggesting that irrelevant context actively degrades the reader's ability to extract answers.

---

## Technical Notes

### FAISS Single-Threaded BLAS Issue (Resolved)
The faiss-cpu v1.9.0 pip package ships with a statically linked OpenBLAS that runs
single-threaded. Fix: LD_PRELOAD=/usr/lib64/libopenblaso.so.0 overrides with the
system's multi-threaded OpenBLAS. **67x speedup** on flat index search.

### PYTHONUNBUFFERED for SLURM
Python fully buffers stdout when redirected (SLURM). PYTHONUNBUFFERED=1 ensures
real-time logging.

### Runtime

| Experiment | Runtime | Notes |
|-----------|---------|-------|
| IRCoT HQA | ~62 min | 7,405 items, avg 3.05 iterations |
| IRCoT MSQ | ~50 min | 2,417 items, avg 3.19 iterations |
| FLARE HQA | ~9 min  | Minimal retrieval; mostly parametric |
| FLARE MSQ | ~59 min | Minimal retrieval; mostly parametric |

---

## Files

### Configs
- configs/flashrag/ircot_qwen25_hotpotqa.yaml
- configs/flashrag/ircot_qwen25_musique.yaml
- configs/flashrag/flare_qwen25_hotpotqa.yaml
- configs/flashrag/flare_qwen25_musique.yaml

### Scripts
- scripts/run_ircot_rag.py — Custom IRCoT runner with per-round tracking
- scripts/run_flare_rag.py — FLARE runner wrapper
- scripts/analyze_ircot_retrieval.py — Comprehensive Day 4 analysis

### Results
- /projects/prjs1800/results/day4/hotpotqa_2026_02_07_00_20_ircot_qwen25_hotpotqa/
- /projects/prjs1800/results/day4/musique_2026_02_07_00_20_ircot_qwen25_musique/
- /projects/prjs1800/results/day4/hotpotqa_2026_02_06_22_00_flare_qwen25_hotpotqa/
- /projects/prjs1800/results/day4/musique_2026_02_07_00_10_flare_qwen25_musique/

### Analysis Outputs
- outputs/day4/day4_analysis_results.json
- outputs/day4/day4_summary.md (this file)
