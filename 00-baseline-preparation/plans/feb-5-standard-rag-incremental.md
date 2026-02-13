# Week Plan: Incremental RAG Component Analysis on FlashRAG
**Feb 6–12, 2026 | Snellius Supercomputer | Qwen2.5-7B-Instruct + E5-base-v2**

---

## Philosophy

**Don't build a multi-agent system yet. Understand what each FlashRAG component contributes first.**

The plan: Start from Standard RAG, add one component at a time, measure the delta, analyze failures at each stage. Each day's output feeds directly into "why do we need the next component?" — and ultimately, "why do we need multi-agent?"

No naive generation (waste of compute). No thesis writing (too early). Pure empirical investigation.

---

## Reference Numbers (from R3-RAG paper, Qwen2.5-7B + E5-base-v2)

These are published results to sanity-check your runs against:

| Method | Retriever | HotpotQA F1 | 2Wiki F1 | MuSiQue F1 |
|--------|-----------|-------------|----------|------------|
| CoT (no retrieval) | â€” | 34.0 | 31.1 | 12.7 |
| RAG with CoT | E5 | 52.4 | 33.5 | 16.9 |
| IRCoT | E5 | 48.4 | 35.8 | 13.5 |
| R3-RAG (RL-trained) | E5 | 65.5 | 62.3 | 33.6 |

### FlashRAG's Benchmarks (Table 3 - LLaMA-3-8B + E5-base-v2)

**CORRECTED from FlashRAG paper Table 3:**

| Method | Type | Pipeline | HotpotQA F1 | 2Wiki F1 | PopQA F1 | Notes |
|--------|------|----------|-------------|----------|----------|-------|
| Naive Generation | Sequential | - | 28.4 | 33.9 | 21.7 | No retrieval baseline |
| **Standard RAG** | Sequential | SequentialPipeline | **35.3** | 21.0 | 36.7 | Single retrieval, top-5 |
| RECOMP-abstractive | Refiner | SequentialPipeline | 37.5 | 32.4 | 39.9 | Abstractive compression |
| Spring | Generator | SequentialPipeline | 42.6 | 37.3 | 54.8 | Generator optimization |
| Adaptive-RAG | Conditional | ConditionalPipeline | 39.1 | 28.4 | 40.4 | Query complexity routing |
| **IRCoT** | Loop | LoopPipeline | **41.5** | 32.4 | 45.6 | Iterative retrieval+CoT |
| **FLARE** | Loop | FLAREPipeline | **28.0** | 33.9 | 20.7 | Active retrieval (note: same as naive!) |
| Iter-RetGen/ITRG | Loop | LoopPipeline | 38.3 | 21.6 | 37.9 | Iterative generation |
| Self-RAG* | Loop | SelfRAGPipeline | 29.6 | 25.1 | 32.7 | Trained generator needed |

*\*Methods marked with asterisk require trained generators*

---

## Actual Results from Days 1-5 (Qwen2.5-7B-Instruct + E5-base-v2)

### Master Comparison Table

| Day | Method | HotpotQA EM | HotpotQA F1 | MuSiQue EM | MuSiQue F1 | Retrieval Recall@5 (HQA) | Retrieval Recall@5 (MSQ) |
|-----|--------|-------------|-------------|------------|------------|-------------------------|-------------------------|
| 1 | Standard RAG (top-5) | 31.64% | 42.01% | 6.33% | 13.03% | 50.0% | 21.4% |
| 2 | **+ BGE Reranker (top-5)** | **36.41%** | **47.42%** | **7.70%** | **15.52%** | **57.7%** | **26.2%** |
| 3 | + RECOMP Refiner | 29.55% | 40.02% | 5.50% | 11.85% | - | - |
| 3 | + Selective-Context | 27.00% | 36.56% | 5.01% | 11.26% | - | - |
| 4 | IRCoT (5 iter) | 30.64% | 42.46% | 7.24% | 14.29% | 69.6% (accumulated) | - |
| 4 | FLARE (θ=0.2) | 18.85% | 26.57% | 3.93% | 11.44% | ~0.6% | - |
| 5 | + Reranker + CoT | 34.40% | 45.52% | 8.11% | 13.99% | 57.7% | 26.2% |
| 5 | ReasoningPipeline | 3.81% | 17.70% | 1.70% | 10.54% | - | - |
| 5 | SelfAsk (n=500) | 10.80% | 18.79% | 6.40% | 13.88% | - | - |
| 5b | Standard RAG + CoT | 30.45% | 40.44% | 6.41% | 11.62% | 50.0% | 21.4% |
| 5b | Reranker + CoT short (max_tokens=32) | 0.00% | 1.67% | 0.00% | 1.27% | 57.7% | 26.2% |

---

## Day 1 (Thu Feb 6): Standard RAG Baseline + Retrieval Quality Analysis ✅

### Goal
Get Standard RAG running on HotpotQA + MuSiQue. Critically: measure **retrieval quality** against ground-truth supporting documents, not just answer quality.

### Why this matters
MuSiQue and HotpotQA both have annotated supporting facts/paragraphs. This lets you compute retrieval Recall@k — i.e., "did the retriever find the documents the model actually needs?" This is the foundation of your entire argument.

### Tasks

1. **Verify FlashRAG installation on Snellius**
   ```bash
   cd /projects/prjs1800/external/FlashRAG
   source /projects/prjs1800/venvs/flashrag-venv/bin/activate
   pip install -e ".[full]" --break-system-packages
   ```

2. **Verify models and data are available**
   - Qwen2.5-7B-Instruct at `/projects/prjs1800/models/base/qwen2.5-7b-instruct`
   - E5-base-v2 at `/projects/prjs1800/models/base/e5-base-v2`
   - Pre-built wiki18_100w index (check FlashRAG ModelScope page or build with their script)
   - HotpotQA + MuSiQue datasets from `RUC-NLPIR/FlashRAG_datasets` on HuggingFace

3. **Run Standard RAG (SequentialPipeline)**
   ```python
   from flashrag.config import Config
   from flashrag.utils import get_dataset
   from flashrag.pipeline import SequentialPipeline
   from flashrag.prompt import PromptTemplate

   config = Config(config_file_path='configs/standard_rag.yaml')
   test_data = get_dataset(config)['test']
   pipeline = SequentialPipeline(config)
   output = pipeline.run(test_data, do_eval=True)
   ```

   Key config settings:
   ```yaml
   generator_model: "/projects/prjs1800/models/base/qwen2.5-7b-instruct"
   retrieval_method: "e5"
   retrieval_model: "/projects/prjs1800/models/base/e5-base-v2"
   retrieval_topk: 5
   dataset_name: "hotpotqa"  # then "musique"
   metrics: ["em", "f1", "recall", "precision"]
   ```

4. **Extract retrieval quality per item**
   FlashRAG saves intermediate results. For each question, check:
   - Which documents were retrieved (their IDs/content)
   - Which documents are in the ground-truth supporting facts
   - Compute: Recall@5, Precision@5, F1@5 for retrieval
   - **Per-hop retrieval**: For 2-hop MuSiQue questions, does the retriever find hop-1 docs? hop-2 docs?

5. **Save all outputs** — raw predictions, retrieved docs, metrics per item

### SLURM Template
```bash
#!/bin/bash
#SBATCH --job-name=standard_rag_hotpotqa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=/projects/prjs1800/results/logs/%j_standard_rag.out

module load 2024
module load Python/3.11.5-GCCcore-13.2.0
source /projects/prjs1800/venvs/flashrag-venv/bin/activate
cd /projects/prjs1800/external/FlashRAG

python run_experiment.py --config configs/standard_rag_hotpotqa.yaml
python run_experiment.py --config configs/standard_rag_musique.yaml
```

### Deliverable
```
results/day1/
├── standard_rag_hotpotqa_results.json   # Full output with per-item scores
├── standard_rag_musique_results.json
├── retrieval_quality_analysis.json       # Per-item: retrieved_doc_ids vs GT_doc_ids
└── day1_summary.md                       # EM, F1, Recall@5, Precision@5
```

| Dataset | EM | F1 | Precision | Recall |
|---------|------|------|-----------|--------|
| HotpotQA | 31.64 | 42.01 | 44.19 | 43.18 |
| MuSiQue | 6.33 | 13.03 | 13.59 | 14.91 |

**Timing** (A100-SXM4-40GB, 128GB RAM, 16 CPUs):
| Dataset | N | Retrieval | Generation+Eval | Total |
|---------|------|-----------|-----------------|-------|
| HotpotQA | 7,405 | 272.6s | 550.1s | 822.7s (~14 min) |
| MuSiQue | 2,417 | 94.8s | 169.4s | 264.2s (~4.4 min) |

**Setup**: Qwen2.5-7B-Instruct (vLLM 0.15.0) + E5-base-v2 + 61GB flat FAISS index (21M wiki18 passages)
**Fix applied**: MKL LD_PRELOAD to bypass numpy OpenBLAS MAX_THREADS=2 limitation for FAISS search
**Results dir**: `outputs/day1/`

### Day 1 Retrieval Analysis

**HotpotQA Retrieval Recall@5** (2 GT supporting docs per question):
| Recall bucket | N | % | Avg answer F1 | EM rate |
|--------------|------|------|-------------|---------|
| Both docs found | 1,770 | 23.9% | 0.680 | 55.6% |
| 1 of 2 found | 3,870 | 52.3% | 0.414 | 30.4% |
| Neither found | 1,765 | 23.8% | 0.172 | 10.7% |

Average retrieval recall: **50.0%**. Bridge questions harder (F1=37.0%) than comparison (F1=62.1%).
Error breakdown: 31.6% correct, 36.4% partial retrieval, 21.3% total miss, 10.6% reasoning failure.

**MuSiQue Per-hop retrieval recall:**
| Hop | Recall | | Hop | Recall |
|-----|--------|-|-----|--------|
| Hop 1 | 33.5% | | Hop 3 | 6.5% |
| Hop 2 | 11.6% | | Hop 4 | 3.2% |

55.8% of MuSiQue questions had zero GT docs retrieved. Dominant failure mode is retrieval, not reasoning.

---

## Day 2 (Fri Feb 7): Add Reranker + Error Categorization ✅

### Goal
Add a cross-encoder reranker on top of E5 retrieval. Measure how much reranking improves retrieval quality and downstream answer quality. Simultaneously, do error categorization on Day 1 failures.

### Why reranker first?
It's the simplest component to add in FlashRAG (literally a decorator on the retriever) and directly tests: "Is the problem that the right documents are retrieved but ranked too low?" This is a critical diagnostic before moving to iterative methods.

### Tasks

1. **Add reranker to Standard RAG**
   FlashRAG supports cross-encoder rerankers that can be attached to any retriever via decorator. Use a standard cross-encoder like `cross-encoder/ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`.

   ```yaml
   # Add to config
   use_reranker: True
   rerank_model: "BAAI/bge-reranker-base"  # or download locally
   retrieval_topk: 20        # Retrieve more, then rerank to top-5
   rerank_topk: 5
   ```

2. **Run Standard RAG + Reranker on both datasets**

3. **Compare retrieval quality: before vs after reranking**
   - Does Recall@5 improve? (It should — you're selecting from top-20 instead of top-5)
   - Does F1 improve? (If retrieval was the bottleneck, yes)
   - How much improvement on MuSiQue specifically?

4. **Error categorization on 50 MuSiQu failures from Day 1**
   Write a script (or do manually) to categorize each failure:

   | Category | Definition | How to detect |
   |----------|-----------|---------------|
   | **Retrieval miss** | GT docs not in retrieved set | Check retrieved IDs vs GT IDs |
   | **Retrieval present, wrong answer** | GT docs retrieved but model answers wrong | GT docs in top-5 but EM=0 |
   | **Partial retrieval** | Hop-1 docs found, hop-2 missing | Check per-hop recall |
   | **Reasoning failure** | All needed info present, model still wrong | Manual inspection |

5. **Per-hop analysis for MuSiQue**
   MuSiQue questions have annotated decomposition. For each 2-hop question:
   - Was the hop-1 supporting paragraph retrieved?
   - Was the hop-2 supporting paragraph retrieved?
   - Hypothesis: hop-1 recall >> hop-2 recall (because you need hop-1's answer to formulate a good hop-2 query)

### Deliverable
```
results/day2/
├── reranker_hotpotqa_results.json
├── reranker_musique_results.json
├── error_categorization_musique.json     # 50 items with categories
├── per_hop_retrieval_analysis.json       # hop-1 vs hop-2 recall
└── day2_summary.md
```

| Method | HotpotQA EM | HotpotQA F1 | MuSiQue EM | MuSiQue F1 | Δ F1 (HQA) | Δ F1 (MSQ) |
|--------|-------------|-------------|------------|------------|------------|------------|
| Standard RAG | 31.64 | 42.01 | 6.33 | 13.03 | — | — |
| + Reranker (BGE-v2-m3, top-20→5) | 36.41 | 47.42 | 7.70 | 15.52 | **+5.4** | **+2.5** |

**Reranker**: BAAI/bge-reranker-v2-m3 cross-encoder. Retrieve top-20 with E5, rerank to top-5.
**Results dir**: `outputs/day2/`

### Retrieval Analysis: Day 1 (Standard RAG) vs Day 2 (+ Reranker)

**HotpotQA Retrieval Recall@5** (2 GT supporting docs per question, n=7,405):

| Retrieval recall | Day 1 (top-5) | Day 2 (+reranker, top-20→5) | Delta |
|-----------------|---------------|------------------------------|-------|
| Avg recall | 50.0% | 57.7% | **+7.6%** |
| Full recall (all GT found) | 23.9% | 33.6% | +9.7% |
| Zero recall (none found) | 23.8% | 18.2% | -5.6% |

Answer F1 by retrieval recall (Day 1):
| recall=0 (n=1,765): F1=0.172 | recall=0.5 (n=3,870): F1=0.414 | recall=1.0 (n=1,770): F1=0.680 |

By question type: Bridge (n=5,918): F1=37.0% | Comparison (n=1,487): F1=62.1%

HotpotQA error shift (Day 1 → Day 2): +353 correct, -384 total misses, +271 reasoning failures, -240 partial retrieval.

**MuSiQue Retrieval Recall@5** (2-4 GT docs per question, n=2,417):

| Retrieval recall | Day 1 (top-5) | Day 2 (+reranker, top-20→5) | Delta |
|-----------------|---------------|------------------------------|-------|
| Avg recall | 21.4% | 26.2% | **+4.8%** |
| Full recall (all GT found) | 3.3% | 5.1% | +1.8% |
| Zero recall (none found) | 55.8% | 46.5% | -9.3% |

**Per-hop retrieval recall (key finding):**
| Hop | Day 1 | Day 2 (+reranker) | Delta |
|-----|-------|-------------------|-------|
| Hop 1 | 33.5% | 39.6% | +6.2% |
| Hop 2 | 11.6% | 14.8% | +3.2% |
| Hop 3 | 6.5% | 12.4% | +5.9% |
| Hop 4 | 3.2% | 4.7% | +1.5% |

MuSiQue error categories (Day 1 → Day 2):
| Category | Day 1 | Day 2 | Delta |
|----------|-------|-------|-------|
| Correct | 153 (6.3%) | 186 (7.7%) | +33 |
| Reasoning failure | 52 (2.2%) | 85 (3.5%) | +33 |
| Partial retrieval | 910 (37.6%) | 1,069 (44.2%) | +159 |
| Total miss | 1,302 (53.9%) | 1,077 (44.6%) | -225 |

**Conclusions:**
1. Reranker gives +5.4 F1 on HotpotQA by promoting relevant docs from rank 6-20 into top-5.
2. MuSiQue gains only +2.5 F1 — later-hop docs aren't in top-20 at all, so reranking can't help.
3. Per-hop decay (33→12→7→3% Day 1, 40→15→12→5% Day 2) confirms fundamental limitation of single-shot retrieval.
4. 46.5% of MuSiQue questions still have zero GT docs after reranking — motivating iterative retrieval (Day 4).
5. Reasoning failures grew (+33 on MuSiQue, +271 on HotpotQA) — finding more docs exposes the model's reasoning limits.

---

## Day 3 (Sat Feb 8): Add Refiner (RECOMP / Selective-Context) ✅

### Goal
Test whether compressing/refining retrieved documents improves generation. This tests: "Is the problem noise in retrieved passages rather than missing passages?"

### Why refiner?
FlashRAG's own benchmarks show refiners give significant improvements on multi-hop datasets (+3-4 F1 on HotpotQA). The hypothesis: when you retrieve 5 passages, much of the content is irrelevant noise. Abstractive compression (RECOMP) or perplexity-based filtering (LongLLMLingua/Selective-Context) can focus the generator on the relevant parts.

### Tasks

1. **Run Standard RAG + RECOMP abstractive refiner**
   RECOMP has trained checkpoints for HotpotQA. FlashRAG supports it natively.
   ```yaml
   use_refiner: True
   refiner_name: "recomp_abstractive"
   refiner_model: "fangyuan/nq_abstractive"  # or hotpotqa checkpoint
   ```

2. **Run Standard RAG + Selective-Context (perplexity-based refiner)**
   Uses GPT-2 to compute token perplexity and drops low-information tokens.
   ```yaml
   use_refiner: True
   refiner_name: "selective_context"
   sc_compression_rate: 0.5
   ```

3. **Run Standard RAG + Reranker + Refiner (stack the components)**
   Test: does combining reranker + refiner help more than either alone?

4. **Measure token reduction**
   - How many input tokens before vs after refinement?
   - Does reduced input length correlate with better F1?

### Deliverable
```
results/day3/
├── recomp_hotpotqa_results.json
├── recomp_musique_results.json
├── selective_context_results.json
├── reranker_plus_refiner_results.json
└── day3_summary.md
```

| Method | Components | HotpotQA EM | HotpotQA F1 | MuSiQue EM | MuSiQue F1 | Compression | Δ F1 (HQA) | Δ F1 (MSQ) |
|--------|-----------|-------------|-------------|------------|------------|-------------|------------|------------|
| Standard RAG (Day 1) | retriever → generator | 31.64 | 42.01 | 6.33 | 13.03 | — | — | — |
| + Reranker (Day 2) | retriever → reranker → generator | 36.41 | 47.42 | 7.70 | 15.52 | — | +5.4 | +2.5 |
| + RECOMP | retriever → refiner → generator | 29.55 | 40.02 | 5.50 | 11.85 | 8.8% retained | **-2.0** | **-1.2** |
| + Selective-Context | retriever → SC → generator | 27.00 | 36.56 | 5.01 | 11.26 | 67.1% retained | **-5.4** | **-1.8** |

**RECOMP**: fangyuan/hotpotqa_abstractive (T5-based, 737.7M params). Generates abstractive summary of retrieved docs.
**Selective-Context**: openai-community/gpt2 (124.4M params). Drops low-perplexity tokens, 50% reduction ratio.
**Results dir**: `outputs/day3/`

### Timing (A100-SXM4-40GB)

| Experiment | Retrieval | Refining | Generation | Total |
|-----------|-----------|----------|------------|-------|
| RECOMP HotpotQA (n=7,405) | 212.2s | 9,956.5s (~2.8h) | 48.2s | 10,216.9s |
| RECOMP MuSiQue (n=2,417) | 175.3s | 3,455.5s (~58m) | 16.6s | 3,647.4s |
| SC HotpotQA (n=7,405) | 229.1s | 2,716.7s (~45m) | 228.1s | 3,173.9s |
| SC MuSiQue (n=2,417) | 175.3s | 867.7s (~14m) | 72.5s | 1,115.5s |

### Key Findings

1. **Both refiners HURT performance** across all datasets — noise is not the primary bottleneck.
2. **RECOMP compresses too aggressively** (8.8% retained) — the T5 summary loses critical facts needed for multi-hop reasoning.
3. **Selective-Context** retains 67% but still hurts (-5.4 F1 on HotpotQA) — perplexity-based filtering doesn't understand question relevance.
4. **MuSiQue hurt worse relatively** — when retrieval quality is already poor (21.4% recall), compression removes the little useful signal present.
5. **RECOMP is impractical** — 2.8 hours for HotpotQA refining alone (T5 seq2seq is slow on 7,405 examples).
6. **Conclusion: The problem is MISSING information, not NOISY information.** This strengthens the case for iterative retrieval (Day 4: IRCoT) over compression.
7. Refiners may only help in a stacked configuration (reranker → refiner) where retrieval quality is already higher — but the negative delta suggests this is unlikely to be significant.

---

## Day 4 (Sun Feb 9): IRCoT — Iterative Retrieval + Chain-of-Thought

### Goal
Run IRCoT (the simplest iterative/loop RAG). This is where you move from "single-shot retrieval" to "multi-turn retrieval" — the first step toward agentic behavior.

### Why IRCoT?
IRCoT is already implemented in FlashRAG. It interleaves retrieval with chain-of-thought: think → search → think → search → answer. This directly addresses the "hop-2 retrieval miss" problem you identified on Day 2 — after reasoning about hop-1, the model can formulate a better query for hop-2.

### Tasks

1. **Run IRCoT on both datasets**
   FlashRAG uses a LoopPipeline for IRCoT. Key config:
   ```yaml
   pipeline: "ircot"
   max_iter: 2                # Maximum retrieval rounds (start with 2)
   retrieval_topk: 5
   # IRCoT needs few-shot examples — FlashRAG provides demonstration files
   ```

2. **Compare retrieval quality across rounds**
   IRCoT retrieves multiple times. Track:
   - Round 1 Recall@5 vs Round 2 Recall@5
   - Does round 2 retrieve the hop-2 documents that round 1 missed?
   - What queries does the model generate for round 2?

3. **Error analysis: What does IRCoT fix vs what remains broken?**
   Take 30 of Day 2's "partial retrieval" failures (hop-2 miss):
   - How many does IRCoT now solve?
   - What are the new failure modes? (e.g., wrong reformulated query, reasoning loop gets stuck)

4. **Also run FLARE if time permits**
   FLARE (Forward-Looking Active Retrieval) triggers retrieval when the model is uncertain. Different philosophy from IRCoT — reactive vs planned.
   ```yaml
   pipeline: "flare"
   ```

### Deliverable
```
results/day4/
├── ircot_hotpotqa_results.json
├── ircot_musique_results.json
├── ircot_per_round_retrieval.json        # Recall@5 per retrieval round
├── ircot_query_reformulations.json       # What queries were generated per round
├── ircot_error_analysis.json             # Categorize remaining failures
└── day4_summary.md
```

| Method | Type | HotpotQA F1 | MuSiQue F1 | # Retrieval Rounds | Hop-2 Recall |
|--------|------|-------------|------------|--------------------|--------------|
| Standard RAG | Single-shot | ? | ? | 1 | ? |
| + Reranker | Single-shot | ? | ? | 1 | ? |
| + Best Refiner | Single-shot | ? | ? | 1 | ? |
| IRCoT | Iterative (loop) | ? | ? | ~2-3 | ? |
| FLARE | Active (loop) | ? | ? | variable | ? |

**Critical insight to extract**: How much of MuSiQue's improvement comes from better hop-2 retrieval vs better reasoning?

---

## Day 5 (Mon Feb 10): Reasoning Pipeline (Search-R1 / ReSearch)

### Goal
Run FlashRAG's ReasoningPipeline â€” the most advanced single-agent approach. This combines reasoning ability with search, representing methods like Search-R1, R1-Searcher, and ReSearch. FlashRAG reports F1 ~60 on HotpotQA with this pipeline.

### Why this matters
This establishes the **single-agent ceiling**. If the reasoning pipeline already solves most multi-hop problems, the argument for multi-agent becomes harder. If it doesn't, you know exactly where to intervene.

### Tasks

**PRIORITY: Missing Baseline - Standard RAG + Reranker + CoT Prompting**

0. **Run Standard RAG + Reranker + CoT baseline (DO THIS FIRST - ~2 hours)**
   
   This is critical to establish your **single-agent ceiling** before claiming multi-agent helps.
   
   **Why this matters:**
   - Day 2 showed reranker gives +5.4 F1
   - Day 4 showed IRCoT's CoT helps but context dilution (12 docs) hurts
   - Question: What if reranker (better top-5) + CoT prompt (no dilution)?
   
   **Config:**
   ```yaml
   use_reranker: True
   rerank_model: "BAAI/bge-reranker-v2-m3"
   retrieval_topk: 20
   rerank_topk: 5
   
   # Update prompt template to include CoT
   generator_prompt: |
     Answer the question based on the given documents. 
     Think step by step:
     1. What information do I need to answer this question?
     2. Which documents contain relevant information?
     3. How do I combine information from multiple documents?
     4. What is my final answer?
     
     Question: {question}
     Documents: {context}
     Answer:
   ```
   
   **Expected Results:**
   - HotpotQA: ~50-52 F1 (close to R3-RAG's 52.4 "RAG with CoT" benchmark)
   - MuSiQue: ~16-18 F1
   
   **Why this is your single-agent ceiling:**
   - 1 LLM call (vs IRCoT's ~3-4)
   - Best retrieval (reranker from top-20)
   - Best reasoning (CoT prompt)
   - No context dilution
   
   If multi-agent can't beat this, it's not worth the complexity.

1. **Run ReasoningPipeline (if time permits after baseline)**
   ```yaml
   pipeline: "reasoning"
   # May need specific model checkpoint or prompt-based approach
   # If no RL-trained checkpoint: use prompt-based reasoning (Search-o1 style)
   ```

   Note: The reasoning pipeline may require a model that's been trained/prompted for the `<search>...</search>` token pattern. Check FlashRAG docs for:
   - Whether Qwen2.5-7B-Instruct works out of the box with prompt-based reasoning
   - Whether you need a specific Search-R1 checkpoint

2. **If no trained reasoning checkpoint available: use Self-Ask or ReAct prompting**
   Self-Ask is already in FlashRAG â€” it decomposes questions into sub-questions and searches for each.
   ```yaml
   pipeline: "self_ask"
   ```
   This is conceptually close to what a multi-agent decomposer would do, but in a single-agent loop.

3. **Compare with all previous methods**

4. **Analyze: Where does the single-agent ceiling lie?**
   - Which MuSiQue questions does even the best single-agent fail on?
   - Are there patterns? (e.g., 3+ hop questions, compositional questions, questions requiring parallel evidence)

### Deliverable
```
results/day5/
├── hotpotqa_2026_02_07_12_43_reranker_cot_qwen25_hotpotqa/
├── musique_2026_02_07_13_23_reranker_cot_qwen25_musique/
├── hotpotqa_2026_02_07_12_17_reasoning_qwen25_hotpotqa/
├── musique_2026_02_07_12_53_reasoning_qwen25_musique/
├── hotpotqa_2026_02_07_14_39_selfask_qwen25_hotpotqa/
├── musique_2026_02_07_16_15_selfask_qwen25_musique/
├── hotpotqa_2026_02_09_20_17_standard_cot_qwen25_hotpotqa/
├── musique_2026_02_09_20_37_standard_cot_qwen25_musique/
├── hotpotqa_2026_02_09_19_47_reranker_cot_short_qwen25_hotpotqa/
├── musique_2026_02_09_20_33_reranker_cot_short_qwen25_musique/
└── day5_summary.md
```

### Actual Results

| Method | Type | HotpotQA EM | HotpotQA F1 | MuSiQue EM | MuSiQue F1 | # LLM Calls | Notes |
|--------|------|-------------|-------------|------------|------------|-------------|-------|
| + Reranker + CoT | Single-shot | 34.40% | 45.52% | 8.11% | 13.99% | 1 | CoT hurts extractive QA |
| ReasoningPipeline | Reasoning | 3.81% | 17.70% | 1.70% | 10.54% | ~1.6-2.0 | Needs RL-trained model |
| SelfAsk (n=500) | Decomposition | 10.80% | 18.79% | 6.40% | 13.88% | ~5+ | 100x slower, impractical |
| Standard RAG + CoT | Single-shot | 30.45% | 40.44% | 6.41% | 11.62% | 1 | CoT without reranker also hurts |
| Reranker + CoT short (mt=32) | Single-shot | 0.00% | 1.67% | 0.00% | 1.27% | 1 | Catastrophic: 32 tokens all CoT, no answer |

**Key Findings:**
- All reasoning approaches DEGRADE performance vs Day 2 reranker-only
- **Single-agent ceiling IS the reranker baseline**: F1=47.42 (HQA), F1=15.52 (MSQ)
- CoT prompting hurts extractive QA (verbose answers don't match gold labels)
- ReasoningPipeline requires RL-trained checkpoints (Search-R1, R1-Searcher)
- SelfAsk is 100x slower and still performs worse
- Multi-agent must beat reranker ceiling through agent collaboration, not reasoning overhead

**Feb 9 Follow-up: 2x2 Factorial Design (CoT x Reranker x max_tokens)**

Completed the factorial to isolate CoT effect from reranker and max_tokens confounds:

| Configuration | max_tokens | HQA F1 | MSQ F1 | Delta from Day 1 |
|---------------|------------|--------|--------|-------------------|
| Standard RAG (Day 1) | 32 | 42.01% | 13.03% | baseline |
| Standard RAG + CoT | 256 | 40.44% | 11.62% | -1.6 / -1.4 |
| Reranker (Day 2) | 32 | 47.42% | 15.52% | +5.4 / +2.5 |
| Reranker + CoT | 256 | 45.52% | 13.99% | +3.5 / +1.0 |
| Reranker + CoT short | 32 | 1.67% | 1.27% | -40.3 / -11.8 |

**Factorial Conclusions:**
1. **CoT consistently hurts F1 by ~2 points** regardless of reranker (42.0→40.4, 47.4→45.5) — confirms verbose output hypothesis
2. **max_tokens=32 + CoT = catastrophic** (EM=0.0%) — model spends all 32 tokens on reasoning trace ("2. Doc 1 provides...") and never produces an answer
3. **Reranker effect (+5.4 F1) is robust** and independent of prompting strategy
4. The reranker benefit comes from retrieval quality, not generation — it works regardless of CoT prompt
5. **CoT needs token headroom** — with 256 tokens it's mildly harmful; with 32 tokens it's completely destructive

Timing (Feb 9 runs, A100):
| Experiment | Retrieval | Generation | Total |
|-----------|-----------|------------|-------|
| Standard RAG + CoT HQA (n=7,405) | 319.4s | 656.8s | 976.3s |
| Standard RAG + CoT MSQ (n=2,417) | 186.2s | 212.1s | 398.3s |
| Reranker + CoT short HQA (n=7,405) | — | 417.3s | 2,430.2s |
| Reranker + CoT short MSQ (n=2,417) | — | 122.8s | 824.3s |

---

## Day 6 (Tue Feb 10): Bounding Experiments + Statistical Analysis + Error Taxonomy ✅

### Goal
Establish lower bound (naive generation), upper bound (gold context), third dataset (2WikiMultihopQA), bootstrap confidence intervals, significance tests, systematic error taxonomy, and cross-method complementarity analysis.

### Actual Results

#### Performance Ladder with 95% Bootstrap CIs (1000 samples)

| Method | HQA F1 [95% CI] | MSQ F1 [95% CI] | 2Wiki F1 [95% CI] |
|---|---|---|---|
| Naive Gen (lower bound) | 25.29 [24.45, 26.25] | 9.59 [8.63, 10.51] | 29.69 [28.97, 30.41] |
| Standard RAG (Day 1) | 42.01 [41.03, 43.04] | 13.03 [11.94, 14.18] | 32.13 [31.43, 32.91] |
| + Reranker (Day 2, best) | **47.42 [46.39, 48.43]** | **15.52 [14.29, 16.77]** | **34.78 [34.04, 35.57]** |
| IRCoT (Day 4) | 42.61 [41.62, 43.64] | 14.29 [13.25, 15.48] | — |
| **Gold Context (upper)** | **51.31 [50.29, 52.34]** | **59.62 [57.68, 61.49]** | **69.96 [69.21, 70.68]** |

#### Significance Tests (paired bootstrap, all p < 0.001 unless noted)

| Comparison | HQA F1 diff | MSQ F1 diff | 2Wiki F1 diff |
|---|---|---|---|
| Retrieval value (Std RAG − Naive) | +16.72*** | +3.44*** | +2.45*** (EM: ns) |
| Reranking value (Reranker − Std RAG) | +5.40*** | +2.49*** | +2.65*** |
| IRCoT vs Reranker | -4.81*** | — | — |
| CoT effect (Rnk+CoT − Reranker) | -1.86*** | — | — |
| **Room for Improvement (Gold − Reranker)** | **+3.90***** | **+44.09***** | **+35.17***** |

#### Error Taxonomy: Reranker (Best Method) Failure Distribution

| Category | HQA (n=7,405) | MSQ (n=2,417) | 2Wiki (n=12,576) |
|---|---|---|---|
| Correct | 49.5% | 16.5% | 34.9% |
| Retrieval Miss (Total) | 14.7% | 42.0% | 20.7% |
| Retrieval Miss (Partial) | 26.9% | 38.6% | 39.7% |
| Reasoning Failure | 8.3% | 2.7% | 4.4% |
| Extraction Failure | 0.6% | 0.2% | 0.3% |

**Key insight**: 80.6% of MuSiQue failures and 60.4% of 2Wiki failures are retrieval misses → multi-agent decomposition/verification should target these.

#### Cross-Method Complementarity (Venn Analysis)

| Dataset | Ensemble Ceiling | Best Single | Gap | Unique to Reranker | Unique to IRCoT |
|---|---|---|---|---|---|
| HotpotQA | 61.1% | 49.5% | +11.6% | 510 | 497 |
| MuSiQue | 24.8% | 16.5% | +8.3% | 126 | 113 |
| 2WikiMultihopQA | 41.2% | 34.9% | +6.3% | 1,173 | — |

**Methods solve DIFFERENT questions.** A routing-based multi-agent system could reach the ensemble ceiling (+6-12% above best single method).

#### Remaining Gap Analysis

| Metric | HotpotQA | MuSiQue | 2WikiMultihopQA |
|---|---|---|---|
| % of gap closed (Naive → Gold) | 85.0% | 11.8% | 12.6% |
| % remaining for multi-agent | 15.0% | **88.2%** | **87.4%** |

HotpotQA is nearly solved by the reranker. MuSiQue and 2Wiki have massive room for improvement — the multi-agent thesis argument is strongest on these datasets.

#### Per-Hop Recall (MuSiQue) — Steep Decay Across All Methods

| Hop | Std RAG | Reranker | IRCoT |
|---|---|---|---|
| Hop 1 | 33.4% | 39.6% | 46.2% |
| Hop 2 | 11.6% | 14.8% | 24.9% |
| Hop 3 | 6.5% | 12.4% | 15.1% |
| Hop 4 | 3.2% | 4.7% | 7.2% |

### Result Directories
```
/projects/prjs1800/results/day6/
├── hotpotqa_2026_02_10_10_52_naive_gen_qwen25_hotpotqa/
├── musique_2026_02_10_10_56_naive_gen_qwen25_musique/
├── 2wikimultihopqa_2026_02_10_10_57_naive_gen_qwen25_2wiki/
├── hotpotqa_2026_02_10_10_58_gold_context_qwen25_hotpotqa/
├── musique_2026_02_10_11_00_gold_context_qwen25_musique/
├── 2wikimultihopqa_2026_02_10_11_08_gold_context_qwen25_2wiki/
├── 2wikimultihopqa_2026_02_10_11_16_standard_rag_qwen25_2wiki/
└── 2wikimultihopqa_2026_02_10_11_52_reranker_qwen25_2wiki/

/projects/prjs1800/analysis/day6/
├── bootstrap_results_all.json
├── significance_tests.json
├── error_taxonomy.json
├── cross_method_venn.json
└── hop_analysis_musique.json
```

### Failure-to-Solution Mapping for Multi-Agent Design

| Error Category | % of Failures (MSQ) | Proposed Multi-Agent Solution |
|---|---|---|
| Retrieval Miss (Total): 42.0% | **Decomposition Agent**: break into sub-queries per hop |
| Retrieval Miss (Partial): 38.6% | **Verification Agent**: check coverage, re-retrieve for gaps |
| Reasoning Failure: 2.7% | **Dedicated Reasoning Agent**: with verified context |
| Later-hop Decay: hop3=12.4%, hop4=4.7% | **Iterative Agent**: use earlier-hop answers for later queries |

---

## Day 7 (Wed Feb 12): Prototype Multi-Agent + Plan Next Sprint

### Goal
Implement a minimal prototype of your chosen multi-agent design. Even if numbers aren't great, having running code gives you momentum and concrete things to iterate on.

### Tasks

1. **Implement multi-agent prototype**
   Extend FlashRAG's pipeline. The simplest approach is to orchestrate multiple SequentialPipeline calls:
   ```python
   # Pseudo-code for Parallel Search Paths
   class MultiAgentRAGPipeline:
       def __init__(self, config):
           self.decomposer = Generator(config)  # LLM for decomposition
           self.rag_pipeline = SequentialPipeline(config)  # Reuse FlashRAG
           self.aggregator = Generator(config)   # LLM for synthesis

       def run(self, dataset):
           for item in dataset:
               # Step 1: Decompose
               sub_questions = self.decomposer.generate(
                   f"Break this into independent sub-questions: {item.question}"
               )
               # Step 2: RAG each sub-question
               sub_answers = []
               for sq in sub_questions:
                   result = self.rag_pipeline.run_single(sq)
                   sub_answers.append(result)
               # Step 3: Aggregate
               final = self.aggregator.generate(
                   f"Given these findings, answer: {item.question}\n"
                   f"Finding 1: {sub_answers[0]}\n"
                   f"Finding 2: {sub_answers[1]}"
               )
               item.pred = final
           return dataset
   ```

2. **Test on 100 MuSiQue questions first** (fast iteration)

3. **Compare with best single-agent method**

4. **Plan next sprint based on findings**

### Deliverable
```
results/day7/
├── multi_agent_prototype_musique_100.json
├── multi_agent_vs_single_agent.md
├── next_sprint_plan.md
└── code/
    └── multi_agent_pipeline.py
```

---

## Master Summary: What You'll Have After This Week

| Day | What you do | Key output | Feeds into |
|-----|------------|------------|------------|
| 1 | Standard RAG baseline | EM, F1, Recall@5 numbers + per-item retrieval quality | Foundation for everything |
| 2 | + Reranker + error categorization | Reranker delta + failure taxonomy (retrieval miss / reasoning fail / partial retrieval) | Motivation for refiners and iterative methods |
| 3 | + Refiner (RECOMP / SC) | Both hurt: RECOMP -2.0, SC -5.4 F1 on HQA | Noise is NOT the problem |
| 4 | IRCoT / FLARE | Iterative retrieval delta + per-round retrieval analysis + hop-2 improvement | Shows value of multi-turn retrieval |
| 5 | Reasoning pipeline | Single-agent ceiling + hardest remaining failures | Upper bound for single-agent |
| 6 | Consolidation + design | Master table + component deltas + multi-agent hypothesis + design doc | The "why multi-agent" argument |
| 7 | Multi-agent prototype | Running code + initial numbers | Momentum for next sprint |

---

## The Incremental Component Stack (Your Thesis Narrative)

```
Standard RAG                    [Day 1: baseline]
    │   HotpotQA F1=42.01, MuSiQue F1=13.03
    │
    ├── + Reranker              [Day 2: better ranking of retrieved docs]
    │       Δ: +5.4 F1 (HQA), +2.5 F1 (MSQ). Helps ranking, not total misses.
    │
    ├── + Refiner               [Day 3: compress noise from retrieved passages]
    │       Δ: -2.0 F1 (HQA), -1.2 F1 (MSQ) with RECOMP. NOISE IS NOT THE PROBLEM."
    │
    ├── + Iterative Retrieval   [Day 4: multi-round search]
    │       IRCoT: +0.5 F1 (HQA), +1.3 F1 (MSQ). Recall improves but context dilution hurts.
    │       FLARE: -15.4 F1 (HQA), -1.6 F1 (MSQ). Model overconfidence blocks retrieval.
    │
    ├── + Reasoning             [Day 5: single-agent reasoning approaches]
    │       Reranker+CoT: -1.9 F1 (HQA), -1.6 F1 (MSQ). CoT hurts extractive QA.
    │       Standard RAG+CoT: -1.6 F1 (HQA), -1.4 F1 (MSQ). CoT effect independent of reranker.
    │       Reranker+CoT(mt=32): -40.3 F1 (HQA). Catastrophic — no room for answer.
    │       ReasoningPipeline: -29.7 F1 (HQA). Needs RL-trained model.
    │       SelfAsk: -28.6 F1 (HQA). Format parsing failures + 100x slower.
    │       2x2 FACTORIAL: CoT always hurts ~2 F1. Reranker always helps ~5 F1.
    │       SINGLE-AGENT CEILING = DAY 2 RERANKER: F1=47.4 (HQA), F1=15.5 (MSQ)
    │
    └── + Multi-Agent           [Day 7: parallel decomposition + aggregation]
            Must beat reranker ceiling. Value from coordination, not reasoning overhead.
```

Each step answers a specific question. The cumulative deltas tell your thesis story.

---

## Technical Notes for Claude Code

### Key FlashRAG Architecture
- **Pipelines**: SequentialPipeline, LoopPipeline (IRCoT/FLARE), ConditionalPipeline (SKR/Adaptive-RAG), SelfRAGPipeline, ReasoningPipeline
- **Components**: Judger, Retriever, Reranker (decorator on retriever), Refiner (extractive/abstractive/perplexity), Generator
- **Datasets**: Pre-processed at `RUC-NLPIR/FlashRAG_datasets` on HuggingFace
- **Corpus**: wiki18_100w (Dec 2018 Wikipedia snapshot, 100-word chunks)
- **Index**: Pre-built E5 index available on ModelScope: `FlashRAG_Dataset/retrieval_corpus/wiki18_100w_e5_index.zip`

### Model Sizes
- Qwen2.5-7B-Instruct: ~14GB FP16 / ~5GB Q4 — single GPU ✅
- E5-base-v2: ~400MB — runs on CPU
- BGE-reranker-base: ~400MB — runs on CPU or GPU
- RECOMP abstractive: small T5-based model

### Snellius Specifics
- Partition: `gpu`
- Max GPU request: 1 A100 (80GB) should be more than enough
- Project dir: `/projects/prjs1800/`
- Results: `/projects/prjs1800/results/`

### Important FlashRAG Tips
- FlashRAG auto-saves intermediate results (retrieved docs, refined text, etc.) — use these for analysis
- Retrieval metrics (recall@k, precision@k) are built-in via the `metrics` config
- For per-item analysis, the output dataset object contains all intermediate states
- FlashRAG supports vLLM for faster inference — use it if available on Snellius
- Config files can override everything — keep a separate yaml per experiment for reproducibility

### Datasets with GT Supporting Facts
- **HotpotQA**: `supporting_facts` field contains (title, sentence_idx) pairs
- **MuSiQue**: Has decomposed sub-questions + supporting paragraphs per hop
- **2WikiMultihopQA**: Similar structure — use as tertiary dataset if time permits