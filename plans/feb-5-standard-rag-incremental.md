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
| CoT (no retrieval) | — | 34.0 | 31.1 | 12.7 |
| RAG with CoT | E5 | 52.4 | 33.5 | 16.9 |
| IRCoT | E5 | 48.4 | 35.8 | 13.5 |
| R3-RAG (RL-trained) | E5 | 65.5 | 62.3 | 33.6 |

FlashRAG's own benchmarks (LLaMA-3-8B + E5-base-v2):

| Method | Type | Pipeline | HotpotQA F1 | Notes |
|--------|------|----------|-------------|-------|
| Standard RAG | Sequential | SequentialPipeline | ~35 | Single retrieval, top-5 |
| IRCoT | Loop | LoopPipeline | ~41.5 | Iterative retrieval+CoT |
| FLARE | Loop | FLAREPipeline | ~38 | Active retrieval on uncertainty |
| RECOMP | Sequential+Refiner | SequentialPipeline | ~39 | Abstractive compression |
| Self-RAG | Conditional | SelfRAGPipeline | ~36 | Trained generator needed |
| Reasoning (Search-R1) | Reasoning | ReasoningPipeline | ~60 | Combines reasoning + search |

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

---

## Day 2 (Fri Feb 7): Add Reranker + Error Categorization

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

| Method | HotpotQA F1 | MuSiQue F1 | Recall@5 (HotpotQA) | Recall@5 (MuSiQue) |
|--------|-------------|------------|----------------------|---------------------|
| Standard RAG | 42.01 | 13.03 | — | — |
| + Reranker | ? | ? | ? | ? |

Error breakdown (MuSiQue, 50 failures):
| Category | Count | % |
|----------|-------|---|
| Retrieval miss (both hops) | ? | ? |
| Partial retrieval (hop-2 miss) | ? | ? |
| Reasoning failure | ? | ? |

---

## Day 3 (Sat Feb 8): Add Refiner (RECOMP / LongLLMLingua)

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

| Method | Components | HotpotQA F1 | MuSiQue F1 | Avg Input Tokens |
|--------|-----------|-------------|------------|------------------|
| Standard RAG | retriever → generator | ? | ? | ? |
| + Reranker | retriever → reranker → generator | ? | ? | ? |
| + RECOMP | retriever → refiner → generator | ? | ? | ? |
| + Reranker + RECOMP | retriever → reranker → refiner → generator | ? | ? | ? |
| + Selective-Context | retriever → SC → generator | ? | ? | ? |

**Key question to answer**: Did refining help more on questions where retrieval was noisy (lots of irrelevant content) vs where it was clean?

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
Run FlashRAG's ReasoningPipeline — the most advanced single-agent approach. This combines reasoning ability with search, representing methods like Search-R1, R1-Searcher, and ReSearch. FlashRAG reports F1 ~60 on HotpotQA with this pipeline.

### Why this matters
This establishes the **single-agent ceiling**. If the reasoning pipeline already solves most multi-hop problems, the argument for multi-agent becomes harder. If it doesn't, you know exactly where to intervene.

### Tasks

1. **Run ReasoningPipeline**
   ```yaml
   pipeline: "reasoning"
   # May need specific model checkpoint or prompt-based approach
   # If no RL-trained checkpoint: use prompt-based reasoning (Search-o1 style)
   ```

   Note: The reasoning pipeline may require a model that's been trained/prompted for the `<search>...</search>` token pattern. Check FlashRAG docs for:
   - Whether Qwen2.5-7B-Instruct works out of the box with prompt-based reasoning
   - Whether you need a specific Search-R1 checkpoint

2. **If no trained reasoning checkpoint available: use Self-Ask or ReAct prompting**
   Self-Ask is already in FlashRAG — it decomposes questions into sub-questions and searches for each.
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
├── reasoning_hotpotqa_results.json
├── reasoning_musique_results.json
├── self_ask_results.json
├── single_agent_ceiling_analysis.json    # Hardest remaining failures
└── day5_summary.md
```

---

## Day 6 (Tue Feb 11): Consolidation + Gap Analysis + Multi-Agent Design

### Goal
Compile all results, identify the gap, and design your first multi-agent extension.

### Tasks

1. **Build the master comparison table**

| # | Method | Components | Pipeline Type | HotpotQA F1 | MuSiQue F1 | Retrieval Recall@5 | # LLM Calls | Latency |
|---|--------|-----------|---------------|-------------|------------|--------------------|--------------| --------|
| 1 | Standard RAG | E5 → Qwen | Sequential | ? | ? | ? | 1 | ? |
| 2 | + Reranker | E5 → Reranker → Qwen | Sequential | ? | ? | ? | 1 | ? |
| 3 | + RECOMP | E5 → RECOMP → Qwen | Sequential | ? | ? | ? | 1 | ? |
| 4 | + Reranker + RECOMP | E5 → Reranker → RECOMP → Qwen | Sequential | ? | ? | ? | 1 | ? |
| 5 | IRCoT | E5 ↔ Qwen (loop) | Loop | ? | ? | ? | ~3-4 | ? |
| 6 | FLARE | E5 ↔ Qwen (active) | Loop | ? | ? | ? | variable | ? |
| 7 | Self-Ask / Reasoning | E5 ↔ Qwen (reasoning) | Reasoning | ? | ? | ? | variable | ? |

2. **Compute component-level deltas**

| Component Added | Δ HotpotQA F1 | Δ MuSiQue F1 | Cost (extra LLM calls) | Worth it? |
|----------------|---------------|--------------|------------------------|-----------|
| Reranker | ? | ? | 0 | ? |
| Refiner (RECOMP) | ? | ? | 0 (small model) | ? |
| Iterative retrieval (IRCoT) | ? | ? | +2-3 calls | ? |
| Reasoning (Self-Ask) | ? | ? | +3-5 calls | ? |

3. **Categorize remaining failures after best single-agent method**
   From 50 remaining MuSiQue failures:
   - How many are **fundamentally hard** (ambiguous, requires world knowledge not in corpus)?
   - How many are **decomposition failures** (single agent went down wrong path)?
   - How many are **evidence aggregation failures** (found pieces but couldn't combine)?
   - How many are **query diversity failures** (single perspective missed relevant docs)?

4. **Draft multi-agent hypothesis**
   Based on failure categories, articulate which specific failures a multi-agent system could address:
   - "X% of failures are decomposition failures → a dedicated Decomposer agent could improve sub-question generation"
   - "Y% are evidence aggregation failures → an Aggregator agent that explicitly synthesizes multi-source evidence could help"
   - "Z% are query diversity failures → parallel search agents with different query strategies could improve recall"

5. **Design minimal multi-agent extension**
   Building on FlashRAG's existing pipeline architecture:
   ```
   Option A: Parallel Search Paths (extends Branching pipeline)
   Query → Decomposer (LLM call) → [Sub-Q1, Sub-Q2]
                                      ↓           ↓
                                   RAG Path 1  RAG Path 2
                                      ↓           ↓
                                   Aggregator (LLM call) → Final Answer

   Option B: Iterative with Critic (extends Loop pipeline)
   Query → RAG Agent → Draft Answer
                ↓
           Critic Agent → "Missing info about X"
                ↓
           RAG Agent → Refined search for X → Final Answer

   Option C: Retrieve-then-Verify (extends Sequential pipeline)
   Query → Standard RAG → Candidate Answer
                              ↓
                    Verifier Agent → searches for contradicting evidence
                              ↓
                    If contradiction found → re-search and revise
                    If no contradiction → confirm answer
   ```

### Deliverable
```
results/day6/
├── master_comparison_table.md
├── component_delta_analysis.md
├── remaining_failure_categorization.json
├── multi_agent_hypothesis.md             # 1-page argument
└── multi_agent_design_doc.md             # Architecture for chosen approach
```

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
| 3 | + Refiner (RECOMP / SC) | Refiner delta + token reduction analysis | Shows if noise is a problem |
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
    │       Δ: ?F1, tests "are the right docs just ranked poorly?"
    │
    ├── + Refiner               [Day 3: compress noise from retrieved passages]
    │       Δ: ?F1, tests "is noise the problem?"
    │
    ├── + Iterative Retrieval   [Day 4: multi-round search]
    │       Δ: ?F1, tests "do we need multiple searches?"
    │
    ├── + Reasoning             [Day 5: think + search interleaved]
    │       Δ: ?F1, tests "single-agent ceiling"
    │
    └── + Multi-Agent           [Day 7: parallel decomposition + aggregation]
            Δ: ?F1, tests "does coordination help?"
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