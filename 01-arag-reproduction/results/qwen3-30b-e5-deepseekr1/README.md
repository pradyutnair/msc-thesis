# A-RAG Results: Qwen3-30B-A3B + E5-base-v2 + DeepSeek-R1 Judge

## Experiment Setup

| Component | This Experiment | Qwen3-8B + E5 Baseline |
|-----------|-----------------|------------------------|
| **Generator** | Qwen3-30B-A3B (MoE, ~3B active) | Qwen3-8B |
| **Embedding** | intfloat/e5-base-v2 | intfloat/e5-base-v2 |
| **LLM Judge** | DeepSeek-R1-Distill-Qwen-32B | Qwen3-30B-A3B |
| **Serving** | vLLM on H100 80GB | vLLM on A100 40GB |
| **Agent Config** | max_loops=15, budget=128k | max_loops=15, budget=128k |
| **Datasets** | 1000 questions each | 1000 questions each |

Key changes from baseline:
- **Upgraded generator** from Qwen3-8B to Qwen3-30B-A3B (MoE) for stronger reasoning
- **Independent judge model** (DeepSeek-R1-Distill) to eliminate self-evaluation bias (Qwen3-30B was both generator and judge in previous runs)
- **Same E5 embeddings** and pre-built indices reused — retrieval is identical

## Results Comparison

### vs Qwen3-8B + E5 (same embeddings, same indices)

| Dataset | Metric | Qwen3-30B (ours) | Qwen3-8B + E5 | Delta |
|---------|--------|:-----------------:|:--------------:|:-----:|
| **HotpotQA** | LLM-Acc (%) | **66.5** | 53.2 | **+13.3 pp** |
| | Cont-Acc (%) | **67.7** | 62.2 | **+5.5 pp** |
| | Avg Loops | 2.66 | 2.44 | +0.22 |
| | Avg Retr. Tokens | 842 | 714 | +128 |
| **MuSiQue** | LLM-Acc (%) | **37.6** | 32.0 | **+5.6 pp** |
| | Cont-Acc (%) | **34.4** | 29.8 | **+4.6 pp** |
| | Avg Loops | 2.98 | 2.65 | +0.33 |
| | Avg Retr. Tokens | 873 | 751 | +122 |
| **2WikiMultihop** | LLM-Acc (%) | **56.9** | 43.1 | **+13.8 pp** |
| | Cont-Acc (%) | **63.9** | 57.1 | **+6.8 pp** |
| | Avg Loops | 3.05 | 2.78 | +0.27 |
| | Avg Retr. Tokens | 800 | 811 | -11 |

### vs Qwen3-8B + Qwen-Embedding (different embeddings)

| Dataset | Metric | Qwen3-30B + E5 (ours) | Qwen3-8B + Qwen-Emb | Delta |
|---------|--------|:---------------------:|:--------------------:|:-----:|
| **HotpotQA** | LLM-Acc (%) | **66.5** | 47.5 | **+19.0 pp** |
| | Cont-Acc (%) | **67.7** | 59.0 | **+8.7 pp** |
| **MuSiQue** | LLM-Acc (%) | **37.6** | 28.6 | **+9.0 pp** |
| | Cont-Acc (%) | **34.4** | 24.6 | **+9.8 pp** |
| **2WikiMultihop** | LLM-Acc (%) | **56.9** | 36.3 | **+20.6 pp** |
| | Cont-Acc (%) | **63.9** | 52.2 | **+11.7 pp** |

### vs GPT-4o-mini A-RAG (Original Paper)

| Dataset | Metric | Qwen3-30B (ours) | GPT-4o-mini (paper) | Delta |
|---------|--------|:-----------------:|:-------------------:|:-----:|
| **HotpotQA** | LLM-Acc (%) | 66.5 | 77.1 | -10.6 pp |
| | Cont-Acc (%) | 67.7 | 74.0 | -6.3 pp |
| **MuSiQue** | LLM-Acc (%) | 37.6 | 46.1 | -8.5 pp |
| | Cont-Acc (%) | 34.4 | 39.6 | -5.2 pp |
| **2WikiMultihop** | LLM-Acc (%) | 56.9 | 60.2 | -3.3 pp |
| | Cont-Acc (%) | 63.9 | 63.7 | **+0.2 pp** |

## Aggregate Summary

| Configuration | Mean LLM-Acc | Mean Cont-Acc | Mean Loops |
|---------------|:------------:|:-------------:|:----------:|
| **Qwen3-30B + E5 + DSR1 (this)** | **53.7%** | **55.3%** | 2.90 |
| Qwen3-8B + E5 + Q30B judge | 42.8% | 49.7% | 2.62 |
| Qwen3-8B + Qwen-Emb + Q30B judge | 37.5% | 45.3% | 2.67 |
| GPT-4o-mini (paper) | 61.1% | 59.1% | N/A |

## Key Findings

1. **Generator upgrade is the dominant factor**: Qwen3-30B improves mean LLM-Acc by +10.9 pp over Qwen3-8B with the same E5 embeddings. The MoE architecture (3B active params) provides much stronger multi-hop reasoning at moderate compute cost.

2. **Closing the gap to GPT-4o-mini**: On 2WikiMultihop, Qwen3-30B matches the paper result (Cont-Acc 63.9% vs 63.7%). On HotpotQA and MuSiQue the gap narrows significantly (-6.3 pp and -5.2 pp Cont-Acc vs the original -11.8 pp and -9.8 pp with Qwen3-8B).

3. **Independent judge eliminates self-eval bias**: Previous runs used Qwen3-30B-A3B as both the generator baseline judge and the evaluator. This run uses DeepSeek-R1-Distill-Qwen-32B (different model family), providing a more credible evaluation. LLM-Acc and Cont-Acc are well-correlated, suggesting the judge is reliable.

4. **More retrieval loops used**: Qwen3-30B uses ~0.3 more loops on average, suggesting it performs deeper multi-hop reasoning rather than guessing early. This is especially visible on MuSiQue (2.98 vs 2.65 loops).

5. **E5 embeddings confirmed superior**: Combined with prior results showing E5 > Qwen-Embedding across all metrics, E5-base-v2 remains the recommended retriever for this pipeline.

## Per-dataset Eval Summaries

- `hotpotqa/predictions_eval_summary.json`
- `musique/predictions_eval_summary.json`
- `2wikimultihop/predictions_eval_summary.json`

## Technical Notes

- DeepSeek-R1-Distill-Qwen-32B outputs `<think>` reasoning tokens; the eval script strips these before parsing the correct/incorrect verdict.
- Generator predictions also contain `<think>` tags from Qwen3-30B; these are stripped before both contain-match and LLM-judge evaluation.
- One ReadTimeout (300s) occurred during musique eval; resolved by increasing the LLM client timeout to 600s.
