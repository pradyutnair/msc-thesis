# A-RAG Reproduction Results: Qwen3-8B + Qwen3-Embedding-0.6B (vLLM)

## Scope

This folder contains results for:
- **Generator**: Qwen3-8B (vLLM)
- **Embedding**: Qwen3-Embedding-0.6B
- **Judge**: Qwen3-30B-A3B (vLLM)
- **Datasets**: HotpotQA, MuSiQue, 2WikiMultihop (1000 each)

Comparison baseline: `/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-8b-vllm/README.md` (same generator/judge, **E5-base-v2** embeddings).

## Results vs Baseline (E5-base-v2)

| Dataset | Metric | Qwen-Emb | E5 Baseline | Delta (Qwen-Emb - E5) |
|---|---:|---:|---:|---:|
| HotpotQA | LLM-Acc (%) | 47.5 | 53.2 | -5.7 pp |
| HotpotQA | Cont-Acc (%) | 59.0 | 62.2 | -3.2 pp |
| HotpotQA | Avg Loops | 2.53 | 2.44 | +0.09 |
| HotpotQA | Avg Retrieved Tokens | 783.3 | 714 | +69.3 |
| MuSiQue | LLM-Acc (%) | 28.6 | 32.0 | -3.4 pp |
| MuSiQue | Cont-Acc (%) | 24.6 | 29.8 | -5.2 pp |
| MuSiQue | Avg Loops | 2.63 | 2.65 | -0.02 |
| MuSiQue | Avg Retrieved Tokens | 803.7 | 751 | +52.7 |
| 2WikiMultihop | LLM-Acc (%) | 36.3 | 43.1 | -6.8 pp |
| 2WikiMultihop | Cont-Acc (%) | 52.2 | 57.1 | -4.9 pp |
| 2WikiMultihop | Avg Loops | 2.84 | 2.78 | +0.06 |
| 2WikiMultihop | Avg Retrieved Tokens | 935.0 | 811 | +124.0 |

## Aggregate View

- Mean **LLM-Acc**: 37.5% vs 42.8% (**-5.3 pp**)
- Mean **Cont-Acc**: 45.3% vs 49.7% (**-4.4 pp**)
- Mean **Avg Loops**: 2.67 vs 2.62 (**+0.04**)
- Mean **Avg Retrieved Tokens**: 840.7 vs 758.7 (**+82.0**, ~+10.8%)
- Answer rate: **100%** on all three datasets

## Per-dataset Eval Summaries

- `/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-8b-qwen-emb-vllm/hotpotqa/predictions_eval_summary.json`
- `/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-8b-qwen-emb-vllm/musique/predictions_eval_summary.json`
- `/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-8b-qwen-emb-vllm/2wikimultihop/predictions_eval_summary.json`

## Notes

- In this setup, replacing E5 with Qwen3-Embedding-0.6B reduced both LLM-Acc and Cont-Acc across all datasets.
- Retrieval became more verbose (higher retrieved-token counts) with little or no loop-count reduction.
- One transient eval API reset occurred on 2Wiki and was rerun successfully with lower eval concurrency.
