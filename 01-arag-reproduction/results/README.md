# A-RAG Reproduction: Unified-Judge Results (DeepSeek-R1-Distill-Qwen-32B)

Updated on 2026-02-17 after:
- Re-judging **E2** and **E3** with DeepSeek-R1-Distill-Qwen-32B
- Re-running **E1** on the correct ARAG per-dataset chunked corpus/index (not FlashRAG wiki18 flat)
- Applying evaluator sanitation for both `<think>` and `<thnk>` tags

All experiments use 1000 questions per dataset across HotpotQA, MuSiQue, and 2WikiMultihop.

## Experiment Configurations

| ID | Generator | Embedding | Judge | Index Source | Hardware |
|---|---|---|---|---|---|
| **E1** | Qwen2.5-7B-Instruct | E5-base-v2 | DeepSeek-R1-Distill-Qwen-32B | ARAG chunked index | A100 (gen), H100 (eval) |
| **E2** | Qwen3-8B | E5-base-v2 | DeepSeek-R1-Distill-Qwen-32B | ARAG chunked index | A100 (gen), H100 (eval) |
| **E3** | Qwen3-8B | Qwen3-Embedding-0.6B | DeepSeek-R1-Distill-Qwen-32B | ARAG chunked index | A100 (gen), H100 (eval) |
| **E4** | Qwen3-30B-A3B | E5-base-v2 | DeepSeek-R1-Distill-Qwen-32B | ARAG chunked index | H100 (gen + eval) |
| **Ref** | GPT-4o-mini | paper default | GPT-4o-mini | paper default | API |

## Config Files

- E1: `configs/arag_qwen25_vllm_e5_{dataset}.yaml`
- E2: `configs/arag_qwen3_vllm_e5_{dataset}.yaml`
- E3: `configs/arag_qwen3_vllm_qwenemb_{dataset}.yaml`
- E4: `configs/arag_qwen3_30b_vllm_e5_{dataset}.yaml`

## Main Results: LLM-Accuracy (%)

| Dataset | E1 (Q2.5-7B) | E2 (Q3-8B+E5) | E3 (Q3-8B+QEmb) | E4 (Q3-30B+E5) | Paper (GPT-4o-mini) |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 65.9 | 59.3 | 54.8 | **66.5** | 77.1 |
| MuSiQue | 31.9 | 30.3 | 25.6 | **37.6** | 46.1 |
| 2WikiMultihop | 48.4 | 47.5 | 39.8 | **56.9** | 60.2 |
| **Mean** | 48.7 | 45.7 | 40.1 | **53.7** | 61.1 |

## Main Results: Contain-Accuracy (%)

| Dataset | E1 (Q2.5-7B) | E2 (Q3-8B+E5) | E3 (Q3-8B+QEmb) | E4 (Q3-30B+E5) | Paper (GPT-4o-mini) |
|---|---:|---:|---:|---:|---:|
| HotpotQA | **68.3** | 59.4 | 55.4 | 67.7 | 74.0 |
| MuSiQue | 32.2 | 27.1 | 21.8 | **34.4** | 39.6 |
| 2WikiMultihop | 53.1 | 54.4 | 48.9 | **63.9** | 63.7 |
| **Mean** | 51.2 | 47.0 | 42.0 | **55.3** | 59.1 |

## Efficiency (Average)

| Dataset | Metric | E1 | E2 | E3 | E4 |
|---|---|---:|---:|---:|---:|
| HotpotQA | Avg Loops | 3.33 | 2.44 | 2.53 | 2.66 |
| HotpotQA | Avg Retrieved Tokens | 2222.2 | 714.4 | 783.3 | 842.1 |
| MuSiQue | Avg Loops | 3.99 | 2.65 | 2.63 | 2.98 |
| MuSiQue | Avg Retrieved Tokens | 2584.0 | 751.1 | 803.7 | 873.2 |
| 2WikiMultihop | Avg Loops | 3.48 | 2.78 | 2.84 | 3.05 |
| 2WikiMultihop | Avg Retrieved Tokens | 1886.8 | 810.5 | 935.0 | 800.1 |

## Key Findings

1. Judge confound is removed for E1-E4: all now use DeepSeek-R1-Distill-Qwen-32B.
2. E4 remains the strongest overall configuration (best mean LLM-Acc and mean Contain-Acc).
3. E5-base-v2 still outperforms Qwen3-Embedding-0.6B for Qwen3-8B (E2 > E3 across all datasets).
4. Correcting E1 corpus/index changes its conclusions substantially; the prior FlashRAG-wiki18 E1 numbers were not comparable.
5. E1 reaches high containment but with much higher retrieval/tool-use cost (2-3x retrieved tokens vs E2/E3/E4).

## Result File Locations

- `results/qwen25-7b-instruct/{hotpotqa,musique,2wikimultihop}/predictions_eval_summary.json`
- `results/qwen3-8b-vllm/{hotpotqa,musique,2wikimultihop}/predictions_eval_summary.json`
- `results/qwen3-8b-qwen-emb-vllm/{hotpotqa,musique,2wikimultihop}/predictions_eval_summary.json`
- `results/qwen3-30b-e5-deepseekr1/{hotpotqa,musique,2wikimultihop}/predictions_eval_summary.json`
