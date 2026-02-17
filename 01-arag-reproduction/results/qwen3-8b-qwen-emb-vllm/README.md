# A-RAG Results: Qwen3-8B + Qwen3-Embedding-0.6B (DeepSeek Judge)

Updated on 2026-02-17 with unified judge: **DeepSeek-R1-Distill-Qwen-32B**.

## Setup

- Generator: Qwen3-8B (vLLM)
- Embedding: Qwen3-Embedding-0.6B
- Judge: DeepSeek-R1-Distill-Qwen-32B
- Datasets: HotpotQA, MuSiQue, 2WikiMultihop (1000 each)

## Results

| Dataset | LLM-Acc (%) | Contain-Acc (%) | Avg Loops | Avg Retrieved Tokens |
|---|---:|---:|---:|---:|
| HotpotQA | 54.8 | 55.4 | 2.53 | 783.3 |
| MuSiQue | 25.6 | 21.8 | 2.63 | 803.7 |
| 2WikiMultihop | 39.8 | 48.9 | 2.84 | 935.0 |
| **Mean** | **40.1** | **42.0** | **2.67** | **840.7** |

## Delta vs E2 (Qwen3-8B + E5)

| Dataset | LLM-Acc Delta | Contain-Acc Delta | Loops Delta | Retrieved Tokens Delta |
|---|---:|---:|---:|---:|
| HotpotQA | -4.5 pp | -4.0 pp | +0.09 | +68.9 |
| MuSiQue | -4.7 pp | -5.3 pp | -0.02 | +52.6 |
| 2WikiMultihop | -7.7 pp | -5.5 pp | +0.06 | +124.5 |
| **Mean** | **-5.6 pp** | **-5.0 pp** | **+0.04** | **+82.0** |

## Notes

- Qwen3-Embedding remains worse than E5-base-v2 in this pipeline.
- LLM-Acc and Contain-Acc both dropped across all datasets vs E5.
- Summary files:
  - `hotpotqa/predictions_eval_summary.json`
  - `musique/predictions_eval_summary.json`
  - `2wikimultihop/predictions_eval_summary.json`
