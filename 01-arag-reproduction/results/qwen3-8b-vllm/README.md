# A-RAG Results: Qwen3-8B + E5-base-v2 (DeepSeek Judge)

Updated on 2026-02-17 with unified judge: **DeepSeek-R1-Distill-Qwen-32B**.

## Setup

- Generator: Qwen3-8B (vLLM)
- Embedding: intfloat/e5-base-v2
- Judge: DeepSeek-R1-Distill-Qwen-32B
- Datasets: HotpotQA, MuSiQue, 2WikiMultihop (1000 each)

## Results

| Dataset | LLM-Acc (%) | Contain-Acc (%) | Avg Loops | Avg Retrieved Tokens |
|---|---:|---:|---:|---:|
| HotpotQA | 59.3 | 59.4 | 2.44 | 714.4 |
| MuSiQue | 30.3 | 27.1 | 2.65 | 751.1 |
| 2WikiMultihop | 47.5 | 54.4 | 2.78 | 810.5 |
| **Mean** | **45.7** | **47.0** | **2.62** | **758.7** |

## vs Paper (GPT-4o-mini)

| Dataset | LLM-Acc Delta | Contain-Acc Delta |
|---|---:|---:|
| HotpotQA | -17.8 pp | -14.6 pp |
| MuSiQue | -15.8 pp | -12.5 pp |
| 2WikiMultihop | -12.7 pp | -9.3 pp |

## Notes

- These values replace older Qwen3-30B-judge numbers.
- This directory now uses the same judge family as E1/E3/E4 for apples-to-apples LLM-Acc comparison.
- Summary files:
  - `hotpotqa/predictions_eval_summary.json`
  - `musique/predictions_eval_summary.json`
  - `2wikimultihop/predictions_eval_summary.json`
