# A-RAG Results: Qwen3-30B-A3B + E5-base-v2 (DeepSeek Judge)

Updated on 2026-02-17 after E2/E3 re-judge and corrected E1 rerun.

## Setup

- Generator: Qwen3-30B-A3B
- Embedding: intfloat/e5-base-v2
- Judge: DeepSeek-R1-Distill-Qwen-32B
- Datasets: HotpotQA, MuSiQue, 2WikiMultihop (1000 each)

## Results

| Dataset | LLM-Acc (%) | Contain-Acc (%) | Avg Loops | Avg Retrieved Tokens |
|---|---:|---:|---:|---:|
| HotpotQA | 66.5 | 67.7 | 2.66 | 842.1 |
| MuSiQue | 37.6 | 34.4 | 2.98 | 873.2 |
| 2WikiMultihop | 56.9 | 63.9 | 3.05 | 800.1 |
| **Mean** | **53.7** | **55.3** | **2.90** | **838.5** |

## Delta vs E2 (Qwen3-8B + E5)

| Dataset | LLM-Acc Delta | Contain-Acc Delta |
|---|---:|---:|
| HotpotQA | +7.2 pp | +8.3 pp |
| MuSiQue | +7.3 pp | +7.3 pp |
| 2WikiMultihop | +9.4 pp | +9.5 pp |
| **Mean** | **+8.0 pp** | **+8.3 pp** |

## Delta vs E3 (Qwen3-8B + Qwen-Embedding)

| Dataset | LLM-Acc Delta | Contain-Acc Delta |
|---|---:|---:|
| HotpotQA | +11.7 pp | +12.3 pp |
| MuSiQue | +12.0 pp | +12.6 pp |
| 2WikiMultihop | +17.1 pp | +15.0 pp |
| **Mean** | **+13.6 pp** | **+13.3 pp** |

## Delta vs Paper (GPT-4o-mini)

| Dataset | LLM-Acc Delta | Contain-Acc Delta |
|---|---:|---:|
| HotpotQA | -10.6 pp | -6.3 pp |
| MuSiQue | -8.5 pp | -5.2 pp |
| 2WikiMultihop | -3.3 pp | +0.2 pp |

## Notes

- E4 remains the best overall run in this reproduction set.
- E4 quality gains over E2/E3 hold under the unified independent judge.
- Summary files:
  - `hotpotqa/predictions_eval_summary.json`
  - `musique/predictions_eval_summary.json`
  - `2wikimultihop/predictions_eval_summary.json`
