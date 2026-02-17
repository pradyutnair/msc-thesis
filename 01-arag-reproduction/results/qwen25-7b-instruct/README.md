# A-RAG Results: Qwen2.5-7B-Instruct + E5-base-v2 (Corrected E1)

Updated on 2026-02-17.

This E1 run replaces the earlier invalid E1 numbers that used the wrong FlashRAG wiki18 corpus/index.
Current E1 uses the same ARAG per-dataset chunked setup as E2/E3/E4 and the same judge model family.

## Setup

- Generator: Qwen2.5-7B-Instruct (vLLM)
- Embedding: intfloat/e5-base-v2
- Judge: DeepSeek-R1-Distill-Qwen-32B
- Datasets: HotpotQA, MuSiQue, 2WikiMultihop (1000 each)

## Results

| Dataset | LLM-Acc (%) | Contain-Acc (%) | Avg Loops | Avg Retrieved Tokens |
|---|---:|---:|---:|---:|
| HotpotQA | 65.9 | 68.3 | 3.33 | 2222.2 |
| MuSiQue | 31.9 | 32.2 | 3.99 | 2584.0 |
| 2WikiMultihop | 48.4 | 53.1 | 3.48 | 1886.8 |
| **Mean** | **48.7** | **51.2** | **3.60** | **2231.0** |

## Delta vs E2 (Qwen3-8B + E5)

| Dataset | LLM-Acc Delta | Contain-Acc Delta | Loops Delta | Retrieved Tokens Delta |
|---|---:|---:|---:|---:|
| HotpotQA | +6.6 pp | +8.9 pp | +0.89 | +1507.8 |
| MuSiQue | +1.6 pp | +5.1 pp | +1.34 | +1832.9 |
| 2WikiMultihop | +0.9 pp | -1.3 pp | +0.70 | +1076.3 |
| **Mean** | **+3.0 pp** | **+4.2 pp** | **+0.98** | **+1472.3** |

## Notes

- E1 is now directly comparable to E2-E4 on corpus/index and judge choice.
- Compared with E2, E1 improves quality metrics but uses substantially more retrieval/tool budget.
- Summary files:
  - `hotpotqa/predictions_eval_summary.json`
  - `musique/predictions_eval_summary.json`
  - `2wikimultihop/predictions_eval_summary.json`
