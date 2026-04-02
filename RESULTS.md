# Wiki18 Benchmark Results (2026-03-26)

## Experimental Setting

- **Corpus**: FlashRAG wiki18_100w - 21,015,324 passages (~100 words each)
- **Retriever**: intfloat/e5-base-v2, FAISS Flat index (exact search, 61GB)
- **Generator**: Qwen/Qwen3-8B (vLLM, temperature=0.0, thinking disabled)
- **Questions**: First 1,000 from each dataset test split (seed=2024)
- **Eval**: Normalized Exact Match (EM) and Token F1

## Main Results


| Method    | Type                    | HotpotQA EM / F1 | 2WikiMH EM / F1 | MuSiQue EM / F1 | Mean EM  | Time (1000q) |
| --------- | ----------------------- | ---------------- | --------------- | --------------- | -------- | ------------ |
| Naive RAG | Single retrieval        | 13.4 / 21.3      | 9.4 / 16.3      | 3.4 / 8.7       | 8.7      | ~3 min       |
| E2 ReAct  | Single agent, 15 loops  | 31.1 / 40.5      | 19.9 / 26.5     | 12.9 / 21.7     | 21.3     | ~30 min      |
| IRCoT     | Iterative CoT, 5 rounds | 34.9 / 46.1      | 30.0 / 38.4     | 19.3 / 26.7     | **28.1** | ~10 min      |
| SAGE      | Multi-agent, 4 rounds   | 35.4 / 47.2      | 17.8 / 26.0     | 24.5 / 32.8     | 25.9     | ~90 min      |


## Published Baselines (from papers, different retrieval/models - not directly comparable)


| Method                | Type                    | HotpotQA EM | 2WikiMH EM | MuSiQue EM | LLM            |
| --------------------- | ----------------------- | ----------- | ---------- | ---------- | -------------- |
| IRCoT (Trivedi 2023)  | CoT                     | 42.7        | 43.3       | 18.8       | original paper |
| OPERA (AAAI 2026)     | RL multi-agent          | 57.3        | 60.2       | 39.7       | Qwen2.5-7B     |
| Search-R1 (COLM 2025) | Outcome RL              | 43.3        | 38.2       | 19.6       | Qwen2.5-7B     |
| ProRAG (Jan 2026)     | Process RL              | 41.4        | 46.0       | 23.5       | Qwen3-8B       |
| HiPRAG (Oct 2025)     | Hierarchical process RL | 71.7        | 34.1       | 52.8       | Qwen2.5-7B     |
| DualRAG (ACL 2025)    | Dual-process            | 49.7        | 65.6       | 40.8       | Qwen2.5-72B    |


## Key Findings

1. **Naive RAG is useless on 21M passages** - 8.7% mean EM.
2. **IRCoT is the best cost-performance tradeoff** - 28.1% mean EM in 10 min.
3. **E2 ReAct underperforms IRCoT** - agentic loop wastes tool calls. 21.3% in 30 min.
4. **SAGE wins on MuSiQue (+5.2pp over IRCoT)** but loses on 2WikiMH (-12.2pp). 29% empty on MuSiQue.
5. **SAGE is 9x slower than IRCoT** for marginal overall gain.
6. **ARAG toy corpus inflates results** - E4 53.7% on 1.3K corpus vs E2 21.3% on wiki18.

## SAGE Failure Analysis (MuSiQue)

- 289/1000 empty predictions (all answerable)
- Empty questions avg 3.9 waves (near max 4) vs 2.9 for answered
- Root cause: investigator queries fail on 21M passages or answer extraction fails

## File Locations


| Component         | Path                                                                    |
| ----------------- | ----------------------------------------------------------------------- |
| Naive RAG results | /projects/prjs1800/results/baselines/naive_rag_qwen3/                   |
| IRCoT results     | /projects/prjs1800/results/baselines/ircot_qwen3/                       |
| E2 ReAct results  | 01-arag-reproduction/results/e2_wiki18_qwen3/                           |
| SAGE results      | 04-sage-autonomous/results/sage_wiki18_1000/                            |
| Wiki18 corpus     | /projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl |
| FAISS index       | /projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index              |
| Questions         | 01-arag-reproduction/data/questions_wiki18/                             |


