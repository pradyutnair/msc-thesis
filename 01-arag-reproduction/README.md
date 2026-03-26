# 01-arag-reproduction — Baselines on Wiki18 (21M passages)

## Experimental Setting

- **Corpus**: FlashRAG wiki18_100w (21,015,324 passages, ~100 words each)
- **Retriever**: intfloat/e5-base-v2 + FAISS Flat index (61GB, exact search)
- **Generator**: Qwen/Qwen3-8B (vLLM, temperature=0.0)
- **Questions**: First 1,000 from each dataset test split
- **Eval**: Normalized Exact Match (EM) and Token F1

## Baseline Results (2026-03-26)

| Method | HotpotQA EM / F1 | 2WikiMH EM / F1 | MuSiQue EM / F1 | Mean EM |
|--------|-------------------|------------------|------------------|---------|
| Naive RAG | 13.4 / 21.3 | 9.4 / 16.3 | 3.4 / 8.7 | 8.7 |
| E2 ReAct | 31.1 / 40.5 | 19.9 / 26.5 | 12.9 / 21.7 | 21.3 |
| **IRCoT** | **34.9 / 46.1** | **30.0 / 38.4** | **19.3 / 26.7** | **28.1** |

### Method Details

- **Naive RAG**: Single retrieval (top-5) + generate. FlashRAG SequentialPipeline.
- **IRCoT** (Trivedi et al., ACL 2023): Interleaved retrieval + chain-of-thought, 5 iterations, top-5 per round, batch_size=50. FlashRAG custom script.
- **E2 ReAct**: Single-agent ReAct loop (max 15 loops, 128K token budget) with keyword_search (FTS5), semantic_search (FAISS e5), read_chunk tools. 50 concurrent workers.

### Notes

- Naive RAG and IRCoT answers extracted with regex (Qwen3-8B produces verbose output with thinking tokens).
- E2 ReAct uses a concise-answer system prompt (`prompts/e2_wiki18.txt`).
- All methods use the same first-1000 questions per dataset (seed=2024, no random sampling).

## File Locations

| Component | Path |
|-----------|------|
| Naive RAG configs | `../00-baseline-preparation/configs/baselines/naive_rag_qwen3_*.yaml` |
| IRCoT configs | `../00-baseline-preparation/configs/baselines/ircot_qwen3_*.yaml` |
| E2 configs | `configs/e2_wiki18_qwen3_*.yaml` |
| E2 wiki18 batch runner | `scripts/batch_runner_wiki18.py` |
| Wiki18 tools (FAISS+FTS5) | `scripts/arag-single-agent-scaffold/05_flashrag_tools_scaffold.py` |
| E2 system prompt | `prompts/e2_wiki18.txt` |
| Naive RAG results | `/projects/prjs1800/results/baselines/naive_rag_qwen3/` |
| IRCoT results | `/projects/prjs1800/results/baselines/ircot_qwen3/` |
| E2 results | `results/e2_wiki18_qwen3/` |
| FAISS index | `/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index` |
| Wiki18 corpus | `/projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl` |
| FTS5 index | `data/index/wiki18_fts.db` |
| ID offset map | `data/index/wiki18_id_offset.json` |
