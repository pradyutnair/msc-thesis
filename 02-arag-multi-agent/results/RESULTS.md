# M6 Multi-Agent RAG — Benchmark Results

All results on 1000 questions per dataset, using Qwen3-8B, wiki18_100w corpus with E5-base-v2 retrieval.

## Main Comparison

| Dataset | Metric | **M6v29** | M6v23 | OPERA v2 |
|---|---|---|---|---|
| **HotpotQA** | EM | 45.6% | **48.8%** | 36.3% |
| | Contain | 61.1% | **63.6%** | 51.0% |
| **2WikiMH** | EM | **56.3%** | 45.9% | 20.4% |
| | Contain | **64.4%** | 56.4% | 31.8% |
| **MuSiQue** | EM | **25.9%** | 22.5% | 25.7% |
| | Contain | **38.9%** | 34.6% | 38.0% |
| **Mean** | EM | **42.6%** | 39.1% | 27.5% |
| | Contain | **54.8%** | 51.5% | 40.3% |

## M6v29 Changes (from v23)

Zero extra LLM calls. All improvements are better information flow + prompts:

1. **Answer-type aware workers** — Decomposer produces specific `unknown_entities` per sub-question (e.g., `"country name"`, `"birth year"`). Workers see "What You Must Find: country name" and answer accordingly (e.g., "United States" not "Virginia").

2. **Dependency evidence inheritance** — For bridge chains (SQ0→SQ1), blackboard computes `dependency_chunk_ids` from predecessor's verified evidence. Workers read predecessor evidence first before searching.

3. **Better decomposer** — Comparison questions ask "What year was X born?" (not "When was X born?") with typed `unknown_entities`. Better examples for bridge questions.

4. **Cleaner synthesizer** — Tighter comparison/yes-no/bridge instructions.

## Result Paths

- `results/m6v29_1000/` — M6v29 (answer-type aware + dependency evidence)
- `results/m6v23_1000/` — M6v23 baseline
- `/projects/prjs1800/external/OPERA/results/arag_qwen3_v2_norm/` — OPERA baseline

## Configs

- `configs/m6v29.yaml` — M6v29 config
- `configs/m6v23.yaml` — M6v23 config
