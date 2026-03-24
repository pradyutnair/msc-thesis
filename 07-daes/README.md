# 07-daes: Diffusion-Native Hypothesize-Retrieve-Verify for Multi-Hop QA

## Core Idea

Diffusion LLMs enable a **hypothesize-retrieve-verify** loop for multi-hop QA:

1. **Hypothesize**: Single dLLM forward pass produces probability distribution over candidate bridge entities
2. **Retrieve**: Each candidate drives a targeted retrieval query for the next hop
3. **Verify**: Re-denoise with retrieved evidence; confidence *change* (not raw confidence) identifies evidence-supported paths
4. **Select**: Highest-scoring path becomes the answer

This is diffusion-native: AR models cannot re-score earlier tokens after seeing new evidence, and require k separate generations for k candidates (vs one forward pass for dLLMs).

## Current Results (Pilot, 50 questions, MuSiQue)


| Method                                                         | F1        | Precision | Recall    | Contain-Acc |
| -------------------------------------------------------------- | --------- | --------- | --------- | ----------- |
| Single-shot baseline                                           | 27.8%     | 28.4%     | 31.7%     | 26.0%       |
| SPREAD (query-relevance)                                       | 24.3%     | 24.0%     | 27.6%     | 22.0%       |
| **Branch-Verify (ours)**                                       | **30.5%** | **30.1%** | **35.7%** | **30.0%**   |
| SPREAD paper (Dream-7B baseline, 1000q + NV-Embed-v2 embedder) | 30.6%     | -         | -         | -           |


Branch-Verify matches SPREAD paper baseline using a weaker retriever (E5-base-v2 vs NV-Embed-v2).

## Key Findings

- **Retrieval corpus matters**: ARAG 1.3K chunks = 7% F1. MuSiQue native 26K paragraphs = 28% F1.
- **SPREAD underperforms baseline on multi-hop**: Single query embedding diluted across hops.
- **Branch-verify compensates for weak retrieval**: Multiple candidates + evidence verification achieves competitive results without a strong retriever.

## Setup

- **Model**: Dream-7B-Instruct (Dream-org/Dream-v0-Instruct-7B)
- **Retriever**: E5-base-v2 over MuSiQue paragraph pool (26,326 paragraphs)
- **Denoising**: 128 steps, 512 mask tokens, temperature 0.1
- **Branching**: 3 candidates per hop, scored by confidence change after evidence retrieval
- **Compute**: Single H100 GPU per job

## File Structure

```
src/daes/
  branch_verify.py         # Branch-and-Verify pipeline (core contribution)
  spread_reproduce.py      # SPREAD faithful reproduction
  build_musique_index.py   # Build E5 index from MuSiQue paragraph pool
  mvp_experiment.py        # Early sample() vs infill() experiments
  mvp_espread.py           # Early E-SPREAD experiments
  rag_debug.py             # Debug script for timing/diagnostics
jobs/                      # SLURM job files
results/                   # Prediction outputs (.jsonl) and logs (.out)
```

## Next Steps

1. Recursive branching at every hop (not just hop 1)
2. Adaptive branching (skip when model is confident)
3. NV-Embed-v2 retriever
4. Scale to 1000 questions
5. Multi-dataset: HotpotQA, 2WikiMultihopQA, MuSiQue
6. SPREAD integration at final synthesis step


## How This Differs From SPREAD

SPREAD modifies how the dLLM **generates** (token selection strategy during denoising). We modify what the dLLM **retrieves** (using its token distribution to drive multi-query retrieval for multi-hop QA). Different parts of the pipeline:

- **SPREAD**: retrieval is single-shot, improves generation quality via query-relevance-guided denoising
- **Ours**: generation is standard, improves retrieval coverage via candidate-driven multi-query retrieval

They are orthogonal — could be combined (SPREAD denoising + our multi-path retrieval).

## Current Status

**Working**: dLLM candidates + per-candidate retrieval = +4.8pp to +8.1pp F1 across 3 datasets. Proven by retrieval budget ablation (targeted 14 passages > naive 14 passages).

**Not working**: Confidence-based path verification. Random candidate selection outperforms scored selection. Scoring function is broken. The "verify" part of "hypothesize-retrieve-verify" is unresolved.

**Open**: Whether multi-query retrieval from dLLM candidates is a sufficient standalone contribution, or if working verification is needed for publication.
