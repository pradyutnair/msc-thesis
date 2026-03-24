# 07-daes Experiment Log

## Setup

- **Model**: Dream-7B-Instruct (`Dream-org/Dream-v0-Instruct-7B`), LLaDA-8B-Instruct (tested for SPREAD)
- **Library**: dllm (github.com/ZHZisZZ/dllm), patched for inference-only (no trainers/eval imports)
- **Venvs**: `arag-venv` (inference, retrieval), `nvembed-venv` (NV-Embed-v2 encoding only, transformers 4.42.4)
- **Datasets**: MuSiQue, HotpotQA, 2WikiMultihopQA (1000 questions each from ARAG)
- **Corpora**: Built from native paragraph pools — MuSiQue 26K paras, HotpotQA 43K paras, 2WikiMH 11K paras
- **Evaluation**: Token-level F1, Precision, Recall, Contain-Acc, Copy Rate
- **EMNLP deadline**: May 25, 2026

---

## Final Results (1000 questions per dataset, E5-base-v2 everywhere)


| Dataset  | Baseline F1 | Branch-Verify F1 | **Δ F1**   | Baseline Contain | BV Contain | **Δ Contain** |
| -------- | ----------- | ---------------- | ---------- | ---------------- | ---------- | ------------- |
| HotpotQA | 39.8%       | **47.9%**        | **+8.1pp** | 51.9%            | 56.3%      | +4.4pp        |
| MuSiQue  | 20.0%       | **24.8%**        | **+4.8pp** | 17.5%            | 23.4%      | +5.9pp        |
| 2WikiMH  | 23.9%       | **25.7%**        | **+1.8pp** | 20.4%            | 22.2%      | +1.8pp        |


- Retriever: E5-base-v2 for ALL queries (initial + branch). One retriever, one index per dataset. No mixing.
- Baseline: Dream-7B confidence-based denoising, single-shot retrieval top-5
- Branch-verify: Propose 3 hop candidates from dLLM distribution, retrieve per candidate, re-denoise with evidence, select by confidence change

---

## Retriever Experiments

### Retriever Impact on Baseline (MuSiQue, 50q pilot)


| Retriever      | Index             | Recall@5 | Baseline F1 |
| -------------- | ----------------- | -------- | ----------- |
| E5-base-v2     | ARAG 1.3K chunks  | 36%      | 7.3%        |
| E5-base-v2     | MuSiQue 26K paras | 64%      | 27.1%       |
| GTE-Qwen2-1.5B | MuSiQue 26K paras | —        | 26.3%       |
| NV-Embed-v2    | MuSiQue 26K paras | 54%      | 33.3%       |


**Key finding**: Corpus coverage matters more than retriever quality. ARAG's 1.3K chunks → 7% F1. MuSiQue native 26K paragraphs → 27% F1 (same retriever). NV-Embed-v2 adds another +6pp over E5.

---

## SPREAD Reproduction (ALL FAILED)

**Paper claims**: Dream-7B MuSiQue P=31.35 (maskgit-plus) → P=37.78 (SPREAD), +6pp. LLaDA P=23.91 → P=26.67, +3pp. No code released.

### All 13 attempts:


| #   | Variant                               | Retriever   | Result                | Issue                      |
| --- | ------------------------------------- | ----------- | --------------------- | -------------------------- |
| 1   | Manual loop, replace selection        | E5 (1.3K)   | 9.2% F1               | Wrong corpus               |
| 2   | Manual loop, replace selection        | E5 (26K)    | 24.3% F1 (-3pp vs BL) | Relevance flat (std=0.019) |
| 3   | Manual loop + EOS suppression (L=512) | NV-Embed-v2 | 4.8% F1               | "the the the" degeneration |
| 4   | Short masks (L=20) + EOS suppression  | NV-Embed-v2 | 12.1% F1              | Still worse                |
| 5   | Short masks (L=64) + EOS suppression  | NV-Embed-v2 | 5.6% F1               | Tied baseline              |
| 6   | Native Dream infill() + logits hook   | NV-Embed-v2 | 32.0% F1 (-1.3pp)     | Closest, no improvement    |
| 7   | LLaDA-8B-Instruct, manual loop        | NV-Embed-v2 | 12.9% F1 (-6.7pp)     | Even flatter (std=0.006)   |
| 8   | h_q from same forward pass            | NV-Embed-v2 | std=0.016             | Worse than separate        |
| 9   | Layer 12 hidden states (20q)          | NV-Embed-v2 | 19.0% F1              | Better than last layer     |
| 10  | Layer 16 hidden states (20q)          | NV-Embed-v2 | 14.1% F1              | Worse                      |
| 11  | Layer 20 hidden states (20q)          | NV-Embed-v2 | 21.0% F1              | Best layer, still -12pp    |
| 12  | Entropy vs argmax sampling (20q)      | E5          | 19% both              | Sampling irrelevant        |
| 13  | Without AR-shifted logits             | NV-Embed-v2 | 0-4 content tokens    | Much worse                 |


### Root cause analysis

- Hidden states at mask positions have near-zero variance: std=0.006 (LLaDA) to 0.034 (Dream layer 20)
- At step 0, 510/512 positions predict EOS. Relevance selection over all-EOS predictions is random.
- Variance peaks at layer 20 (std=0.034) but decreases at last layer (std=0.019)
- Relevance std across denoising steps: peaks at step 16-32 (0.044), never exceeds 0.044
- Baseline works by committing EOS first (high confidence), content tokens last (steps 123-127)
- SPREAD commits content tokens EARLIER (steps 2-12) but FEWER (2-6 vs 4-14)

### What SPREAD debug revealed

- SPREAD DOES identify content-producing positions earlier than baseline
- Q1: SPREAD got "June" (correct), baseline got "1950" (wrong)
- But SPREAD commits fewer total content tokens → lower overall F1
- Paper's Algorithm 1 has no EOS handling — loop runs exactly T steps unconditionally
- Efficiency numbers confirm model fills all 512 positions (~600 tokens at 31 tok/s)
- **Emailed authors for code (March 23)**

---

## Branch-and-Verify Development

### Mechanism

1. **Hypothesize**: Single dLLM forward pass → read token distribution → top-3 candidate bridge entities
2. **Retrieve**: Per candidate, retrieve evidence for next hop
3. **Verify**: Re-denoise with expanded evidence → measure confidence change
4. **Select**: Highest confidence-change path = evidence-supported hypothesis

### Pilot results (50q MuSiQue)


| Method             | Retriever    | F1    | Contain |
| ------------------ | ------------ | ----- | ------- |
| Baseline           | E5           | 27.1% | 26.0%   |
| Branch 1-hop       | E5           | 30.1% | 28.0%   |
| Recursive (3 hops) | E5           | 29.5% | 36.0%   |
| Baseline           | NV-Embed-v2  | 33.3% | 34.0%   |
| Branch-verify      | NV-Embed-v2* | 38.7% | 32.0%   |


*NV-Embed-v2 for initial, E5 fallback for branch queries (inconsistent — fixed in final run)

### Parametric ablation (20q MuSiQue)

55% of top-1 candidates change with vs without retrieved context. Model IS using evidence, not just parametric knowledge.

---

## Generation Behavior Analysis

### Dream-7B with 512 mask tokens

- Predicts EOS at ~508/512 positions regardless of content
- `skip_special_tokens=True` strips EOS → 4-word average output
- Confidence-based selection commits EOS first (highest neg_entropy)
- Content tokens committed only on last 2-5 steps when forced
- Copy Rate ~72% (4 words, ~3 from context) — matches SPREAD paper CR=77.65%
- Baseline F1=33.3% with NV-Embed-v2 matches paper's 30.56% — baseline IS reproduced

### Prompt format ablation (20q MuSiQue)


| Variant                | F1    | CR    | Avg words |
| ---------------------- | ----- | ----- | --------- |
| sample + chat template | 23.6% | 72.3% | 3         |
| sample + raw text      | 16.7% | 56.7% | 14        |
| infill + chat template | 23.6% | 72.3% | 3         |
| infill + raw text      | 16.7% | 56.7% | 14        |


sample() vs infill() makes zero difference. Chat template produces shorter but more accurate answers.

---

## File Map


| File                                    | Purpose                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------- |
| `src/daes/track2_bv_nvembed.py`         | **Main experiment script** — baseline + branch-verify, parameterized by dataset/shard |
| `src/daes/spread_reproduce.py`          | SPREAD reproduction with configurable retriever                                       |
| `src/daes/spread_native.py`             | SPREAD via Dream native sample() + logits hook                                        |
| `src/daes/spread_fixed.py`              | SPREAD with EOS suppression variants                                                  |
| `src/daes/branch_verify.py`             | Original single-hop branch-verify                                                     |
| `src/daes/recursive_branch_verify.py`   | Recursive multi-hop branch-verify                                                     |
| `src/daes/track1_llada_spread.py`       | SPREAD on LLaDA-8B                                                                    |
| `src/daes/build_musique_index.py`       | Build E5 index from MuSiQue paragraphs                                                |
| `src/daes/build_e5_indices.py`          | Build E5 indices for HotpotQA + 2WikiMH                                               |
| `src/daes/build_nvembed_index_v2.py`    | Build NV-Embed-v2 index (nvembed-venv)                                                |
| `src/daes/build_all_indices.py`         | Build NV-Embed-v2 indices for all datasets                                            |
| `src/daes/ablation_parametric.py`       | Context vs no-context candidate comparison                                            |
| `src/daes/reproduce_spread_baseline.py` | Prompt format ablation                                                                |
| `src/daes/debug_arshift.py`             | AR shift vs no shift debug                                                            |


---

## Timeline

- **March 22 evening**: Started. Set up dllm on Snellius, got Dream-7B running.
- **March 22 night**: First experiments — sample vs infill vs SPREAD on HotpotQA (gold evidence). SPREAD 70% > baseline 64%.
- **March 23 morning**: Discovered MuSiQue has no gold evidence in ARAG. Built MuSiQue index from native paragraphs. First SPREAD reproduction attempts fail.
- **March 23 afternoon**: Built NV-Embed-v2 index. Baseline reproduced (33.3% ≈ paper 30.6%). SPREAD still doesn't improve. Branch-verify shows +5.4pp on 50q pilot.
- **March 23 evening**: Scaled branch-verify to 1000q MuSiQue (+3.8pp with mixed retriever). LLaDA SPREAD also fails. Debugged SPREAD — found content tokens committed earlier but fewer.
- **March 23 night**: Fixed retriever consistency (E5 everywhere). Built E5 indices for all 3 datasets. Ran 60 parallel jobs (3 datasets × 1000q × 2 modes).
- **Final results**: +8.1pp HotpotQA, +4.8pp MuSiQue, +1.8pp 2WikiMH.

---

## Open Issues

1. **SPREAD reproduction**: 13 attempts, all failed. Emailed authors. Without their code, cannot reproduce claimed improvements.
2. **2WikiMH gain is small** (+1.8pp). May need recursive branching or different candidate extraction for comparison-type questions.
3. **No AR baseline comparison**: Need to compare branch-verify against AR models (Qwen3-8B) to show dLLM-specific value.
4. **Statistical significance**: Need confidence intervals or significance tests on 1000q results.
5. **EMNLP writing**: 8 weeks remaining.


---

## Retrieval Budget Ablation (50q MuSiQue, E5-base-v2)

**Critical test**: Is branch-verify just retrieving more passages?

| Method | Total passages | F1 | Contain |
|--------|---------------|-----|---------|
| baseline (top-5) | 5 | 27.8% | 26.0% |
| baseline (top-14) | 14 | 23.7% | 24.0% |
| **branch-verify** | ~14 | **31.8%** | **30.0%** |

**Result**: More passages HURTS (23.7% < 27.8%). Branch-verify with same budget HELPS (31.8% > 27.8%). The improvement comes from targeted per-candidate retrieval and confidence-change verification, not from seeing more evidence.

---

## Random Selection Ablation (50q MuSiQue, E5)

**Critical test**: Does confidence-based verification add value over random candidate selection?

| Method | F1 | Contain |
|--------|-----|---------|
| baseline_5 (top-5, no branching) | 27.8% | 26.0% |
| branch_verify (3 candidates, scored) | 28.6% | 28.0% |
| **branch_random** (1 random candidate) | **32.6%** | **32.0%** |

**Result**: Random selection BEATS confidence-scored selection. The scoring function (verified_conf + 0.5 * conf_gain) actively selects the wrong candidate due to scale mismatch between init_conf (temp=0.3 softmax, one token) and verified_conf (no-temp, 10-token average).

**Implication**: The improvement comes from multi-query retrieval (dLLM candidates diversify retrieval), NOT from confidence-based path verification. The "hypothesize-retrieve-verify" story collapses to "hypothesize-retrieve."

## Current Status (March 24)

### What Works
- dLLM token distribution produces useful bridge entity candidates for multi-hop QA
- Per-candidate retrieval improves over single-query retrieval (+4.8pp F1 on MuSiQue)
- Targeted multi-query retrieval > naive top-14 retrieval (budget ablation)
- Consistent gains across 3 datasets with E5 (HotpotQA +8.1pp, MuSiQue +4.8pp, 2WikiMH +1.8pp)
- NV-Embed-v2 + Dream-7B working in same venv (transformers 4.44.2)
- NV-Embed-v2 1000q MuSiQue run in progress

### What Does Not Work
- SPREAD reproduction (13 attempts, all failed. Author confirmed undocumented weighted scoring)
- Confidence-based path verification (broken scoring, random > scored)
- Any "diffusion-native verification" claim

### Open Questions
- Can we fix the verification scoring? (alternatives: self-consistency, answer length, cross-consistency)
- Is "dLLM-guided multi-query retrieval" a sufficient contribution for EMNLP without verification?
- How different is this from standard multi-query retrieval (IRCoT etc.) done for AR models?
- EMNLP deadline: May 25, 2026 (~8 weeks)


---

## Candidate Source Ablation (50q MuSiQue, E5) -- GO/NO-GO TEST

**Critical test**: Are dLLM candidates special, or does any query augmentation help?

### Setup
All methods use the same pipeline: retrieve top-5 for original question, then pick ONE random candidate from the candidate set, retrieve top-3 for "question + candidate", generate with expanded context. Only the candidate SOURCE differs.

| Source | How candidates are generated | F1 | Contain |
|--------|-----------------------------|----|---------|
| no_branch (baseline) | No candidates, just top-5 retrieval | 27.8% | 26.0% |
| **dLLM candidates** | Top-3 from Dream-7B token distribution at answer position | **33.0%** | **32.0%** |
| question_entities | Content words extracted from the question | 27.3% | 26.0% |
| random_words | Random common English words (city, country, person, etc.) | 28.7% | 26.0% |

### Result
dLLM candidates (+5.2pp F1) are the ONLY source that improves over baseline. Question entities and random words are no better than no branching at all. The dLLM distribution produces uniquely useful bridge entity candidates that other sources cannot replicate.

### What this proves
1. The improvement is NOT from "any query augmentation helps" -- question entities and random words add noise
2. The dLLM's token distribution captures meaningful bridge entity hypotheses for multi-hop questions
3. These hypotheses drive targeted retrieval that single-query retrieval cannot achieve

### Implementation details
- dLLM candidates: single forward pass on [evidence + question + "The answer is:" + MASK*20], read top-3 from softmax(logits/0.3) at first mask position, expand each by seeding first token + 16-step quick denoise
- Question entities: extract content words from question after stopword removal, take up to 3
- Random words: sample 3 from list of 20 common nouns (city, country, person, year, etc.)
- All use random.seed(42) for reproducibility
- All use same retrieve() and generate() functions

---

## Per-Hop Analysis (1000q MuSiQue, E5)

| Hops | N | Baseline F1 | Branch-verify F1 | Delta |
|------|---|------------|------------------|-------|
| 2-hop | 518 | 22.7% | 28.9% | +6.2pp |
| 3-hop | 316 | 19.7% | 22.0% | +2.3pp |
| 4-hop | 166 | 12.1% | 17.4% | +5.3pp |

Branch-verify improves on ALL hop counts. Largest gains on 2-hop (+6.2pp) and 4-hop (+5.3pp).
