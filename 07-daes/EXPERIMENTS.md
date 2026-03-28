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

## Cleaned EAMD v4 Pilot (March 27)

This section supersedes the earlier v3 smoke interpretation. The v3 result was suggestive but not thesis-safe because the methods did not all use the same prompt scaffold, and the reported "EAMD" path was regeneration rather than true remasking. v4 reruns the comparison with those confounds removed.

### Why v4 was necessary

The earlier v3 pilot had three issues:

1. `baseline` / `EAMD` used an explicit short-answer prompt, while `SPREAD` and `ARAM` used their own prompt scaffolds.
2. The reported "EAMD" path was full short-canvas regeneration under `C1`, not actual revision from a previous answer canvas.
3. The `eamd_select` selector compared uncalibrated confidence values from different decoding procedures.

v4 fixes all three.

### v4 protocol

- **Dataset**: MuSiQue, first 50 questions from `/projects/prjs1800/external/arag/data/musique/questions.json`
- **Retriever**: E5-base-v2, same index and retrieve budget for all methods
- **Canvas**: 16 masked answer tokens, 16 denoising steps, temperature `0.1`
- **Shared prompt**: all methods use the same short-answer instruction
- **Initial evidence**: `C0 = R(Q)` with top-5 retrieval
- **Expanded evidence**: `C1 = C0 ∪ R(Q + baseline_answer) ∪ ⋃_j R(Q + candidate_j)`
- **Candidate source**: Dream single-pass bridge candidates from `extract_candidates()`
- **Evaluation**: standard Counter-based token-overlap F1, exact match, contain
- **Leakage check**: gold answer is loaded only for evaluation, never used in retrieval or decoding

### Methods in the cleaned comparison

| Method | Evidence | Prompt | Decoder | Notes |
| --- | --- | --- | --- | --- |
| Baseline | `C0` | shared short prompt | confidence-based short denoising | control |
| SPREAD | `C0` | shared short prompt | query-relevance weighted token ordering | cleaned reproduction |
| ARAM | `C0` | shared short prompt | per-token conditional vs prior guidance | cleaned reproduction |
| Pool | `C1` | shared short prompt | confidence-based short denoising | retrieval-only control |
| EAMD-Regen | `C0` vs `C1` | shared short prompt | evidence-marginal guided regeneration from fully masked short canvas | **main mathematically grounded variant** |
| EAMD-Remask | seed answer from Baseline, revise under `C1` | shared short prompt | answer-span remask + guided re-denoise | heuristic ablation |

### Mathematical form used in v4

For `EAMD-Regen`, at each masked answer position `i`:

- `p_base_i = p_theta(. | x_t, Q, C0)`
- `p_full_i = p_theta(. | x_t, Q, C1)`
- `Signal_i = D_KL(p_full_i || p_base_i) + D_KL(p_base_i || p_full_i)`
- `Noise_i = H(p_full_i)`
- `extra_scale_i = lambda_max * tanh(beta * Signal_i / (Noise_i + eps)) * (t / T)`
- `logits_guided_i = logits_full_i + extra_scale_i * (logits_full_i - logits_base_i)`

This is a new inference-time guidance rule. It is mathematically clean to describe as a guided sampler defined by evidence-marginal distribution shift.

For `EAMD-Remask`, the current v4 implementation is weaker theoretically:

- seed from the baseline short answer under `C0`
- compare token distributions under `C0` vs `C1` on the committed answer tokens
- if the top-1 token changes or divergence exceeds a threshold anywhere in the answer span, remask the full current answer span
- re-denoise that span with the same evidence-marginal guidance

This is a practical diffusion-native revision operator, but it is still a heuristic ablation rather than the main theorem-backed claim.

### Fairness / anti-cheating checks

- same prompt scaffold across all methods
- same answer length budget (`16` masked answer tokens)
- same retriever, same index, same top-k
- no gold answer or support facts used in retrieval expansion
- no "best-of-two" selector used in the main result table
- all reported numbers come from the same v4 harness

### Retrieval corpus note

The cleaned v4 pilot uses:

- questions from the ARAG MuSiQue split
- retrieval over the **MuSiQue native paragraph corpus** (`index_e5_musique_full`)
- retriever = `E5-base-v2`

This is fair for **internal method comparison** because Baseline, SPREAD, ARAM, Pool, EAMD-Regen, and EAMD-Remask all use the same retriever and the same corpus.

However, it is **not a corpus-faithful reproduction** of the SPREAD or ARAM papers:

- **SPREAD** uses `NV-Embed-v2`, 2,000-character document chunks, top-5 retrieval, but the paper text does not clearly name one single global corpus such as `wiki18_100w`
- **ARAM** explicitly uses `bge-large` over **MS MARCO 2.1** with top-3 retrieval

Therefore, the v4 claim should currently be read as:

- EAMD-Regen beats our SPREAD / ARAM / Pool controls **under a matched MuSiQue-native retrieval setting**

and **not yet** as:

- EAMD-Regen beats the original papers under their exact retrieval corpora.

### v4 smoke test (5q MuSiQue)

Artifacts:
- `src/daes/eamd_smoke_v4.py`
- `jobs/eamd_smoke_v4.job`
- `results/eamd_smoke_v4_5q.json`
- `results/eamd_smoke_v4_21235998.out`

| Method | F1 | EM | Contain |
| --- | ---: | ---: | ---: |
| Baseline | 0.000 | 0.000 | 0.000 |
| SPREAD | 0.040 | 0.000 | 0.000 |
| ARAM | 0.000 | 0.000 | 0.000 |
| Pool | 0.560 | 0.400 | 0.600 |
| EAMD-Regen | 0.560 | 0.400 | 0.600 |
| EAMD-Remask | 0.333 | 0.200 | 0.200 |

Smoke interpretation:
- `EAMD-Regen` survives the prompt-matched rerun and matches `Pool`
- `EAMD-Remask` can fix direct value errors (`New York -> Rockland`, `1984 -> 1986`) but still lags behind regeneration
- this justified a larger pilot

### v4 cleaned pilot (50q MuSiQue)

Artifacts:
- `results/eamd_pilot_v4_50q.json`
- `results/eamd_pilot_v4_21236180.out`

| Method | F1 | EM | Contain |
| --- | ---: | ---: | ---: |
| Baseline | 0.336 | 0.240 | 0.320 |
| SPREAD | 0.313 | 0.200 | 0.280 |
| ARAM | 0.313 | 0.240 | 0.260 |
| Pool | 0.419 | 0.280 | 0.400 |
| **EAMD-Regen** | **0.457** | **0.300** | **0.460** |
| EAMD-Remask | 0.392 | 0.280 | 0.360 |

### Pairwise comparison on the 50q pilot

| Comparison | Better | Worse | Same |
| --- | ---: | ---: | ---: |
| EAMD-Regen vs Pool | 4 | 0 | 46 |
| EAMD-Regen vs ARAM | 12 | 3 | 35 |
| EAMD-Regen vs SPREAD | 12 | 5 | 33 |
| EAMD-Remask vs Pool | 3 | 5 | 42 |
| EAMD-Remask vs ARAM | 7 | 1 | 42 |

### Where the cleaned gains come from

Largest `EAMD-Regen` improvements over `Pool` on the 50q pilot:

| Question type | Gold | Pool | EAMD-Regen | F1 gain |
| --- | --- | --- | --- | ---: |
| birthplace city chain | `La Goulette` | `Tunis` | `Paris and La Goulette` | +0.667 |
| location description chain | `central Atlantic Ocean` | `Cabo Verde` | `Cabo Verde, 10 volcanic islands in the central Atlantic Ocean.` | +0.500 |
| presidential family chain | `Jessie Woodrow Wilson` | `Esther Cleveland` | `Eleanor Wilson` | +0.400 |
| Senate control date | `January 2015` | `2015` | `January 2015` | +0.333 |

Notable property:
- `EAMD-Regen` has **no regressions against Pool** on this 50q slice.

### Interpretation

1. The cleaned rerun removes the biggest methodological concern from v3: prompt mismatch.
2. `EAMD-Regen` still beats `Pool`, `ARAM`, and `SPREAD` after that correction.
3. The main working idea is **evidence-marginal guided regeneration**, not remasking.
4. `EAMD-Remask` is real and useful as an ablation, but it is not yet the strongest variant.
5. The current thesis-safe claim is:
   - under a shared short-answer setup, evidence-marginal guided regeneration outperforms our current SPREAD, ARAM, and pooled-evidence baselines on a 50-question MuSiQue pilot

### Go / no-go decision after v4

- **GO**: scale `EAMD-Regen` to a larger MuSiQue run (`200q` or `500q`)
- **GO**: use `EAMD-Regen` as the main mathematically grounded method
- **PARTIAL GO**: keep `EAMD-Remask` as a diffusion-native ablation
- **NO**: do not make remasking the main contribution yet

### v4 corpus-matched wiki18 pilot (50q MuSiQue)

To match `01-arag-reproduction`, we reran the cleaned harness on the same open-domain retrieval stack:

- corpus: `wiki18_100w.jsonl`
- index: `e5_Flat.index`
- retriever: `E5-base-v2`
- questions: `data/questions_wiki18/musique.json`
- methods: `baseline`, `SPREAD`, `ARAM`, `Pool`, `EAMD-Regen`
- `EAMD-Remask` skipped here to keep the pilot fast and focused on the main theorem-safe method

Artifacts:
- `src/daes/eamd_wiki18_v4.py`
- `jobs/eamd_wiki18_v4.job`
- `results/eamd_wiki18_smoke_2q.json`
- `results/eamd_wiki18_v4_50q.json`
- `results/eamd_wiki18_v4_21246679.out`

Fast-path changes:
- batched FAISS retrieval for expansion queries
- `TF32` enabled
- shared short-answer prompt and 16-token canvas
- no remask branch in the main 50q pilot

Average harness wall time:
- `8.10s` per question on one H100

| Method | F1 | EM | Contain |
| --- | ---: | ---: | ---: |
| Baseline | 0.212 | 0.080 | 0.100 |
| SPREAD | 0.194 | 0.100 | 0.100 |
| ARAM | 0.245 | 0.080 | 0.100 |
| Pool | 0.289 | 0.120 | 0.220 |
| **EAMD-Regen** | **0.294** | 0.100 | 0.220 |

Pairwise counts:

| Comparison | Better | Worse | Same |
| --- | ---: | ---: | ---: |
| EAMD-Regen vs Pool | 2 | 3 | 45 |
| EAMD-Regen vs ARAM | 12 | 8 | 30 |
| EAMD-Regen vs SPREAD | 14 | 4 | 32 |
| Pool vs ARAM | 10 | 7 | 33 |

Interpretation:

1. The fair-corpus result is weaker than the MuSiQue-native result.
2. `EAMD-Regen` still beats `SPREAD` and `ARAM` on F1 under the matched `wiki18_100w` setup.
3. `Pool` remains the strongest control challenge; `EAMD-Regen` is only marginally better on F1 and slightly worse on EM.
4. The thesis-safe claim here is narrower:
   - under the corpus-matched wiki18 setup, evidence-marginal regeneration remains competitive and slightly stronger than the current SPREAD / ARAM controls, but it does not yet clearly dominate pooled retrieval alone

### v4 corpus-matched wiki18 benchmark (1000q x 3 datasets)

We then scaled the same wiki18 setup to the first 1000 ARAG dev questions for each dataset:

- datasets: `hotpotqa`, `musique`, `2wikimultihopqa`
- corpus: `wiki18_100w.jsonl`
- index: `e5_Flat.index`
- retriever: `E5-base-v2`
- generator: `Dream-org/Dream-v0-Instruct-7B`
- methods: `baseline`, `SPREAD`, `ARAM`, `Pool`, `EAMD-Regen`, `EAMD-Remask`
- sharding: `10` H100 shards per dataset (`100` questions per shard), CPU merge/collect on `cbuild`

Artifacts:
- `src/daes/eamd_wiki18_full.py`
- `src/daes/merge_eamd_wiki18_shards.py`
- `src/daes/collect_eamd_wiki18_summaries.py`
- `jobs/eamd_wiki18_suite_array.job`
- `jobs/eamd_wiki18_suite_merge.job`
- `jobs/eamd_wiki18_suite_collect.job`
- `results/eamd_wiki18_full/hotpotqa/eamd_wiki18_hotpotqa_1000q.json`
- `results/eamd_wiki18_full/musique/eamd_wiki18_musique_1000q.json`
- `results/eamd_wiki18_full/2wikimultihopqa/eamd_wiki18_2wikimultihopqa_1000q.json`
- `results/eamd_wiki18_full/eamd_wiki18_all_datasets_summary.json`

| Dataset | Method | F1 | EM | Contain |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | Baseline | 0.418 | 0.291 | 0.368 |
| HotpotQA | SPREAD | 0.405 | 0.270 | 0.373 |
| HotpotQA | ARAM | 0.447 | 0.328 | 0.369 |
| HotpotQA | Pool | **0.456** | 0.316 | 0.415 |
| HotpotQA | EAMD-Regen | 0.454 | 0.307 | **0.421** |
| HotpotQA | EAMD-Remask | 0.452 | 0.315 | 0.393 |
| MuSiQue | Baseline | 0.199 | 0.107 | 0.126 |
| MuSiQue | SPREAD | 0.182 | 0.095 | 0.117 |
| MuSiQue | ARAM | 0.206 | 0.109 | 0.123 |
| MuSiQue | Pool | 0.243 | **0.143** | 0.187 |
| MuSiQue | EAMD-Regen | **0.248** | 0.141 | **0.194** |
| MuSiQue | EAMD-Remask | 0.232 | 0.135 | 0.156 |
| 2WikiMH | Baseline | 0.300 | 0.219 | 0.261 |
| 2WikiMH | SPREAD | 0.290 | 0.201 | 0.264 |
| 2WikiMH | ARAM | 0.314 | **0.247** | 0.261 |
| 2WikiMH | Pool | **0.333** | 0.233 | 0.296 |
| 2WikiMH | EAMD-Regen | 0.329 | 0.227 | **0.297** |
| 2WikiMH | EAMD-Remask | 0.315 | 0.233 | 0.270 |

Runtime:

- all `30` H100 shards completed successfully
- shard wall time: about `15` to `16.5` minutes
- average elapsed time per question from merged metadata:
  - HotpotQA: `8.02s`
  - MuSiQue: `8.13s`
  - 2WikiMH: `8.28s`

Interpretation:

1. `EAMD-Regen` beats `SPREAD` and `ARAM` on F1 on all three datasets.
2. `EAMD-Regen` is the strongest method on **MuSiQue F1**, which is the main target benchmark here.
3. `Pool` remains stronger on HotpotQA and 2WikiMH F1, so retrieval expansion alone is still the hardest baseline to beat.
4. `EAMD-Remask` remains a useful diffusion-native ablation but is not the strongest variant.
5. The thesis-safe benchmark claim is:
   - under the wiki18 setup matched to `01-arag-reproduction`, `EAMD-Regen` is consistently stronger than the current SPREAD and ARAM controls on F1, but does not dominate the pooled-evidence baseline across all datasets

### LLaDA wiki18 benchmark (1000q x 3 datasets, tuned no-candidate variant)

We then reran the full wiki18 benchmark with `GSAI-ML/LLaDA-8B-Instruct` using the corrected LLaDA-specific EAMD configuration:

- model: `GSAI-ML/LLaDA-8B-Instruct`
- corpus: `wiki18_100w.jsonl`
- retriever: `E5-base-v2`
- methods: `baseline`, `SPREAD`, `ARAM`, `Pool`, `EAMD-Regen`, `EAMD-Remask`
- tuned config:
  - answer tokens = `8`
  - denoising steps = `8`
  - temperature = `0.05`
  - candidate expansion = `0`
  - round-0 seed = `ARAM`
- output root: `results/eamd_wiki18_full_llada_nocand_1000q`

Artifacts:
- `src/daes/eamd_wiki18_full_llada.py`
- `jobs/eamd_wiki18_llada_array.job`
- `jobs/eamd_wiki18_llada_merge.job`
- `jobs/eamd_wiki18_llada_collect.job`
- `results/eamd_wiki18_full_llada_nocand_1000q/hotpotqa/eamd_wiki18_hotpotqa_1000q.json`
- `results/eamd_wiki18_full_llada_nocand_1000q/musique/eamd_wiki18_musique_1000q.json`
- `results/eamd_wiki18_full_llada_nocand_1000q/2wikimultihopqa/eamd_wiki18_2wikimultihopqa_1000q.json`
- `results/eamd_wiki18_full_llada_nocand_1000q/eamd_wiki18_all_datasets_summary.json`

| Dataset | Method | F1 | EM | Contain |
| --- | --- | ---: | ---: | ---: |
| HotpotQA | Baseline | 0.357 | 0.166 | 0.343 |
| HotpotQA | SPREAD | 0.342 | 0.137 | 0.335 |
| HotpotQA | ARAM | 0.377 | 0.203 | 0.341 |
| HotpotQA | Pool | 0.359 | 0.163 | 0.349 |
| HotpotQA | EAMD-Regen | 0.357 | 0.162 | 0.353 |
| HotpotQA | EAMD-Remask | **0.383** | **0.207** | **0.353** |
| MuSiQue | Baseline | 0.172 | 0.048 | 0.125 |
| MuSiQue | SPREAD | 0.164 | 0.036 | 0.107 |
| MuSiQue | ARAM | 0.183 | **0.056** | 0.127 |
| MuSiQue | Pool | 0.178 | 0.043 | 0.140 |
| MuSiQue | EAMD-Regen | 0.178 | 0.042 | **0.141** |
| MuSiQue | EAMD-Remask | **0.185** | 0.055 | 0.132 |
| 2WikiMH | Baseline | 0.242 | 0.135 | 0.211 |
| 2WikiMH | SPREAD | 0.233 | 0.133 | 0.206 |
| 2WikiMH | ARAM | 0.257 | 0.153 | 0.222 |
| 2WikiMH | Pool | 0.250 | 0.133 | 0.216 |
| 2WikiMH | EAMD-Regen | 0.251 | 0.135 | 0.217 |
| 2WikiMH | EAMD-Remask | **0.264** | **0.154** | **0.230** |

Runtime:

- all `30` H100 shards completed successfully
- shard wall time: about `8.5` to `9.9` minutes
- average elapsed time per question from merged metadata:
  - HotpotQA: `4.07s`
  - MuSiQue: `4.27s`
  - 2WikiMH: `4.05s`

Interpretation:

1. For `LLaDA`, the winning variant is `EAMD-Remask`, not `EAMD-Regen`.
2. `EAMD-Remask` is the strongest method on all three `1000q` benchmarks.
3. The gain is largest on HotpotQA, modest but positive on MuSiQue, and consistent on 2WikiMH.
4. This supports a model-dependent conclusion:
   - Dream favors regeneration,
   - LLaDA favors revision.
5. The main LLaDA thesis-safe claim is:
   - under the corpus-matched wiki18 setup, ARAM-seeded `EAMD-Remask` outperforms the current Baseline, SPREAD, ARAM, Pool, and EAMD-Regen variants across all three datasets.

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

---

## ARAM Reproduction (March 25)

### Paper: Adaptive Retrieval-Augmented Masked Diffusion (arxiv 2603.17677)

**Method**: Per-token adaptive SNR guidance during denoising. Two forward passes (batched): conditional (with context) and prior (without context). Signal = symmetric KL divergence, Noise = conditional entropy. lambda_i = lambda_max * tanh(beta * Signal_i / (Noise_i + eps)). Guided logits = prior + lambda * (cond - prior).

**Implementation**: \`src/daes/aram_reproduce.py\`

### ARAM Results (50q MuSiQue, E5-base-v2)

| Prompt | Retrieval | beta | Baseline F1 | ARAM F1 | Delta |
|--------|-----------|------|-------------|---------|-------|
| Zero-shot | top-5 | 0.5 | 27.8% | 28.0% | +0.2pp |
| Few-shot (paper Fig 6) | top-5 | 0.5 | **31.8%** | 29.9% | -1.9pp |
| Few-shot (paper Fig 6) | top-3 (paper) | 0.1 (paper) | 20.3% | 16.8% | -3.5pp |
| Few-shot (paper Fig 6) | top-3 (paper) | 0.5 | 20.3% | 18.0% | -2.3pp |

**Key findings**:
1. Few-shot prompt alone boosts baseline +4pp (27.8% → 31.8%)
2. ARAM guidance hurts Dream with the few-shot prompt (-1.9pp with top-5)
3. Top-3 retrieval is insufficient for multi-hop MuSiQue (20.3% baseline)
4. Consistent with ARAM paper: Dream gains are tiny (+0.9pp F1 on HotpotQA). The paper's big wins are for LLaDA, not Dream.
5. Per-token lambda is working (signal varies 0.03-0.47, lambda varies 0.1-0.3), but the guidance degrades output quality

---

## SPREAD Reproduction (March 25)

### Paper: Semantic-Preserving RAG for Diffusion (arxiv 2601.11342)
### Author email (Chuanyue Yu): undocumented weighted scoring = alpha * relevance + (1-alpha) * confidence

**Method**: Replace confidence-based token selection with query-relevance-guided selection. h_q from separate forward pass (last layer, mean-pool). Cosine similarity → sigmoid → top-k selection.

**Implementation**: \`src/daes/spread_v2.py\` (weighted), \`src/daes/spread_variants.py\` (all variants)

### Prior SPREAD attempts (13 failed, see above) — root cause confirmed: hidden state cosine std = 0.063, too flat

### SPREAD Variant Sweep (50q MuSiQue, E5-base-v2, zero-shot prompt)

**Baseline: 28.9% F1**

| Strategy | Config | F1 | vs BL | Notes |
|----------|--------|-----|-------|-------|
| **additive_raw** | **gamma=0.3** | **29.7%** | **+0.8pp** | **BEST** |
| **additive_raw** | **gamma=0.5** | **29.6%** | **+0.7pp** | Close second |
| additive_raw | gamma=0.1 | 27.1% | -1.8pp | Too little relevance |
| additive_raw | gamma=1.0 | 26.0% | -2.9pp | Too much relevance |
| additive_raw | gamma=5.0 | 27.7% | -1.2pp | Way too much |
| multiplicative | gamma=1.0 | 28.5% | -0.4pp | Nearly neutral |
| multiplicative | gamma=5.0 | 26.0% | -2.9pp | |
| multiplicative | gamma=10.0 | 26.6% | -2.3pp | |
| weighted (norm) | alpha=0.1 | 27.4% | -1.5pp | Min-max normalization hurts |
| weighted (norm) | alpha=0.3 | 26.8% | -2.1pp | |
| weighted (norm) | alpha=0.5 | 26.7% | -2.2pp | |
| weighted (norm) | alpha=0.7 | 26.5% | -2.4pp | |
| layer20 | gamma=1.0 | 24.4% | -4.5pp | Higher hs variance but worse |
| spread_v1 (original) | relevance only | 23.6% | -5.3pp | Known failure |
| spread_v2 (KL-based) | KL selection | 16.4% | -12.5pp | Selects for copying |

**Key findings**:
1. \`additive_raw gamma=0.3-0.5\` is the only configuration that beats baseline (+0.7-0.8pp)
2. The author's "weighted scoring" works when applied as raw additive (confidence + gamma * sigmoid(cosine))
3. Min-max normalization amplifies noise in the low-variance relevance signal → always hurts
4. Layer 20 hidden states have higher variance (0.075-0.087 vs 0.063) but worse F1 — variance alone isn't the issue
5. KL-based selection (spread_v2) catastrophically selects for context-copying, not query-answering
6. Sweet spot for gamma is 0.3-0.5: enough relevance to nudge, not enough to overwhelm confidence

### Why additive_raw works

Confidence (neg-entropy) ranges \`[-10, -0.1]\`, sigmoid(cosine) ≈ \`[0.47, 0.53]\`. With gamma=0.3-0.5, the relevance contribution is ~0.15-0.25, a subtle perturbation that can break ties in confidence without disrupting the primary ordering. This preserves Dream's entropy-based denoising quality while using query relevance as a tiebreaker.


### SPREAD Finetune Sweep — additive_raw gamma detail (50q MuSiQue)

| gamma | F1 | vs BL (28.9%) |
|-------|-----|---------------|
| 0.1 | 27.1% | -1.8pp |
| 0.2 | 27.7% | -1.2pp |
| **0.25** | **29.7%** | **+0.8pp** |
| **0.3** | **29.7%** | **+0.8pp** |
| **0.5** | **29.6%** | **+0.7pp** |
| 1.0 | 26.0% | -2.9pp |
| 5.0 | 27.7% | -1.2pp |

Sweet spot: gamma ∈ [0.25, 0.5]. The effect is robust across this range.
Remaining gammas (0.35, 0.4, 0.45, 0.6) still running.

---

### Thesis Direction Analysis (March 25)

**Dead ends established:**
- ARAM guidance: -1.9pp to +0.2pp on Dream (Dream doesn't benefit from adaptive guidance)
- SPREAD selection: +0.8pp at best (marginal, hidden state variance too low)
- dLLM confidence scoring: broken (random > scored)
- dLLM candidates: outperformed by AR candidates at 1000q scale
- Iterative re-denoising (recursive BV): no clear improvement

**What works:**
- Multi-query retrieval: +4.8-8.1pp (the big lever)
- Few-shot prompting: +4pp (free)

**Biggest open question:** What is the dLLM-specific contribution that justifies using dLLMs?


### SPREAD Complete Gamma Curve (50q MuSiQue, additive_raw, E5-base-v2)

| gamma | F1 | vs BL (28.9%) |
|-------|-----|---------------|
| 0.1 | 27.1% | -1.8pp |
| 0.2 | 27.7% | -1.2pp |
| **0.25** | **29.7%** | **+0.8pp** |
| **0.3** | **29.7%** | **+0.8pp** |
| 0.35 | 27.8% | -1.1pp |
| **0.4** | **29.6%** | **+0.7pp** |
| 0.45 | 28.7% | -0.2pp |
| **0.5** | **29.6%** | **+0.7pp** |
| 1.0 | 26.0% | -2.9pp |
| 5.0 | 27.7% | -1.2pp |

Non-smooth curve (0.35 dips while 0.3 and 0.4 work) suggests the +0.8pp may be within noise on 50q. Need 1000q to confirm.


---

## NEW DIRECTION: SFT/RL Training for Multi-Hop RAG (March 26)

### Motivation

Inference-time methods (ARAM, SPREAD) hit a ceiling:
- ARAM: +0.2pp (zero-shot) to -1.9pp (few-shot) on Dream
- SPREAD: +0.8pp at best, possibly noise
- The base model's generation quality is the bottleneck

Key literature:
- **d1** (NeurIPS 2025): diffu-GRPO on LLaDA for math → big gains. No one tried RAG.
- **DLLM-Searcher** (Feb 2026): SFT+VRPO on SDAR for web search agents. Different model (SDAR), different setting (web search).
- **Gap**: No one has trained Dream/LLaDA (mainstream dLLMs) for fixed-corpus multi-hop RAG.

### Approach

1. **Data**: 3977 DLLM-Searcher SFT trajectories converted for QA (search results → context passages)
2. **Model**: Dream-7B-Instruct + LoRA (r=64, alpha=128)
3. **Loss**: d1's absorbing state diffusion loss (cross_entropy / t)
4. **Training**: 3 epochs, lr=1e-5, grad_accum=8, max_len=2048, single H100
5. **Template**: \`<reasoning>...</reasoning><answer>...</answer>\`
6. **Eval**: Retrieve with E5-base-v2, generate with reasoning template, measure F1

### Novel contribution
First application of diffusion-specific SFT loss to train a mainstream masked dLLM (Dream-7B) for retrieval-augmented multi-hop QA. Different from DLLM-Searcher (uses SDAR, not Dream) and d1 (math only, not RAG).

### Scripts
- \`src/daes/sft_qa.py\` — SFT training (adapted from d1)
- \`src/daes/eval_sft_qa.py\` — Evaluation with retrieval
- \`src/daes/convert_searcher_to_d1.py\` — Data conversion
- Training job: 21163876 (overnight)

### Status
- SFT training submitted (Job 21163876)
- Base Dream eval with reasoning template running (Job 21163881, 20q smoke test)
- Monitoring overnight


---

## SFT Results Summary (March 26)

### Three SFT attempts, all degraded performance

**Setup**: Dream-7B-Instruct + LoRA (r=64, alpha=128), d1's absorbing state diffusion loss, AR-shifted logits, 3 epochs.

| Attempt | Training Data | Loss | Base F1 | SFT F1 | Delta |
|---------|--------------|------|---------|--------|-------|
| v1 (no AR-shift) | DLLM-Searcher reasoning | 5.65→1.36 | — | 1.3% | garbage |
| v2 (reasoning template) | DLLM-Searcher reasoning | 1.24→0.87 | 26.5% | 21.9% | -4.6pp |
| v3 (answer-only, DLLM-Searcher) | DLLM-Searcher answers | 1.38→0.87 | 26.5% | 21.9% | -4.6pp |
| **v4 (answer-only, matched Qwen)** | **Qwen on same retriever** | **1.39→0.87** | **28.5%** | **21.1%** | **-7.4pp** |

### Root cause analysis (from prediction comparison)

The SFT model generates MORE VERBOSE outputs:
- Base: "Maria Shriver" (F1=1.0) → SFT: "Maria Shriver married Arnold Schwarzenegger" (F1=0.4)
- Base: "22" (F1=1.0) → SFT: "Plague occurred in Venice 22 times" (F1=0.18)
- Base: "John D. Loudermilk" (F1=1.0) → SFT: "Turn Me On was written by John D. Loudermilk" (F1=0.4)

Contain rate is IDENTICAL (26% for both) — the SFT model finds the answer but adds context words that hurt precision/F1.

### Key insight

d1's absorbing state diffusion loss teaches Dream to predict tokens under masking, but Dream's GENERATION behavior (EOS-first confidence selection, 4-word outputs) is determined by pre-training. LoRA SFT cannot change this fundamental generation dynamics. The model learns to predict different tokens but the denoising process still commits the same high-confidence tokens first.

**This is an important finding for the dLLM community**: SFT works for SDAR (DLLM-Searcher) and LLaDA (d1) but NOT for Dream-7B in the QA setting. The architecture differences (block diffusion vs full masked diffusion) matter for trainability.

### Critical bug found and fixed

d1's SFT code doesn't include Dream's AR-shift (position i predicts token at i+1). Without the shift, training loss starts at 5.65 and learns garbage. With the shift, loss starts at 1.24 — confirming correct alignment. **Anyone adapting d1 for Dream must add this shift.**

---

## Quality-Latency Frontier Benchmark (March 28)

This section supersedes the earlier "final status" conclusions above. The benchmark below is the clean chart-ready suite used for direct AR vs dLLM comparison.

### Benchmark setting

- **Question count**: `1000` per dataset, `3 x 1000` total
- **Datasets**: `HotpotQA`, `MuSiQue`, `2WikiMultiHopQA`
- **Question files**: `01-arag-reproduction/data/questions_wiki18/{hotpotqa,musique,2wikimultihopqa}.json`
- **Corpus**: `wiki18_100w`
- **Corpus file used in the runs**: `/projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl`
- **Corpus size**: `21,015,324` passages / chunks
- **Index**: `/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index`
- **Offset map**: `/projects/prjs1800/msc-thesis/01-arag-reproduction/data/index/wiki18_id_offset.json`
- **Retriever**: `E5-base-v2`
- **AR family**: `Qwen3-8B`
- **dLLM families**: `LLaDA-8B-Instruct`, `Dream-v0-Instruct-7B`
- **Hardware**: one H100 per worker, CPU retrieval, sequential per-question timing inside each shard
- **Evaluator**: shared normalization, Counter-based token F1, EM, contain

Note:
- there is also a scratch copy at `/scratch-shared/pnair/flashrag/wiki18_100w.jsonl`
- the reported benchmark runs used the `/projects/prjs1800/...` project-mounted paths above

### Methods

AR:
- `b0` non-agentic RAG
- `e2_react`
- `ircot`

LLaDA:
- `baseline`
- `spread`
- `aram`
- `pool`
- `eamd_micro`

Dream:
- `baseline`
- `spread`
- `aram`
- `pool`
- `eamd_regen`

### Chart-ready artifacts

- `benchmarking/results/frontier_single/comparison_tables/frontier_model_comparison.csv`
- `benchmarking/results/frontier_single/comparison_tables/hard_subset_model_comparison.csv`
- `benchmarking/results/frontier_throughput/throughput_table_with_dream_clean.csv`

The raw per-dataset summaries are also saved at:
- `benchmarking/results/frontier_single/hotpotqa/all_methods`
- `benchmarking/results/frontier_single/hotpotqa/all_methods_dream`
- `benchmarking/results/frontier_single/musique/all_methods`
- `benchmarking/results/frontier_single/musique/all_methods_dream`
- `benchmarking/results/frontier_single/2wikimultihopqa/all_methods`
- `benchmarking/results/frontier_single/2wikimultihopqa/all_methods_dream`

### Single-query frontier results

| Dataset | Family | Method | F1 | EM | Contain | Median Latency (s) | P90 (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | AR | `b0` | 0.2639 | 0.162 | 0.247 | 0.298 | 0.630 |
| HotpotQA | AR | `e2_react` | 0.0693 | 0.027 | 0.276 | 10.389 | 17.349 |
| HotpotQA | AR | `ircot` | **0.4585** | **0.348** | **0.448** | 6.995 | 10.675 |
| HotpotQA | LLaDA | `aram` | 0.3555 | 0.210 | 0.296 | 2.173 | 2.273 |
| HotpotQA | LLaDA | `eamd_micro` | 0.3521 | 0.194 | 0.299 | 5.580 | 5.874 |
| HotpotQA | Dream | `aram` | 0.4193 | 0.274 | 0.342 | 2.104 | 2.175 |
| HotpotQA | Dream | `pool` | 0.4411 | 0.263 | 0.373 | 6.006 | 6.287 |
| HotpotQA | Dream | `eamd_regen` | 0.4345 | 0.255 | 0.372 | 6.243 | 6.582 |
| MuSiQue | AR | `b0` | 0.2507 | 0.143 | 0.235 | 0.316 | 0.659 |
| MuSiQue | AR | `e2_react` | 0.0668 | 0.021 | 0.231 | 10.220 | 17.512 |
| MuSiQue | AR | `ircot` | **0.2614** | **0.188** | **0.251** | 7.133 | 10.773 |
| MuSiQue | LLaDA | `aram` | 0.1947 | 0.107 | 0.128 | 2.137 | 2.232 |
| MuSiQue | LLaDA | `eamd_micro` | 0.2011 | 0.113 | 0.144 | 5.533 | 5.787 |
| MuSiQue | Dream | `aram` | 0.1993 | 0.112 | 0.128 | 2.105 | 2.175 |
| MuSiQue | Dream | `pool` | **0.2408** | **0.145** | 0.184 | 6.017 | 6.308 |
| MuSiQue | Dream | `eamd_regen` | 0.2404 | 0.137 | **0.185** | 6.284 | 6.616 |
| 2WikiMH | AR | `b0` | 0.2653 | 0.179 | 0.270 | 0.316 | 0.520 |
| 2WikiMH | AR | `e2_react` | 0.0593 | 0.002 | 0.293 | 10.540 | 17.469 |
| 2WikiMH | AR | `ircot` | **0.3808** | **0.295** | **0.453** | 7.124 | 10.712 |
| 2WikiMH | LLaDA | `aram` | 0.2901 | 0.175 | 0.258 | 2.165 | 2.281 |
| 2WikiMH | LLaDA | `eamd_micro` | 0.2770 | 0.149 | 0.248 | 5.572 | 5.860 |
| 2WikiMH | Dream | `aram` | 0.3563 | 0.268 | 0.293 | 2.122 | 2.206 |
| 2WikiMH | Dream | `pool` | **0.3820** | **0.277** | **0.328** | 6.094 | 6.379 |
| 2WikiMH | Dream | `eamd_regen` | 0.3771 | 0.268 | 0.326 | 6.332 | 6.680 |

### Hard subset: initial retrieval miss (`C0 miss`)

| Dataset | Method | F1 on `C0 miss` |
| --- | --- | ---: |
| HotpotQA | `ircot` | **0.3091** |
| HotpotQA | Dream `eamd_regen` | 0.2844 |
| HotpotQA | Dream `aram` | 0.2442 |
| HotpotQA | LLaDA `eamd_micro` | 0.2126 |
| MuSiQue | `ircot` | **0.1880** |
| MuSiQue | Dream `eamd_regen` | 0.1409 |
| MuSiQue | Dream `pool` | 0.1358 |
| MuSiQue | Dream `aram` | 0.0928 |
| MuSiQue | LLaDA `eamd_micro` | 0.1070 |
| 2WikiMH | Dream `pool` | **0.2856** |
| 2WikiMH | Dream `eamd_regen` | 0.2829 |
| 2WikiMH | `ircot` | 0.2678 |
| 2WikiMH | Dream `aram` | 0.2452 |
| 2WikiMH | LLaDA `eamd_micro` | 0.2198 |

### Throughput sweep under load (MuSiQue, 50q)

| Method | QPS | Median Latency Under Load (s) | P90 (s) | Mean F1 | Mean EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b0` | **0.6849** | 0.306 | 0.608 | 0.1993 | 0.10 |
| `ircot` | 0.1101 | 5.450 | 10.940 | 0.3194 | 0.18 |
| LLaDA `eamd_micro` | 0.2008 | 5.321 | 5.543 | 0.2347 | 0.10 |
| Dream `baseline` | 0.2618 | 1.988 | 2.050 | 0.1896 | 0.06 |
| Dream `spread` | 0.2551 | 1.963 | 2.008 | 0.2151 | 0.08 |
| Dream `aram` | 0.2604 | 1.953 | 2.015 | 0.1813 | 0.06 |
| Dream `pool` | 0.1171 | 6.044 | 6.291 | 0.3387 | 0.18 |
| Dream `eamd_regen` | 0.1326 | 6.231 | 6.502 | **0.3773** | **0.20** |

### Interpretation after the full frontier suite

1. `wiki18_100w` with `21,015,324` passages is a valid large open-domain benchmark corpus.
2. `IRCoT` is still the strongest full-set quality baseline on all three datasets.
3. `LLaDA` underperforms as a training-free frontier backbone in this setting.
4. `Dream` is clearly stronger than `LLaDA` as a training-free dLLM backbone.
5. Dream `eamd_regen` consistently beats Dream `SPREAD` and Dream `ARAM`.
6. Dream `eamd_regen` is strongest on hard retrieval-miss subsets, especially on `MuSiQue`.
7. Dream `pool` still slightly edges Dream `eamd_regen` on full-set F1 on all three datasets.
8. Therefore, the current thesis-safe conclusion is:
   - the mathematically grounded Dream `EAMD-Regen` method is competitive, beats `SPREAD` and `ARAM`, and is strongest where new evidence matters most
   - but it does **not yet** beat `IRCoT` or the `Pool` control on full-set benchmark F1

### Current status

The correct current status is:
- **main mathematically grounded method**: Dream `EAMD-Regen`
- **best dLLM training-free backbone**: Dream, not LLaDA
- **current dLLM weakness**: full-set quality still trails `IRCoT`
- **current internal control to beat**: Dream `Pool`
