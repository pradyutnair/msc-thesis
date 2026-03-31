# DNMR Experiment Checklist

**Last updated**: 2026-03-31
**Goal**: Complete all experiments for EMNLP paper by end of day

## Code Fixes

- [ ] **Fix LLaDA decode in idnmr_pilot.py**: 32 tokens, 32 steps, neg_entropy=False when model=llada
- [ ] **Fix LLaDA decode in baselines_1k.py**: same settings
- [ ] **Speed: batch extraction rollouts**: batch 3 candidate rollouts in one forward pass (both models)
- [ ] **Speed: reduce extraction steps**: 12 → 4 steps for candidate completion (both models)

## Config Per Model

| Setting | Dream-7B | LLaDA-8B | Qwen3-8B (AR) |
|---------|----------|----------|---------------|
| Steps | 32 | 32 | N/A |
| Answer tokens | 32 | 32 | 32 |
| neg_entropy | True | **False** | N/A |
| Unmasking | entropy | low-confidence | N/A |
| Prompt | few-shot short | zero-shot short | few-shot short |
| Extractor | mixed (pos0+entropy) | mixed (pos0+entropy) | sampling |
| Extraction steps | 12 (→4 after speedup) | 12 (→4 after speedup) | N/A |

## Experiments

### DONE ✅

| # | Experiment | Model | Datasets | Questions | Results Dir |
|---|-----------|-------|----------|-----------|-------------|
| 1 | Dream DNMR (mixed extractor) | Dream-7B | Mus/Hot/2Wi | 1000×3 | results/mixed/dream_* |
| 2 | Dream SPREAD/ARAM baselines | Dream-7B | Mus/Hot/2Wi | 1000×3 | results/baselines/dream_* |
| 3 | AR MQR (no thinking) | Qwen3-8B | Mus/Hot/2Wi | 1000×3 | results/ar_mqr/qwen3_* |
| 4 | Dream DNMR (old pos-0, ablation) | Dream-7B | Mus/Hot/2Wi | 1000×3 | results/idnmr/dream_* |

### Dream Results (mixed extractor, 1000q)

| Method | MuSiQue F1 | HotpotQA F1 | 2WikiMH F1 |
|--------|:----------:|:-----------:|:----------:|
| baseline | 0.227 | 0.476 | 0.330 |
| SPREAD | 0.213 | 0.461 | 0.307 |
| ARAM | 0.225 | 0.484 | 0.338 |
| AR-MQR (Qwen3) | 0.207 | 0.455 | 0.295 |
| iPool | 0.254 | 0.500 | 0.340 |
| iARAM | 0.263 | 0.504 | 0.346 |
| **DNMR (pool)** | **0.276** | **0.509** | **0.353** |

### TODO 🔲

| # | Experiment | Model | Datasets | Questions | Blocked By |
|---|-----------|-------|----------|-----------|------------|
| 5 | LLaDA DNMR (mixed, fixed settings) | LLaDA-8B | Mus/Hot/2Wi | 1000×3 | Code fix #1 + speed fix |
| 6 | LLaDA SPREAD/ARAM baselines | LLaDA-8B | Mus/Hot/2Wi | 1000×3 | Code fix #2 + speed fix |
| 7 | Significance tests (all final) | All | All | - | After #5 + #6 |
| 8 | Bootstrap CIs (all final) | All | All | - | After #5 + #6 |
| 9 | Efficiency metrics (wall-clock, fwd passes) | All | All | - | After #5 + #6 |

### Validated ✅ (10q smoke)

- LLaDA with 32/32/neg_entropy=False: F1=0.418, EM=0.300 on 10q HotpotQA
  - Matches ARAM paper's reported LLaDA performance
  - Key fixes: 32 tokens (was 8), neg_entropy=False (was True)

## Paper Structure

- Main table: Dream DNMR vs baselines (SPREAD/ARAM/AR-MQR/iPool)
- Cross-architecture: LLaDA DNMR vs LLaDA baselines
- Mechanism ablation: iDNMR vs iPool (distribution vs answer-conditioned)
- Extractor ablation: mixed vs pos-0 (Dream only)
- Efficiency table: wall-clock, forward passes, retrieval queries
- Math: IDNMR_FORMALIZATION.md (760 lines, reviewed)

## Key Files

- `src/daes/idnmr_pilot.py` — Main DNMR runner (baseline/pool/ipool/idnmr)
- `src/daes/baselines_1k.py` — SPREAD/ARAM/iSPREAD/iARAM runner
- `src/daes/ar_mqr_pilot.py` — AR multi-query baseline
- `src/daes/eamd_v2_wiki18.py` — Shared utils, extractor, prompts
- `src/daes/significance_tests.py` — Bootstrap tests
- `src/daes/retrieval_metrics.py` — CIs and efficiency
- `docs/IDNMR_FORMALIZATION.md` — Math formalization
