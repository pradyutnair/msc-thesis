# DNMR Experiment Checklist

**Last updated**: 2026-04-01 16:00
**Goal**: Complete all experiments for EMNLP paper (deadline: 2026-05-25)
**Status**: CRITICAL — DNMR works on Dream but fails on LLaDA. Seeking unified method.

---

## Phase 1: Code Fixes — DONE

- [x] Fix neg_entropy in eamd_v2_wiki18.py (9 instances)
- [x] Fix neg_entropy in idnmr_pilot.py (1 instance)
- [x] Fix neg_entropy in eamd_iterative.py (2 instances)
- [x] Fix neg_entropy in eamd_wiki18_full_llada.py (8 instances)
- [x] baselines_1k.py — No direct changes needed
- [x] Configurable extraction steps — `--extraction_steps` CLI param
- [x] Batch extraction — Implemented then REVERTED (slower, not faster)
- [x] Extraction prompt fix — `build_short_prompt` in agnostic extractor (made LLaDA worse)

## Phase 2: Smoke Tests — DONE

- [x] Dream 10q (sequential, 12-step): F1=0.706, 96s/q — OK
- [x] Dream 10q (sequential, 4-step): F1=0.656, 74s/q — 23% faster, slight quality drop
- [x] LLaDA 10q (neg_entropy fix): baseline F1=0.424, EM=0.300 — matches ARAM paper
- [x] LLaDA baselines 10q (SPREAD/ARAM): SPREAD=0.606, ARAM=0.600 — working
- [x] Dream EAMD 20q at 32/32: pool=0.514, eamd_regen=0.514, eamd_remask=0.479
- [x] LLaDA EAMD-Remask 20q at 32/32: remask=0.404 vs ARAM=0.372 (won't hold at scale)

## Phase 3: Full Runs — PARTIALLY DONE

- [x] LLaDA DNMR 1000qx3 (32/32, neg_entropy=False) — COMPLETED but DNMR HURTS LLaDA
      Results: baseline=0.230, pool=0.194, ARAM=0.293 (ARAM best, DNMR worst)
      Path: results/mixed/llada_{dataset}_mix_s{0-4}.json (incomplete: 740-890q, hit 6h limit)
- [x] LLaDA baselines 1000qx3 (SPREAD/ARAM/iSPREAD/iARAM) — COMPLETE
      Results: ARAM=0.293, SPREAD=0.269, iARAM=0.244, iSPREAD=0.220
      Path: results/baselines/llada_{dataset}_baselines_s{0-4}.json
- [ ] LLaDA DNMR continuation (remaining ~200q per dataset) — NOT NEEDED given negative results

## Phase 4: LLaDA Fix Attempts — ALL FAILED

- [x] DNMR expand (standard): F1 -0.047 vs baseline. Context pollution.
- [x] DNMR replace (fixed budget rerank): F1 -0.008 vs baseline. Passages not better.
- [x] Extraction prompt fix (build_short_prompt): Made pool WORSE (0.336→0.249).
- [x] Temporal DNMR (extract at step tau=4): F1 -0.110 vs baseline. Candidates still bad.
- [x] CST v2 (counterfactual score transport): F1 +0.000 on Dream, -0.003 on LLaDA. Signal too weak.
- [x] iARAM/iSPREAD/iPool: All worse than single-query ARAM on LLaDA.
- [x] EAMD-Remask at 32/32: +0.032 over ARAM at 20q but doesn't beat DNMR on Dream (inverse problem).

## Phase 5: Diagnostics — DONE

- [x] Bridge extraction diagnostic (Dream vs LLaDA, 5q):
      Dream pos-0: diverse (0.05-0.40 per branch). LLaDA pos-0: peaked (0.99+).
      Path: results/diag_dream_5q.json, results/diag_llada_5q.json
- [x] Temporal distribution diagnostic (steps 0,2,4,8,16,32):
      LLaDA peaked at step 0, diversity at step 4 (but function words, not entities).
      Dream diverse throughout all steps.
      Path: results/diag_temporal_dream_5q.json, results/diag_temporal_llada_5q.json
- [x] Contain metric analysis:
      LLaDA contain RISES with expansion (+0.097) but F1 DROPS (-0.047).
      Right passages retrieved, model can't decode from expanded context.

## Root Cause Analysis

LLaDA fails because:
1. Context pollution: any form of expansion/replacement hurts LLaDA decoding
2. Peaked posterior: LLaDA pos-0 is 0.99+ concentrated, no diversity for bridge extraction
3. Conditioning instability: changing context shifts the score field globally, answer-start token falls into wrong lexical basin
4. Not fixable at inference time: CST logit shifts too small to overcome the instability

ARAM works on both because it NEVER modifies context (guided decoding on same passages).

---

## Valid Results (organized in results_organized/)

### Dream (32/32, all valid)

| Experiment | Method | MuSiQue | HotpotQA | 2WikiMH | Mean F1 |
|-----------|--------|:-------:|:--------:|:-------:|:-------:|
| dnmr_mixed | baseline | 0.227 | 0.476 | 0.330 | 0.344 |
| dnmr_mixed | pool (DNMR) | 0.276 | 0.509 | 0.353 | 0.379 |
| dnmr_mixed | ipool | 0.254 | 0.500 | 0.340 | 0.364 |
| dnmr_mixed | idnmr | 0.274 | 0.518 | 0.346 | 0.379 |
| dllm_baselines | SPREAD | 0.213 | 0.461 | 0.307 | 0.327 |
| dllm_baselines | ARAM | 0.225 | 0.484 | 0.338 | 0.349 |
| dllm_baselines | iARAM | 0.263 | 0.504 | 0.346 | 0.371 |
| ar_mqr | AR-MQR | 0.207 | 0.455 | 0.295 | 0.319 |

### LLaDA (32/32, valid but negative for DNMR)

| Experiment | Method | MuSiQue | HotpotQA | 2WikiMH | Mean F1 |
|-----------|--------|:-------:|:--------:|:-------:|:-------:|
| dnmr_mixed | baseline | 0.145 | 0.365 | 0.181 | 0.230 |
| dnmr_mixed | pool (DNMR) | 0.107 | 0.318 | 0.157 | 0.194 |
| dllm_baselines | SPREAD | 0.170 | 0.407 | 0.231 | 0.269 |
| dllm_baselines | ARAM | 0.200 | 0.425 | 0.255 | 0.293 |
| dllm_baselines | iARAM | 0.143 | 0.376 | 0.212 | 0.244 |

### Pilots (results_organized/_pilots/)

| Pilot | Model | Result |
|-------|-------|--------|
| EAMD-Remask 20q 32/32 | LLaDA | F1=0.404 vs ARAM=0.372 (+0.032) |
| EAMD-Remask 20q 32/32 | Dream | F1=0.479 vs pool=0.514 (-0.035) |
| DNMR-Replace 20q | LLaDA | F1=0.343 vs baseline=0.351 (-0.008) |
| Temporal DNMR 20q tau=4 | LLaDA | F1=0.241 vs baseline=0.351 (-0.110) |
| CST v2 50q | Dream | F1=0.517 vs baseline=0.517 (+0.000) |
| CST v2 50q | LLaDA | F1=0.268 vs baseline=0.272 (-0.003) |

---

## Open Questions

1. Is SFT/RL the path to make DNMR work on LLaDA? Oracle bridge has 2x headroom.
2. Can we scope the paper to Dream-only + architectural analysis?
3. Is there another training-free approach we haven't considered?

## Key Files

| File | Purpose |
|------|---------|
| src/daes/idnmr_pilot.py | Main DNMR runner |
| src/daes/baselines_1k.py | SPREAD/ARAM runner |
| src/daes/eamd_v2_wiki18.py | Shared utils, extractor |
| src/daes/eamd_wiki18_full_llada.py | EAMD full suite (Dream+LLaDA) |
| src/daes/cst_pilot.py | CST v2 implementation |
| src/daes/temporal_dnmr_pilot.py | Temporal DNMR |
| src/daes/dnmr_replace_pilot.py | Fixed-budget replacement |
| src/daes/diagnose_extraction.py | Bridge extraction diagnostic |
| src/daes/diagnose_temporal.py | Temporal distribution diagnostic |
| docs/IDNMR_FORMALIZATION.md | Math formalization (760 lines) |
| docs/DNMR_FINAL_RESULTS.md | Dream results + significance |
