# DNMR Experiment Log

**Single source of truth. Updated continuously.**
**Last updated**: April 2, 2026 17:00 CEST

---

## Status Checklist

### Done
- [x] Dream 1000q x 3 datasets: DNMR beats all baselines (p<0.001)
- [x] LLaDA 1000q x 3 datasets: DNMR pool F1 below baseline (verbosity issue)
- [x] LLaDA baselines 1000q: ARAM=0.293, SPREAD=0.269
- [x] Oracle bridge (10q LLaDA): +7.4pp, proves model CAN use good evidence
- [x] Retrieval analysis (740q): pool finds gold in 81 extra Qs where baseline cannot
- [x] Contain analysis: pool contain=22.3% vs ARAM=11.4% on LLaDA
- [x] Root cause: verbosity (110 chars pool vs 29 chars ARAM), not retrieval failure
- [x] Pipeline ablation 2x2 (10q): query prefix essential for Dream, verbosity confirmed for LLaDA
- [x] Diagnostics: remasking, logit lens, PAQCD, ABRD — all dead ends for extraction
- [x] **Verbosity fix pilot (50q LLaDA): pool_8 F1=0.194, matches ARAM, +5.6pp over baseline**
- [x] IVI node410 setup: 3xA6000, working env, ~10s/q for LLaDA

### In Progress
- [ ] Dream verbosity fix confirmation (expect pool_8 to stay positive)
- [ ] Update idnmr_pilot.py with adaptive n_tokens for final decode

### TODO: Paper-Critical
- [ ] LLaDA 1000q x 3 datasets with n_tokens=8 final decode (THE key rerun)
- [ ] Dream 1000q x 3 datasets with n_tokens=8 final decode (confirm no regression)
- [ ] Fair efficiency benchmark: fast-dLLM + FLOPs measurement
- [ ] LLM judge eval on existing predictions (semantic accuracy beyond F1)
- [ ] Statistical significance tests on new results
- [ ] Write paper draft

### TODO: Nice-to-Have
- [ ] HotpotQA + 2WikiMH 50q pilots on IVI before full 1000q
- [ ] Sweep n_tokens (6, 8, 10, 12) at larger scale
- [ ] Dream on IVI for comparison runs

---

## 1. Main Results (1000q x 3 datasets, F1)

### Dream-7B

| Method | MuSiQue | HotpotQA | 2WikiMH | Mean |
|--------|:-------:|:--------:|:-------:|:----:|
| Baseline | 0.227 | 0.476 | 0.330 | 0.344 |
| SPREAD | 0.213 | 0.461 | 0.307 | 0.327 |
| ARAM | 0.225 | 0.484 | 0.338 | 0.349 |
| iARAM | 0.263 | 0.504 | 0.346 | 0.371 |
| **DNMR (pool)** | **0.276** | **0.509** | **0.353** | **0.379** |
| iDNMR | 0.274 | 0.518 | 0.346 | 0.379 |

DNMR beats all baselines on Dream (p<0.001).

### LLaDA-8B-Instruct (OLD — n_tokens=32, before verbosity fix)

| Method | MuSiQue | HotpotQA | 2WikiMH | Mean |
|--------|:-------:|:--------:|:-------:|:----:|
| Baseline | 0.144 | 0.365 | 0.181 | 0.230 |
| SPREAD | 0.170 | 0.407 | 0.231 | 0.269 |
| ARAM | 0.200 | 0.425 | 0.255 | 0.293 |
| DNMR pool (n=32) | 0.107 | 0.318 | 0.157 | 0.194 |

These LLaDA DNMR numbers are with n_tokens=32 (verbose answers). Needs rerun with n_tokens=8.

---

## 2. Verbosity Fix (April 2, 2026) — GO SIGNAL

### LLaDA 50q MuSiQue on IVI A6000

| Method | F1 | Contain | Avg Len | Delta |
|--------|:--:|:-------:|:-------:|:-----:|
| baseline (32 tok, C0) | 0.138 | 4.0% | 15.7 | — |
| **pool_8 (8 tok, C1)** | **0.194** | **6.0%** | **22.6** | **+5.6pp** |
| pool_12 (12 tok, C1) | 0.157 | 8.0% | 24.0 | +1.9pp |
| pool_16 (16 tok, C1) | 0.146 | 6.0% | 29.5 | +0.8pp |
| pool_32 (32 tok, C1) | 0.173 | 10.0% | 27.1 | +3.5pp |

**pool_8 matches ARAM (0.194 vs 0.191 at 1000q).** Shorter canvas forces concise answers. dLLMs condition content on canvas length — this is diffusion-native.

Key insight: the retrieval works (pool finds more gold answers). The fix is not in retrieval or extraction — it is in how many tokens the model is allowed to produce for the final answer.

---

## 3. Retrieval Analysis (LLaDA MuSiQue 740q)

Pool retrieval genuinely helps. ARAM wins F1 only because of concise answers.

| Metric | Pool (DNMR) | ARAM |
|--------|:-----------:|:----:|
| F1 | 0.107 | 0.191 |
| Contain | 22.3% | 11.4% |
| Avg length | 110 chars | 29 chars |
| Per-Q F1 wins | 145 | 139 |
| Finds gold extra | 90 | 9 |

### Does retrieval add new information?

| Category | Count | Pct |
|----------|:-----:|:---:|
| ONLY pool finds gold | 81 | 10.9% |
| ONLY baseline finds gold | 9 | 1.2% |
| Both | 84 | 11.4% |
| Neither | 566 | 76.5% |

---

## 4. Oracle Bridge (10q MuSiQue)

| Method | Dream F1 | LLaDA F1 |
|--------|:--------:|:--------:|
| Baseline | 0.081 | 0.160 |
| Pool | 0.124 | 0.099 |
| Oracle bridge | 0.267 | 0.234 |

Both models benefit from perfect bridges.

---

## 5. Diagnostics (all dead ends)

| Diagnostic | Result |
|-----------|--------|
| Conditional remasking | 0/169 more diverse |
| Logit lens | 0 bridge hits |
| PAQCD query gen | Dream +3.8pp, LLaDA -1.1pp |
| ABRD TAPS+P2 | No effect |
| EAMD-Remask 50q | Did not hold from 20q |
| Pipeline 2x2 ablation | Prefix helps Dream, verbosity confirmed for LLaDA |

---

## 6. Efficiency (needs fair benchmark)

| Method | Model | Optimization | Latency/q |
|--------|-------|:------------:|:---------:|
| DNMR pool | Dream-7B | Vanilla PyTorch | ~5.4s (Snellius H100) |
| DNMR pool | LLaDA-8B | Vanilla PyTorch | ~10.5s (IVI A6000) |
| IRCoT | Qwen3-8B | vLLM + KV cache | ~7.0s |

Not apples-to-apples. fast-dLLM benchmark pending.

---

## 7. Key Files

| File | Purpose |
|------|---------|
| src/daes/idnmr_pilot.py | Main DNMR runner |
| src/daes/baselines_1k.py | SPREAD/ARAM baselines |
| src/daes/eamd_v2_wiki18.py | Shared utils, extraction, prompts |
| src/daes/verbosity_fix_pilot.py | n_tokens ablation pilot |
| src/daes/oracle_bridge.py | Oracle bridge experiment |
| src/daes/seed_ablation.py | 2x2 pipeline ablation |
| scripts/ivi/run.sh | IVI node410 experiment runner |
| docs/IDNMR_FORMALIZATION.md | 760-line math formalization |
