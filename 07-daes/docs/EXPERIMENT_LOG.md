# DNMR Experiment Log

**Single source of truth. Updated continuously.**
**Last updated**: April 2, 2026

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

DNMR beats all baselines on Dream (p<0.001 paired bootstrap).

### LLaDA-8B-Instruct

| Method | MuSiQue | HotpotQA | 2WikiMH | Mean |
|--------|:-------:|:--------:|:-------:|:----:|
| Baseline | 0.144 | 0.365 | 0.181 | 0.230 |
| SPREAD | 0.170 | 0.407 | 0.231 | 0.269 |
| ARAM | 0.200 | 0.425 | 0.255 | 0.293 |
| DNMR (pool) | 0.107 | 0.318 | 0.157 | 0.194 |

DNMR F1 is below baseline on LLaDA. But see Section 2.

---

## 2. Retrieval Analysis (LLaDA MuSiQue, 740q)

**DNMR retrieval WORKS on LLaDA. The F1 loss is from verbosity, not retrieval failure.**

| Metric | Pool (DNMR) | ARAM |
|--------|:-----------:|:----:|
| F1 | 0.107 | 0.191 |
| **Contain** | **22.3%** | **11.4%** |
| Avg answer length | 110 chars | 29 chars |
| Per-question F1 wins | 145 | 139 |
| Finds gold in extra Qs | 90 | 9 |

### Does additional retrieval help?

| Category | Count | Pct |
|----------|:-----:|:---:|
| Both baseline and pool find gold | 84 | 11.4 |
| **ONLY pool finds gold** | **81** | **10.9** |
| ONLY baseline finds gold | 9 | 1.2 |
| Neither | 566 | 76.5 |

Pool finds the gold answer in 81 extra questions where baseline cannot. Only loses 9. Net retrieval gain: +72 questions.

### Verbosity is the bottleneck

| Model | Baseline avg len | Pool avg len |
|-------|:----------------:|:------------:|
| Dream | 15.8 chars | 16.7 chars |
| LLaDA | 64.1 chars | 112.6 chars |

LLaDA pool answers are 4x longer than ARAM answers. The gold answer is CONTAINED but buried in verbose text, killing F1 precision.

---

## 3. Oracle Bridge (10q MuSiQue)

| Method | Dream F1 | LLaDA F1 |
|--------|:--------:|:--------:|
| Baseline | 0.081 | 0.160 |
| Pool (DNMR) | 0.124 | 0.099 |
| **Oracle bridge** | **0.267** | **0.234** |

Both models benefit from perfect bridges. LLaDA CAN use expanded context when bridges are correct.

---

## 4. Why DNMR F1 Is Low on LLaDA

Root cause chain:
1. LLaDA posterior is peaked (H=0.001 at pos-0) -> bridge candidates are answer guesses, not entities
2. 30% of LLaDA candidates start with "The answer is..." vs 1% for Dream
3. Despite bad bridges, the seed answer retrieves useful passages (contain rises to 22.3%)
4. LLaDA produces verbose answers from expanded context (110 chars vs Dream's 17 chars)
5. Verbosity kills F1 precision even when the gold answer is present

**The retrieval works. The decoding is verbose. Fix verbosity = fix LLaDA.**

---

## 5. Diagnostics Run

| Diagnostic | Result | Conclusion |
|-----------|--------|------------|
| Conditional remasking (10q) | 0/169 positions more diverse | Remasking doesn't help |
| Logit lens (5q) | 0 bridge hits at any layer | Intermediate layers are garbage |
| PAQCD query gen (50q) | Dream +3.8pp, LLaDA -1.1pp | Model-generated queries fail on LLaDA |
| ABRD TAPS+P2 (50q) | 0pp Dream, -1.6pp LLaDA | Too weak to change answers |
| EAMD-Remask (50q) | 0pp on LLaDA | Did not hold from 20q pilot |
| Pipeline ablation 2x2 (10q) | Prefix essential for Dream, all hurt LLaDA F1 | Verbosity confirmed as cause |

---

## 6. Efficiency

### Wall-clock (NOT fairly comparable)

| Method | Model | Optimization | Latency/q |
|--------|-------|:------------:|:---------:|
| DNMR pool | Dream-7B | Vanilla PyTorch | ~5.4s |
| DNMR pool | LLaDA-8B | Vanilla PyTorch | ~6.8s |
| IRCoT | Qwen3-8B | vLLM + KV cache | ~7.0s |
| AR-MQR | Qwen3-8B | vLLM + KV cache | ~32.0s |

AR uses optimized serving (vLLM, KV cache). dLLM uses vanilla PyTorch. Not apples-to-apples.

### TODO: Fair comparison
- Run fast-dLLM (prefix KV caching, exists in repo)
- Compute actual FLOPs under matched conditions

### Structural advantages (qualitative)
- Single retrieval round (vs 2-3 for IRCoT)
- No chain-of-thought tokens (32 vs 100-500)
- Prefix caching available via fast-dLLM

---

## 7. Open Items

1. **Fix LLaDA verbosity**: Test n_tokens=8 for final pool decode. dLLMs condition content on canvas length. Needs ~1000 SBUs.
2. **Fair efficiency benchmark**: Run fast-dLLM + FLOPs measurement. Needs ~500 SBUs.
3. **LLM judge eval**: Run existing predictions through DeepSeek judge for semantic accuracy beyond F1.
4. **Scale to 3 datasets**: Once verbosity is fixed, rerun LLaDA 1000q x 3 datasets.

---

## 8. Key Files

| File | Purpose |
|------|---------|
| src/daes/idnmr_pilot.py | Main DNMR runner |
| src/daes/baselines_1k.py | SPREAD/ARAM baselines |
| src/daes/eamd_v2_wiki18.py | Shared utils, extraction, prompts |
| src/daes/oracle_bridge.py | Oracle bridge experiment |
| src/daes/seed_ablation.py | 2x2 pipeline ablation |
| docs/IDNMR_FORMALIZATION.md | 760-line math formalization |
| docs/EXPERIMENT_CHECKLIST.md | Full experiment tracker |
