# DNMR: What We Have, What Works, What Doesn't, and What's Missing

**Date**: April 11, 2026
**Purpose**: Honest assessment of current state and gap analysis for publication.

---

## 1. What We Have Concretely

### Experiments completed (all N=1000 unless noted)


| Experiment                                                           | Models       | Datasets | Status                                      |
| -------------------------------------------------------------------- | ------------ | -------- | ------------------------------------------- |
| Baseline (single-query decode)                                       | Dream, LLaDA | 3        | Done                                        |
| SPREAD                                                               | Dream, LLaDA | 3        | Done                                        |
| ARAM                                                                 | Dream, LLaDA | 3        | Done                                        |
| iSPREAD (iterative)                                                  | Dream, LLaDA | 3        | Done                                        |
| iARAM (iterative)                                                    | Dream, LLaDA | 3        | Done                                        |
| DNMR Pool (1-round)                                                  | Dream, LLaDA | 3        | Done                                        |
| iPool (iterative, answer-cond)                                       | Dream, LLaDA | 3        | Done                                        |
| iDNMR (iterative DNMR)                                               | Dream, LLaDA | 3        | Done                                        |
| iDNMR-2round                                                         | Dream, LLaDA | 3        | Done                                        |
| LLM Judge eval (gpt-4.1-mini)                                        | LLaDA        | 3        | Done                                        |
| Budget ablation (baseline_5 vs baseline_10 vs DNMR)                  | Dream        | MuSiQue  | Done (N=1000)                               |
| Candidate source ablation (dLLM vs Qwen2.5-7B vs Qwen3-8B vs random) | Dream        | MuSiQue  | Done (N=200 for Qwen2.5, N=1000 for others) |
| Oracle bridge                                                        | Dream, LLaDA | MuSiQue  | Done (N=10)                                 |
| IRCoT (Qwen3-8B)                                                     | AR baseline  | 3        | Done (N=1000)                               |
| IRCoT (Qwen2.5-7B)                                                   | AR baseline  | 3        | TO-DO                                       |


### Formalization

760-line document (IDNMR_FORMALIZATION.md) with 7 theorem-backed claims covering:

- Mixed bridge posterior definition
- TopK extraction optimality
- Dominance over answer-conditioned extraction
- Multi-query retrieval coverage bounds
- Filtered evidence selection
- Telescoping answer support
- Finite termination

### Diagnostics completed (all dead ends for improving LLaDA extraction)

Conditional remasking, logit lens, PAQCD, ABRD, CST logit transport, base model extraction, KL extraction, interrupted denoising, temperature sweeps.

---

## 2. What Works

### Dream: strong results

**vs single-query methods (baseline/ARAM/SPREAD):**


| Comparison            | F1 delta | Contain delta | Relative improvement      |
| --------------------- | -------- | ------------- | ------------------------- |
| DNMR Pool vs Baseline | +0.045   | +4.8pp        | +13.1% F1, +18.5% Contain |
| DNMR Pool vs ARAM     | +0.018   | +5.1pp        | +4.6% F1, +19.8% Contain  |


Statistically significant at p<0.001 (paired bootstrap).

**vs matched-budget iterative methods (iSPREAD/iARAM):**


| Comparison | Mean F1 | Mean Contain |
| ---------- | ------- | ------------ |
| DNMR Pool  | 0.389   | 30.8%        |
| iSPREAD    | 0.354   | 28.0%        |
| iARAM      | 0.371   | 27.8%        |


Pool beats both with 1 round/5 queries vs their 2-3 rounds/10 queries.

**vs IRCoT (Qwen3-8B, stronger AR model):**


|              | DNMR Pool (Dream) | IRCoT (Qwen3-8B) |
| ------------ | ----------------- | ---------------- |
| Mean F1      | **0.389**         | 0.367            |
| Mean Contain | 30.8%             | **38.4%**        |
| Passages     | ~7                | ~16              |
| Rounds       | 1                 | ~4               |


Higher F1 with half the passages, but lower Contain.

### LLaDA: works but margins are thin

**DNMR Pool (1 round) matches iterative methods (2-3 rounds):**


|                   | DNMR Pool | iSPREAD   | iARAM     |
| ----------------- | --------- | --------- | --------- |
| Mean Judge%       | 42.9      | 43.0      | 42.1      |
| Mean Ext. Contain | 30.8      | 30.6      | 29.8      |
| Rounds            | 1         | 2.2-2.7   | 2.2-2.6   |
| Queries           | 5         | 10.6-13.1 | 10.3-12.8 |


Same quality, half the queries, one-third the rounds.

**iDNMR is best on all datasets:**


|                   | iDNMR | iSPREAD | iARAM |
| ----------------- | ----- | ------- | ----- |
| Mean Judge%       | 45.5  | 43.0    | 42.1  |
| Mean Ext. Contain | 32.9  | 30.6    | 29.8  |


### Budget ablation confirms gain is from informed retrieval


| Method      | Passages | Contain        |
| ----------- | -------- | -------------- |
| baseline_5  | 5        | 12.1%          |
| baseline_10 | 10       | 12.4% (+0.3pp) |
| DNMR Pool   | ~8       | 19.4% (+7.3pp) |


### Both models benefit from the pipeline

Oracle bridge shows +18.6pp on Dream, +7.4pp on LLaDA. The retrieval expansion framework helps regardless of model.

---

## 3. What Doesn't Work / Weaknesses

### Weakness 1: AR candidates beat dLLM candidates (CRITICAL)

Candidate source ablation (Dream MuSiQue):


| Source                     | F1    | Contain |
| -------------------------- | ----- | ------- |
| Qwen2.5-7B (AR, same base) | 0.312 | 21.5%   |
| Dream-7B (dLLM posterior)  | 0.269 | 18.5%   |
| Random                     | 0.266 | 14.5%   |
| No candidates (baseline)   | 0.250 | 13.0%   |


The same base model in AR mode produces better candidates than its diffusion-trained version. This undermines the "diffusion posterior is uniquely good" claim. A reviewer will say: "The pipeline works, but the dLLM posterior is not the best candidate source. What's diffusion-native about this?"

**Counter-argument available**: dLLM candidates come free from the existing forward pass (no extra model, no sequential sampling). But "convenient" is weaker than "superior."

### Weakness 2: LLaDA gains vs matched-budget methods are marginal


| Comparison                | Relative improvement |
| ------------------------- | -------------------- |
| iDNMR vs iSPREAD (Judge%) | +5.8%                |
| iDNMR vs iARAM (Judge%)   | +8.1%                |
| Pool vs iSPREAD (Judge%)  | -0.2%                |
| Pool vs iARAM (Judge%)    | +1.9%                |


None of these clear 10% relative. On LLaDA, the gains are real but small.

### Weakness 3: Contain gap vs IRCoT


| Dataset  | DNMR Pool | IRCoT (Qwen3-8B) |
| -------- | --------- | ---------------- |
| MuSiQue  | 18.0%     | **25.1%**        |
| HotpotQA | 42.8%     | **44.8%**        |
| 2WikiMH  | 31.5%     | **45.3%**        |


IRCoT finds the gold answer more often on every dataset. The F1 advantage comes from Dream's concise generation (high precision), not from finding more answers. IRCoT uses ~16 passages vs ~7, which partly explains this.

### Weakness 4: LLaDA raw F1 is misleading, and the fix is unvalidated at scale

LLaDA DNMR produces verbose answers (110 chars vs 29 chars for ARAM), killing precision/F1. The pool_8 verbosity fix showed promise at 50q but has not been run at 1000q. Judge eval and extracted F1 are the honest metrics, but reviewers may question why raw F1 is so different.

### Weakness 5: Missing wall-clock numbers for Pool

The efficiency tables show wall-clock for iSPREAD (17-28s) and iARAM (21-38s) but Pool shows "-". The ~5.4s figure for Dream Pool on H100 is from infrastructure notes but not formally benchmarked. The efficiency story needs real numbers.

### Weakness 6: LLaDA posterior peakedness limits the method

LLaDA's posterior entropy is H=0.001. Bridge candidates are 30% "The answer is..." templates. Every extraction variant (standard, KL, base model) converges to ~22-23% contain at 1000q. The method works on LLaDA but the diffusion-native extraction advantage is Dream-specific.

---

## 4. The 10% Relative Threshold

The user wants at least 10% relative improvement on each metric, or a clear efficiency story.

### Dream: vs baseline/ARAM (clears the bar)


| Comparison        | F1 relative | Contain relative |
| ----------------- | ----------- | ---------------- |
| Pool vs Baseline  | +13.1%      | +18.5%           |
| Pool vs ARAM      | +4.6%       | +19.8%           |
| iDNMR vs Baseline | +13.7%      | +24.2%           |


### Dream: vs matched-budget iterative methods (does NOT clear the bar)


| Comparison       | F1 relative | Contain relative |
| ---------------- | ----------- | ---------------- |
| Pool vs iSPREAD  | +9.9%       | +10.0%           |
| Pool vs iARAM    | +4.9%       | +10.8%           |
| iDNMR vs iSPREAD | +10.5%      | +15.4%           |
| iDNMR vs iARAM   | +5.4%       | +16.2%           |


Contain clears 10% relative. F1 is borderline against iSPREAD, fails against iARAM.

### LLaDA: vs matched-budget methods (does NOT clear the bar)


| Comparison       | Judge% relative | Ext. Contain relative |
| ---------------- | --------------- | --------------------- |
| Pool vs iSPREAD  | -0.2%           | +0.7%                 |
| Pool vs iARAM    | +1.9%           | +3.4%                 |
| iDNMR vs iSPREAD | +5.8%           | +7.5%                 |
| iDNMR vs iARAM   | +8.1%           | +10.4%                |


iDNMR vs iARAM on Ext. Contain barely clears. Everything else does not.

### Efficiency: could clear the bar if measured


| Metric                    | Pool    | iSPREAD | iARAM  | Pool advantage |
| ------------------------- | ------- | ------- | ------ | -------------- |
| Rounds                    | 1       | 2-3     | 2-3    | 2-3x fewer     |
| Queries                   | 5       | 10-13   | 10-13  | 2-2.6x fewer   |
| Forward passes            | ~70     | 82-107  | 80-103 | ~1.3x fewer    |
| Wall-clock (Dream, H100)  | ~5.4s*  | 17-21s  | 21-28s | 3-5x faster*   |
| Wall-clock (LLaDA, A6000) | ~10.5s* | 22-28s  | 29-38s | 2-3x faster*   |


*Not formally benchmarked. Needs proper measurement.

---

## 5. Goal

Leverage dLLM-native capabilities for multi-hop QA RAG in a way that is:

- Completely novel
- Mathematically formalized
- Works well in performance or efficiency or both
- Training-free first, SFT/RL later
- Without gaps that make easy reviewer kill-shots
- At least 10% relative improvement, or a clear efficiency story

---

## 6. Gaps Between Current State and Goal

### Gap 1: The "diffusion-native" claim has a hole

AR candidates from the same base model beat dLLM candidates. The pipeline itself is the contribution, not the candidate source. Need either: (a) a fair comparison where dLLM wins (Qwen2.5-7B IRCoT running now), (b) a reframing where "free candidates from existing forward pass" is the novelty, or (c) a new extraction method that actually beats AR.

### Gap 2: Performance threshold not met against matched-budget iterative methods

10% relative is only met on some metric/comparison pairs, not consistently. Options: (a) increase k to get more candidates and passages, (b) lean on the efficiency story instead, (c) find a stronger extraction method.

### Gap 3: No formal wall-clock benchmarks

The efficiency story is the strongest angle but has no proper numbers. Need: timed runs of Pool, iSPREAD, iARAM, IRCoT all on the same hardware with the same framework.

### Gap 4: Qwen2.5-7B IRCoT comparison is missing

This is the fair cross-paradigm comparison. Currently running. If Dream DNMR Pool beats Qwen2.5-7B IRCoT on Contain (not just F1), that could be the headline result.

### Gap 5: LLaDA verbosity fix unvalidated at scale

Pool_8 (n_tokens=8) showed promise at 50q but needs 1000q run to confirm. Without this, LLaDA raw F1 numbers are misleading and reviewers will flag it.

### Gap 6: Contain gap vs IRCoT unexplained

Need to determine if the gap is due to (a) fewer passages (fixable by increasing k), (b) weaker candidates (the AR ablation suggests this), or (c) dLLM generation quality (rephrasing, special tokens). Yijia's suggestion to check F1>0 but Contain=0 cases would help diagnose this.

### Gap 7: No per-hop analysis

No breakdown of 2-hop vs 3-hop vs 4-hop performance. Reviewers at EMNLP will want this for a multi-hop QA paper.

---

## 7. Potential Reviewer Kill-Shots

1. "AR candidates beat dLLM candidates from the same base model. The method is a retrieval framework, not a diffusion contribution." (Weakness 1)
2. "Gains over matched-budget iterative methods are marginal on LLaDA." (Weakness 2)
3. "IRCoT finds the answer more often with the same retriever. Why not just use AR?" (Weakness 3)
4. "The paper claims efficiency but provides no wall-clock benchmarks." (Weakness 5)
5. "The method only strongly works on Dream. LLaDA gains are explained by extra passages, not better extraction." (Weakness 6)
6. "The formalization proves TopK extraction is optimal, which is trivially true for any ranking. The interesting claim would be that the dLLM posterior is a better ranking than alternatives, which the ablation disproves." (Weakness 1 + formalization)

