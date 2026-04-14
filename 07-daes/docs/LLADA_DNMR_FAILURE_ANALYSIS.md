# LLaDA DNMR Failure Analysis

## Scope

This note analyzes `DNMR = pool` only.

It addresses:

1. Whether DNMR is a valid paper direction.
2. Why DNMR is weaker on LLaDA-8B than Dream-7B.
3. Whether the retrieval subqueries are useful.
4. What failure cases suggest about how to improve DNMR.
5. What proposal has the best chance of pushing LLaDA DNMR toward a publishable result.

## Main result

The direction is still valid, but the support is narrower than the earlier `idnmr` framing.

- On Dream, `pool` is consistently better than baseline and better than `iaram` / `ispread`, but not yet by a large margin.
- On LLaDA, `pool` beats baseline on judge, but it does **not** beat the best iterative baselines on MuSiQue or 2Wiki.
- So a DNMR paper is defensible if centered on:
  - real multi-query retrieval gains
  - stronger Dream evidence
  - a clear LLaDA failure analysis
- It is **not** yet defensible as:
  - “DNMR clearly dominates all strong baselines on LLaDA”

## Judge performance of DNMR vs baselines

Using the LLM-judge outputs:


| Model | Dataset  | `pool` | `baseline` | `iaram` | `ispread` | Rel. gain vs best of `iaram`,`ispread` |
| ----- | -------- | ------ | ---------- | ------- | --------- | -------------------------------------- |
| LLaDA | MuSiQue  | 0.3060 | 0.2102     | 0.3136  | 0.3190    | -4.1%                                  |
| LLaDA | HotpotQA | 0.5696 | 0.5040     | 0.5516  | 0.5510    | +3.3%                                  |
| LLaDA | 2Wiki    | 0.4100 | 0.3570     | 0.3960  | 0.4210    | -2.6%                                  |
| Dream | MuSiQue  | 0.2598 | 0.1971     | 0.2420  | 0.2351    | +7.3%                                  |
| Dream | HotpotQA | 0.5637 | 0.5124     | 0.5271  | 0.5373    | +4.9%                                  |
| Dream | 2Wiki    | 0.4387 | 0.3835     | 0.4159  | 0.3988    | +5.5%                                  |


Interpretation:

- Dream DNMR is positive on all 3 datasets.
- LLaDA DNMR is mixed:
  - positive on HotpotQA
  - negative against the best iterative baselines on MuSiQue and 2Wiki

### Distance to the `>10%` relative target on LLaDA

To beat the best of `iaram` / `ispread` by `>10%` relative on judge:


| Dataset  | Current `pool` | Target | Extra correct per 1000 needed |
| -------- | -------------- | ------ | ----------------------------- |
| MuSiQue  | 0.3060         | 0.3509 | 44.9                          |
| HotpotQA | 0.5696         | 0.6067 | 37.1                          |
| 2Wiki    | 0.4100         | 0.4631 | 53.1                          |


This is materially harder than the earlier `idnmr` target.

## Why DNMR is weaker on LLaDA than Dream

### 1. LLaDA DNMR bridge candidates are low quality

Using round 0 candidate traces as the DNMR subquery set:


| Metric                                          | LLaDA old `pool` | Dream old `pool` |
| ----------------------------------------------- | ---------------- | ---------------- |
| template-style candidates (`The answer is ...`) | 80.3%            | 9.1%             |
| question-copy candidates                        | 22.6%            | 45.0%            |
| no new passages after DNMR expansion            | 5.4%             | 10.1%            |
| candidate uniqueness ratio                      | 0.981            | 0.981            |


Interpretation:

- LLaDA DNMR queries are not obviously diverse, but the bigger issue is that they are very often templated answer fragments.
- Dream DNMR candidates are much more natural bridge prompts.
- This directly weakens the retrieval expansion stage for LLaDA.

### 2. LLaDA DNMR final answers are too verbose

Average answer length for `pool` on MuSiQue:


| Model                   | Avg chars | Median chars |
| ----------------------- | --------- | ------------ |
| LLaDA old `pool`        | 110.2     | 120.0        |
| Dream old `pool`        | 15.6      | 13.0         |
| LLaDA current `pool_v2` | 20.4      | 20.0         |


Interpretation:

- Old LLaDA DNMR suffered from severe verbosity.
- Dream DNMR already behaves like a short-answer system.
- The current `pool_v2` fix solved the verbosity issue, but did not solve the underlying correctness issue.

### 3. The current `pool_v2` fix likely over-corrected the bridge stage

Current MuSiQue `pool_v2` candidates:


| Setting                 | avg chars | avg words | short candidates | template candidates |
| ----------------------- | --------- | --------- | ---------------- | ------------------- |
| LLaDA old DNMR          | 31.5      | 5.5       | 28.0%            | 31.2%               |
| LLaDA current `pool_v2` | 14.0      | 2.2       | 100.0%           | 0.5%                |


Interpretation:

- Old LLaDA candidates were too verbose and templated.
- Current `pool_v2` candidates are now too short and too local.
- This explains why raw F1 improved but judge/contain did not recover to the old DNMR level.

## Are DNMR subqueries useful?

Yes, but unevenly.

### Old LLaDA DNMR on MuSiQue

For `baseline wrong -> pool correct`:

- gold answer appears in DNMR candidate set on `19.4%` of wins
- gold answer appears in the final DNMR answer on `75.3%` of wins

For `baseline correct -> pool wrong`:

- gold answer appears in DNMR candidate set on `33.3%` of losses
- gold answer appears in the final DNMR answer on `0.0%` of losses

Interpretation:

- DNMR is doing something real. It can recover multi-hop structure that baseline misses.
- But the candidate set often does not directly expose the final gold answer, so DNMR success comes from improved retrieval context rather than literal answer proposal.

### Current `pool_v2` on MuSiQue

For `baseline wrong -> pool_v2 correct`:

- gold answer appears in candidate set on `23.9%` of wins

For `baseline correct -> pool_v2 wrong`:

- gold answer appears in candidate set on `54.5%` of losses

Interpretation:

- The current bottleneck is not only retrieval.
- In many current losses, the right clue is already present, but the final answer selection still fails.

## What the failure cases show

### DNMR wins

The cleanest DNMR wins are true bridge-recovery questions:

- county / borough / district questions
- education / affiliation questions
- authorship / role questions

Representative old LLaDA `pool` wins:

- `dev_12`: `Huyton -> Knowsley`
- `dev_19`: `Canyon County -> Randall County`
- `dev_21`: `Eton College -> Exeter College`
- `dev_36`: `Corfe Mullen -> East Dorset`

These are legitimate multi-hop improvements.

### DNMR losses

The main loss patterns are:

1. Wrong granularity

- county vs city
- district vs town
- country vs region
- exact year/month vs nearby date

1. Candidate drift

- one candidate is close to the right bridge
- another candidate steers retrieval to a nearby wrong entity

1. Final answer selection failure

- relevant evidence is present
- model still outputs the wrong entity or wrong abstraction level

Representative old LLaDA `pool` losses:

- `dev_67`: asks for founding era, DNMR predicts `1940s` after bad temporal candidates
- `dev_88`: asks for `central Atlantic Ocean`, DNMR outputs `westernmost region of Africa`
- `dev_93`: candidate set includes `James Watt was educated at the Universit...`, but final answer drifts to birthplace
- `dev_205`: candidate set includes `October 1190`, but DNMR abstains instead of extracting `October`

Representative current `pool_v2` losses:

- `dev_16`: candidate set mixes Pasadena and Granby schools, final answer picks the wrong school
- `dev_56`: retrieved answer is `Woolhampton, Berkshire`, but the question asks for the district `West Berkshire`

## Answer to the supervisor questions

### Why is the method less effective on LLaDA-8B?

Because DNMR on LLaDA has two separate problems:

1. **Bridge generation**

- old DNMR: candidates too verbose / templated
- current DNMR: candidates too short / shallow

1. **Answer selection**

- even when the right clue is retrieved, LLaDA often outputs the wrong granularity or the wrong nearby entity

Dream does not suffer from these issues nearly as strongly:

- its DNMR candidates are cleaner
- its answers are already short
- so retrieval gains transfer into final answers more reliably

### Are the retrieval subqueries reasonable?

Sometimes yes, sometimes no.

The old DNMR traces show clear positive cases where bridge-like candidates recover the correct relation chain.
But many LLaDA DNMR candidates are weak:

- answer-template fragments
- partial paraphrases of the question
- vague short spans that are not stable retrieval keys

So the current direction is reasonable, but the current candidate generator is not strong enough.

### What do baseline-success / DNMR-fail cases show?

They show that:

- baseline often wins on questions requiring exact answer granularity
- DNMR often adds evidence but then overgeneralizes or chooses a nearby wrong entity
- current LLaDA DNMR is not failing purely because retrieval is bad

### What queries should be collected to show DNMR advantage?

DNMR’s strongest category is relation-chaining questions where the answer is not the first obvious entity:

- county / district / borough / territorial entity
- child / spouse / father / member-of / author-of
- “where was X educated”, “what team was X on”, “what record label did Y start”

Those are the best cases to feature qualitatively.

## Proposal

If the goal is a stronger DNMR paper result, the next experiments should target `pool` directly.

### Proposal A: Decouple DNMR bridge generation from final answer generation

Use two different decoding styles:

- **Bridge stage**: higher-recall, longer candidate proposals
- **Answer stage**: short canonical answer

Rationale:

- current `pool_v2` made the answer stage cleaner
- but it also made the bridge stage too shallow
- those two stages should not share the same decode budget

This is the most plausible way to improve LLaDA DNMR without losing the current formatting gain.

### Proposal B: Add retrieval-aware answer selection on top of DNMR

After DNMR retrieval expansion:

1. generate a short answer candidate
2. run a constrained answer selector that prefers:
  - exact spans from retrieved passages
  - correct granularity for location/date questions
  - single entities over multi-entity strings

Rationale:

- many current DNMR losses already contain the right clue
- this is likely the cheapest source of extra correct answers

### Proposal C: Replace LLaDA bridge candidates if needed

The repo already contains evidence that candidate source matters:


| Source          | F1     | Contain |
| --------------- | ------ | ------- |
| no_branch       | 0.2008 | 0.177   |
| dLLM candidates | 0.2484 | 0.225   |
| AR candidates   | 0.2729 | 0.243   |


Interpretation:

- DNMR as multi-query retrieval is useful
- but the LLaDA posterior may not be the best bridge proposal mechanism

This is the pragmatic fallback path:

- use a stronger candidate source
- keep the DNMR retrieval expansion
- keep LLaDA as the answer model if needed

It weakens a pure diffusion-specific claim, but it may be the fastest route to a publishable DNMR result.

## Recommended experiment order

1. **Current `pool_v2` + answer selector**

- 50q MuSiQue first
- targets current answer-selection failures directly

1. **Decoupled bridge stage**

- longer / higher-recall bridge candidates
- short final answer budget

1. **If needed, hybrid candidate source**

- AR-generated bridge candidates + DNMR retrieval + LLaDA final answer

1. **Only then scale**

- HotpotQA and 2Wiki after MuSiQue shows a clear signal

## Bottom line

DNMR is still a valid paper direction, but the current evidence supports a narrower claim:

- DNMR-style multi-query retrieval is useful.
- Dream gives the stronger positive result.
- LLaDA shows that the direction is reasonable but bottlenecked by candidate generation and answer selection.

For a DNMR paper, the next strong result will probably come from:

- fixing bridge generation for LLaDA DNMR
- or fixing answer selection after DNMR retrieval
- not from more reruns of the current `pool_v2` design unchanged

---

# Deep Failure Analysis — April 2026

**Author**: Claude Code analysis of all prediction JSONs (mixed/, agnostic/, pool_v2/, pool_v3/, bridge_pilot/, llada_micro_pilots/, qgate/, k2/, cross50c/)
**Last updated**: 2026-04-08

---

## Quantitative Failure Breakdown

Analysed across 740q (MuSiQue), 890q (HotpotQA), 810q (2WikiMH) from `results/mixed/` (n_tokens=32, original pipeline).

### Win/Loss/Tie (pool F1 vs baseline F1)


| Dataset          | Pool Better | Pool Worse | Pool Same |
| ---------------- | ----------- | ---------- | --------- |
| MuSiQue (N=740)  | 143         | 111        | 486       |
| HotpotQA (N=890) | 150         | 128        | 612       |
| 2WikiMH (N=810)  | 131         | 113        | 566       |


### Failure Rates by Category


| Category                                          | MuSiQue     | HotpotQA    | 2WikiMH     |
| ------------------------------------------------- | ----------- | ----------- | ----------- |
| [A1] Verbose: contain=1 but F1<0.5                | 127 (17.2%) | 225 (25.3%) | 214 (26.4%) |
| [B1] Degenerate bridge (>=50% "The answer is...") | 121 (16.4%) | 190 (21.3%) | 126 (15.6%) |
| [B2] Gold absent from ALL candidates              | 529 (71.5%) | 427 (48.0%) | 462 (57.0%) |
| [C1] Baseline correct -> pool corrupts            | 56 (7.6%)   | 77 (8.7%)   | 38 (4.7%)   |
| [C2] Answer completely flipped                    | 9 (1.2%)    | 10 (1.1%)   | 16 (2.0%)   |


### Cross-Model Comparison (MuSiQue)


| Metric                                    | Dream      | LLaDA      | Ratio        |
| ----------------------------------------- | ---------- | ---------- | ------------ |
| Degenerate bridge candidates              | 2.6%       | 31.2%      | 12x worse    |
| Verbose final answers (contain=1, F1<0.5) | 1.0%       | 17.2%      | 17x worse    |
| Avg candidate length                      | 27.8 chars | 31.5 chars | similar      |
| Final answer 1-3 words                    | 90%        | ~10%       | 9x worse     |
| Final answer 16+ words                    | ~1%        | 82%        | catastrophic |
| Pool answer longer than baseline          | 16%        | 49%        | 3x more      |


---

## Category A: Final Answer Verbosity

### A1 — Canvas Size Overrides Instructions

**Status: CONFIRMED ROOT CAUSE. Fix validated at 50q, not yet complete at 1000q.**

LLaDA ignores the "1-6 word" prompt instruction when `n_tokens=32`. The diffusion model fills the entire canvas. Dream at the same 32 tokens writes "Paris"; LLaDA writes "Paris is the capital of France, located in the Ile-de-France region...". This is diffusion-native: the canvas length is a generative prior that supersedes the instruction.

**Verbosity fix pilot result (50q MuSiQue, IVI A6000):**


| Method            | F1    | AvgLen     | Note                  |
| ----------------- | ----- | ---------- | --------------------- |
| baseline (32 tok) | 0.138 | 15.7 chars | —                     |
| pool_8 (8 tok)    | 0.194 | 22.6 chars | +5.6pp — matches ARAM |
| pool_12 (12 tok)  | 0.157 | 24.0 chars | +1.9pp                |
| pool_16 (16 tok)  | 0.146 | 29.5 chars | +0.8pp                |
| pool_32 (32 tok)  | 0.173 | 27.1 chars | +3.5pp                |


**Deployed in**: `agnostic/` (1000q, partial), `qgate/`, `k2/`, `cross50c/`, `pool_v2`, `pool_v3`.

> **User note (April 2026)**: Tried this — helps but does not fully fix the problem. Baseline corruption persists. Yes/no corruption persists.

---

### A2 — Baseline Corruption (Pool Hurts Correct Baseline)

**Status: CONFIRMED, NOT FIXED.**

In 7.6-8.7% of questions where baseline F1 > 0.5, the pool answer has F1 < 0.2. Examples:

- Gold: "YG Entertainment" | Baseline: "YG Entertainment" (F1=1.0) | Pool: "2014 S/S is the debut album of a South Korean boy group that was formed by the group's record label, YG Entertainment." (F1=0.11)
- Gold: "yes" | Baseline: "Yes" (F1=1.0) | Pool: "Yes, both Local H and For Against are from the United States, but Local H is from Illinois..." (F1=0.04)

Mechanism: expanded context + bridge candidates causes LLaDA to generate an explanatory sentence despite already knowing the answer.

> **User note**: Tried reducing bridge candidate canvas size (n_mask). The corruption pattern persists because the issue is in the final decode step, not the extraction step.

---

### A3 — Yes/No Question Failure

**Status: CONFIRMED as F1 artifact; JUDGE CORRECTS FOR THIS.**

Gold: "yes" or "no" | Pool: "Yes, both are..." (F1 ~0.04 vs F1=1.0 for baseline). This is a pure F1 metric artifact. A judge correctly scores verbose "Yes, because..." as correct for gold "yes".

Judge table: LLaDA pool HotpotQA judge=57.0% vs baseline=50.4% (+6.6pp). Pool wins on judge even while losing on raw F1. Yes/no verbosity does NOT explain the LLaDA DNMR failure on judge metrics.

> **User note**: Agreed — judge fixes this. Not a real problem for the paper result.

---

## Category B: Bridge Candidate Degeneration

### B1 — The "Template" Trap

**Status: CONFIRMED, ROOT CAUSE IS POSTERIOR PEAKEDNESS.**

31.2% of LLaDA bridge candidates start with "The answer is..." vs 2.6% for Dream. LLaDA's peaked posterior (H=0.001) generates final-answer-form sentences even during bridge extraction. The extraction step is semantically identical to the final answer step for LLaDA.

On candidate quality when gold IS present in a candidate:


| Model | Short correct (<=4w) | Long correct (>4w) | Degenerate correct |
| ----- | -------------------- | ------------------ | ------------------ |
| Dream | 49%                  | 49%                | 2%                 |
| LLaDA | 24%                  | 48%                | 29%                |


> **User note**: The degenerate "The answer is: X" candidates still retrieve the right documents because X is correct. Recall is +12-13pp over ARAM proven at 740q. The bridge candidates happen to be the final answer, not an intermediate entity — but they still work as retrieval queries. This is confirmed by judge showing pool wins on HotpotQA (+6.6pp) even with 31% degenerate candidates. So B1 is a symptom, not the primary blocker.

---

### B2 — Dead Code Path: `_clean_bridge_candidate()` Not Called

**Status: CODE BUG CONFIRMED, IMPACT NOT ISOLATED.**

`_clean_bridge_candidate()` is defined at line ~778 of `eamd_v2_wiki18.py`. It strips "The answer is:", leading articles, pronouns, and truncates to 6 words. It is called in `extract_candidates_mixed_posterior` (the pool_v3 extractor) but NOT in `extract_candidates_agnostic` (the active extractor for standard pool/idnmr in the 1000q runs).

Effect: Candidates like "The answer is: Time Warner Cable" are used verbatim as retrieval queries instead of "Time Warner Cable". Dense retrievers (E5-base-v2) were not trained on queries with answer-template prefixes.

> **User note**: Impact unclear in isolation. pool_v3 (which does call it) also changes many other things. Not isolated.

---

### B3 — Query Corruption from Degenerate Prefix

**Status: CONFIRMED MECHANISM.**

"The answer is: Tamaulipas" as a retrieval query retrieves different documents than "Tamaulipas". Observed pattern:

- Gold: "Cologne" | All 3 candidates: "Munich, Germany" / "The answer is: Munich, Germany." / "Munich, Germany and Munich, Germany." | Retrieval stays on Munich. Cologne documents never fetched.
- Gold: "Tamaulipas" | Candidates: "The answer is: Benito Juarez borough" / "Benito Juarez borough of Mexico City." | Pool answer: "Madrid, Real Madrid."

> **User note**: Agreed — prefix cleaning should help. Already fixed in pool_v3 via `_clean_bridge_candidate`. Not applied to standard pool/idnmr.

---

## Category C: Structural and Distributional Issues

### C1 — Entropy Collapse (Position Selection Breaks Down)

**Status: ROOT CAUSE CONFIRMED. FIXED IN pool_v3.**

`extract_candidates_agnostic` selects branch positions via `torch.topk(entropy, n_positions)`. For LLaDA with H=0.001, "top entropy positions" are effectively random — no position meaningfully encodes bridge-entity uncertainty.

> **User note**: Why can't we fix this?
>
> The entropy collapse is a fundamental property of LLaDA's masked diffusion posterior. It is not a hyperparameter. What CAN be changed is the position selection criterion:
>
> `extract_candidates_mixed_posterior` (used in pool_v3) replaces entropy alone with `entropy_j x KL(full||base)_j`. This measures how much the retrieved context MOVED the prediction at position j, not how uncertain the model is in absolute terms. Even with H=0.001, the KL from no-context to full-context can be non-trivial at bridge-informative positions.
>
> **This is already implemented and tested**: pool_v3_bridge at 50q LLaDA MuSiQue = F1=0.263 vs standard pool F1=0.132. A +13.1pp gain from the extractor change (combined with n_tokens=8 and candidate cleaning). So entropy collapse IS fixable — it requires KL-based rather than entropy-based position selection.

---

### C2 — Temperature Oversharpening

**Status: PARTIAL TEST, NOT CONCLUSIVE.**

Extraction uses temperature=0.3 for position selection and temperature=0.1 for branch denoising. With H=0.001, temperature=0.3 is effectively argmax. All branches sample the same top-1 token.

> **User note**: Can we fix this somehow?
>
> Yes, in principle: raising extraction temperature to 0.8-1.5 would flatten the distribution enough to create diversity between branches. However:
>
> - Micro pilots tested temperature 0.25 vs 0.30 — identical results (F1=0.191 both). The range tested is too narrow.
> - pool_v3's KL-based position selection sidesteps the temperature problem: seeding branches at KL-high positions creates genuine diversity even at temperature=0.3, because the selected positions encode different contextual information.
> - Temperature sweep at >=0.5 on `extract_candidates_mixed_posterior` has NOT been tried and could yield further gains.

---

### C3 — Canvas Waste in Extraction (n_mask=12)

**Status: CONFIRMED, PARTIALLY ADDRESSED IN pool_v3.**

With n_mask=12 for extraction, LLaDA uses positions 0-3 for "The answer is:" and positions 4-12 for content. First 4 tokens are identical across all branches.

Tested: `pool_m6_8_hint2` (n_mask=6, n_tokens=8, + hint) at 10q MuSiQue: F1=0.187 vs pool_8 (n_mask=12, n_tokens=8): F1=0.141. Clear improvement. pool_v3 uses bridge_n_mask=10 (shorter but not 6).

> **User note**: Can modify this. Shorter extraction canvas forces entity-length output. Confirmed improvement at 10q.

---

## The pool_v3 Pipeline: What It Does and What Happened at 1000q

### What pool_v3 Does (dnmr_pool_v2_lean.py)

pool_v3 combines all working fixes:

1. **Extraction**: `extract_candidates_mixed_posterior` — KL x entropy position selection, `_clean_bridge_candidate` applied to all outputs, n_mask=10, 16 extraction steps
2. **Hint**: Candidates prepended as "Related entities: X, Y, Z." before context
3. **Final decode**: n_tokens=8 (or n_tokens=2 for yes/no questions), steps=32
4. **pool_v3_full**: additionally uses `eamd_regen_shared` token-level regeneration score

### 50q Results (LLaDA MuSiQue)


| Method                 | F1    | Precision | Recall | Contain |
| ---------------------- | ----- | --------- | ------ | ------- |
| baseline               | 0.132 | 0.111     | 0.219  | 8%      |
| pool_m6_8_hint2_yn     | 0.205 | 0.184     | 0.248  | 12%     |
| pool_v3_bridge         | 0.263 | 0.272     | 0.268  | 16%     |
| pool_v3_bridge_curated | 0.281 | 0.292     | 0.285  | 18%     |
| pool_v3_full           | 0.279 | 0.288     | 0.285  | 18%     |


For reference: ARAM at 1000q = 0.188-0.200. ARAM at 50q ≈ 0.164 (from micro pilots). pool_v3_bridge at 50q is +9.9pp above ARAM at 50q.

### Candidate Quality Improvement vs Standard Pool


| Metric                    | Standard pool | pool_v3   |
| ------------------------- | ------------- | --------- |
| Degenerate candidates (%) | 31.2%         | 0.0%      |
| Avg candidate word count  | 5.5 words     | 3.8 words |


### CRITICAL: pool_v3 at 1000q Scale — Something Breaks

> **User observation (April 2026)**: Ran pool_v3_bridge_candidates at 1000q. At 200q checkpoint, contain = 14%. This is worse than the 50q result (16% contain) and barely above the old verbose pool (22.3%). Something going horribly wrong at scale.

This was not investigated in the analysis session. Possible causes to investigate:

1. **Different question distribution at 200q+**: The first 50q pilot may have overrepresented easy bridge-recovery questions (county/district/affiliation). Questions 50-200 may be harder types where pool_v3 gains do not transfer.
2. **KL signal collapse for harder questions**: For questions where retrieved documents have little bearing on the answer, KL(full||base) may be near-zero everywhere. Position selection degrades to random, losing pool_v3's advantage.
3. **Hint format failure for certain question types**: "Related entities: X, Y, Z." prepended to context may confuse LLaDA on yes/no or comparative questions, introducing noise that was not present in the 50q pilot.
4. `**eamd_regen_shared` degradation** (pool_v3_full only): The regen scoring component may fail on out-of-distribution questions at scale. Check whether pool_v3_bridge (without regen) also collapses at 200q.
5. **The 50q pilot was unrepresentative**: Pilot ran on questions 0-49. Questions 50-249 (the next shard) may have a systematically different distribution.

**Priority**: Before running any new experiments, examine the 200q checkpoint predictions to identify which failure mode is occurring.

---

## Status of All Proposed Directions


| Direction                           | Tested?                                  | Best Result                    | 1000q?           |
| ----------------------------------- | ---------------------------------------- | ------------------------------ | ---------------- |
| D1: n_tokens=8 for final decode     | YES (pool_8, agnostic/)                  | +5.6pp at 50q                  | Partial only     |
| D2: _clean_bridge_candidate applied | YES in pool_v3 only                      | Part of pool_v3 +13pp          | No               |
| D3: KL x entropy position selection | YES (extract_candidates_mixed_posterior) | pool_v3_bridge F1=0.263 at 50q | Breaks at 1000q? |
| D4: Bridge-specific NER prompt      | YES (bridge_pilot/)                      | FAILED: F1 dropped to 0.077    | N/A              |
| D5: Extraction temperature >0.3     | PARTIAL (0.25 vs 0.30 only)              | No difference in narrow range  | No               |
| D6: n_mask=6 for extraction         | YES (pool_m6_8_hint2)                    | F1=0.187 at 10q                | No               |
| D7: Two-stage design (pool_v3)      | YES at 50q only                          | F1=0.263-0.281                 | Breaks at 1000q  |


### Confirmed Dead Ends

- **D4 (bridge-specific NER prompting)**: Explicitly fails. Tested in `bridge_pilot/`. NER-style candidates retrieve wrong documents.
- **Conditional remasking**: 0/169 samples more diverse than greedy.
- **Logit lens**: 0 bridge hits.
- **PAQCD / ABRD**: No effect.
- **Extraction temperature 0.25 vs 0.30**: No difference.

---

## Failure Cascade Summary

```
Peaked posterior (H~0.001)
    -> Entropy-based position selection selects random positions (C1)
    -> Branches seeded at irrelevant positions collapse to same answer (C2)
    -> Extraction canvas n_mask=12 gives room for "The answer is:" prefix (C3)
    -> All candidates become "The answer is: [final answer]" (B1)
    -> _clean_bridge_candidate not applied in agnostic extractor (B2)
    -> Query sent to retriever with "The answer is:" prefix (B3)
    -> Retrieval still improves recall +12-13pp (candidates happen to be right answer)
    -> Final decode at n_tokens=32 fills canvas with explanatory sentence (A1)
    -> Verbose answer tanks F1/precision despite correct retrieval (A2)
```

pool_v3 addresses C1 (KL position selection), C3 (n_mask=10), B2+B3 (candidate cleaning), A1 (n_tokens=8), A3 partial (yes/no detection). At 50q it works. At 1000q it breaks. The open problem is diagnosing the 1000q collapse before proposing further fixes.

---

## What Remains Open

1. **Why does pool_v3 fail at 1000q?** Contains=14% at 200q checkpoint. Must examine those prediction JSONs. Was done on another machine and the jsons were lost, so we do not have this information now. However, we did see that it did not work.
2. **Does KL x entropy signal (D3) degrade for harder question types?** Not measured outside the 50q pilot.
3. **Extraction temperature >=0.5 on extract_candidates_mixed_posterior**: Not tested.
4. **Is the hint format hurting for certain question types?** Hypothesis: "Related entities: X, Y, Z." biases LLaDA against binary answers.
5. **agnostic/ 1000q runs are incomplete**: ~10 questions per 200q shard. Need to check whether jobs are still running or hit an error.

