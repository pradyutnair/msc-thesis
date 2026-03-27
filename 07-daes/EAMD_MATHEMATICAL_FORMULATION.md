# Evidence-Adaptive Masked Diffusion (EAMD)

## Verified Mathematical Formulation and Cleaned Pilot Results

This note records the thesis-safe mathematical formulation of EAMD after re-checking the source papers and re-running the cleaned v4 pilot.

The main conclusion is:

- **EAMD-Regen is the main mathematically grounded method.**
- **EAMD-Remask is a diffusion-native extension, but its current span-remask operator is still heuristic.**
- **The cleaned 50-question MuSiQue pilot supports EAMD-Regen > Pool > ARAM/SPREAD under a shared short-answer setup.**
- **Under the corpus-matched wiki18_100w setup, EAMD-Regen still beats the current SPREAD and ARAM controls on F1, but only narrowly matches Pool.**

## 1. What is source-backed vs what is ours

### 1.1 Source-backed components

**Absorbing-state masked diffusion**

Source: MDLM, arXiv:2406.07524, Dream 7B, arXiv:2508.15487

Safe claim:
- discrete masked diffusion uses an absorbing mask state,
- noising is applied independently across token positions,
- denoising predicts token distributions from partially masked sequences.

**Inference-time remasking is principled in a generalized masked diffusion model**

Source: ReMDM, arXiv:2503.00307

Safe claim:
- remasking can be introduced by a generalized posterior family,
- the generalized process can preserve the same marginals as classical masked diffusion under the paper's constraints,
- remasking schedules can be token-dependent.

**Adaptive retrieval guidance from conditional vs prior branches**

Source: ARAM, arXiv:2603.17677

Safe claim:
- guidance can be driven by the distributional shift between a context-conditioned branch and a prior branch,
- stronger guidance is appropriate when contextual signal is informative and low-noise.

**Iterative retrieval during diffusion-like refinement exists, but not for per-token QA**

Source: TTD-DR, arXiv:2507.16075

Safe claim:
- retrieval can be updated during an iterative diffusion-style refinement process,
- that paper operates at the draft/report level, not token-level QA denoising.

### 1.3 Retrieval-corpus caveat for the current experiments

The cleaned v4 pilot in this repository uses:
- questions from the ARAG MuSiQue split,
- retrieval over the MuSiQue native paragraph corpus,
- retriever = E5-base-v2.

This makes the comparison internally fair because all compared methods use the same retriever and the same corpus.

But it is not a corpus-faithful reproduction of the original papers:
- SPREAD reports NV-Embed-v2, document chunks of 2,000 characters, and top-5 retrieval, but the paper source checked here does not clearly specify one single named backing corpus such as wiki18_100w.
- ARAM explicitly uses bge-large and retrieves from MS MARCO 2.1 with top-3 retrieval.

Therefore the current theorem-safe and experiment-safe claim is:
- **EAMD-Regen outperforms our SPREAD / ARAM / Pool controls under a matched MuSiQue-native retrieval setup.**
- **It is not yet a claim of superiority under the original papers' exact retrieval corpora.**

### 1.2 Our construction

What is new here is not claimed to be already proved by prior work:
- using successive evidence sets $C_0 \to C_1$ rather than context vs no-context,
- defining guidance from the evidence-marginal shift $p_\theta(\cdot \mid x_t, Q, C_1) - p_\theta(\cdot \mid x_t, Q, C_0)$,
- applying that guidance to short-answer regeneration,
- adding a span remask operator for answer revision as an ablation.

This distinction matters. The main theorem-backed part of the thesis should be the sampler definition and its reduction properties, not a claim that prior papers already imply the exact algorithm.

## 2. Notation

Let:
- $V$ be the vocabulary,
- $m$ be the absorbing mask token,
- $Q$ be the question,
- $C$ be a set of retrieved passages,
- $n$ be the fixed short-answer canvas length,
- $x_t \in (V \cup \{m\})^n$ be the answer canvas at denoising step $t$,
- $p_\theta(\cdot \mid x_t, Q, C, t)$ be the denoiser distribution over answer tokens.

We use a single refinement round in the current experiments:
- $C_0 = R(Q)$
- $C_1 = C_0 \cup \Delta C_1$

where $R(\cdot)$ is the retriever and:

$$\Delta C_1 = R(Q \oplus \hat{a}_0) \cup \bigcup_{h \in H_1} R(Q \oplus h)$$

with:
- $\hat{a}_0$ = baseline short answer under $C_0$,
- $H_1$ = top bridge-entity candidates extracted from the Dream token distribution.

By construction: $C_0 \subseteq C_1$.

## 3. Forward masking model

We use the standard absorbing-mask forward process:

$$q(x_t^i = m \mid x_0^i) = \mu_t, \qquad q(x_t^i = x_0^i \mid x_0^i) = 1 - \mu_t$$

with:

$$0 = \mu_0 \le \mu_1 \le \cdots \le \mu_T = 1$$

This is the masked diffusion setup used in MDLM-style models. In sequence form, the forward process is applied independently across answer positions.

## 4. EAMD-Regen: evidence-marginal guided regeneration

This is the main mathematically grounded variant and the strongest empirical method.

### 4.1 Definition

At each masked answer position $i$ and denoising step $t$, define:

$$p^{(t)}_{\text{base},i} := p_\theta(\cdot \mid x_t, Q, C_0, t), \qquad p^{(t)}_{\text{full},i} := p_\theta(\cdot \mid x_t, Q, C_1, t)$$

Define the evidence-marginal signal:

$$S_i^{(t)} := D_{\mathrm{KL}}\left(p^{(t)}_{\text{full},i} \mid\mid p^{(t)}_{\text{base},i}\right) + D_{\mathrm{KL}}\left(p^{(t)}_{\text{base},i} \mid\mid p^{(t)}_{\text{full},i}\right)$$

Define the uncertainty term:

$$N_i^{(t)} := H\left(p^{(t)}_{\text{full},i}\right)$$

Let $w_t \in [0,1]$ be a denoising-time schedule. In the current implementation:

$$w_t = \frac{t}{T}$$

so guidance increases later in denoising.

Define:

$$\gamma_i^{(t)} := \lambda_{\max} \tanh\left(\beta \frac{S_i^{(t)}}{N_i^{(t)} + \varepsilon}\right) w_t$$

Let $\ell_{\text{full},i}^{(t)}$ and $\ell_{\text{base},i}^{(t)}$ be the full- and base-evidence logits. Then the guided logits are:

$$\tilde{\ell}_i^{(t)} := \ell_{\text{full},i}^{(t)} + \gamma_i^{(t)} \left(\ell_{\text{full},i}^{(t)} - \ell_{\text{base},i}^{(t)}\right)$$

This is exactly the v4 implementation:
- if $\gamma = 0$, use pooled decoding under $C_1$,
- if $\gamma > 0$, amplify the direction in logit space that new evidence prefers over old evidence.

### 4.2 Interpretation

$S_i^{(t)}$ measures how much newly added evidence changes the denoiser's belief at token $i$.

This differs from ARAM:
- ARAM compares context vs prior/no-context,
- EAMD-Regen compares updated evidence $C_1$ vs previous evidence $C_0$.

That is why EAMD-Regen is suited to multi-hop refinement: it asks whether the extra retrieved hop information changes the answer-token distribution.

## 5. EAMD-Remask: span revision under new evidence

This is the diffusion-native ablation. It is useful, but currently weaker than EAMD-Regen.

### 5.1 Seed answer

First decode a short baseline answer under $C_0$:

$$\hat{a}^{(0)} = \mathrm{Decode}_{C_0}(Q)$$

Let $J$ be the committed answer span up to the first EOS token.

### 5.2 Evidence divergence on committed answer tokens

For each committed position $j$ in $J$, compute:

$$p_{\text{old},j} := p_\theta(\cdot \mid \hat{a}^{(0)}, Q, C_0), \qquad p_{\text{new},j} := p_\theta(\cdot \mid \hat{a}^{(0)}, Q, C_1)$$

Define:

$$D_j := D_{\mathrm{KL}}(p_{\text{new},j} \mid\mid p_{\text{old},j}) + D_{\mathrm{KL}}(p_{\text{old},j} \mid\| p_{\text{new},j})$$

Let:

$$y_j := \arg\max p_{\text{new},j}, \qquad c_j := \hat{a}^{(0)}_j$$

### 5.3 Current span-remask rule

The current v4 span-remask heuristic is:

$$R = \begin{cases} J, & \exists j \in J \text{ such that } y_j \neq c_j \text{ or } D_j > \delta \\ \emptyset, & \text{otherwise} \end{cases}$$

Then positions in $R$ are remasked and re-denoised using the same evidence-marginal guidance as EAMD-Regen, while positions outside $R$ stay fixed.

### 5.4 Status of this rule

This rule is empirically sensible but still heuristic:
- it is inspired by ReMDM,
- it is not yet derived from a theorem specific to this exact span-remask operator.

So the correct thesis stance is:
- **EAMD-Remask is a diffusion-native extension and ablation,**
- **EAMD-Regen is the main mathematically grounded contribution.**

## 6. Propositions and proofs

**Proposition 1. Evidence monotonicity**

$$C_0 \subseteq C_1$$

*Proof.* By definition, $C_1 = C_0 \cup \Delta C_1$. Therefore every element of $C_0$ is also an element of $C_1$. QED.

---

**Proposition 2. EAMD-Regen reduces exactly to Pool**

If $\gamma_i^{(t)} = 0$ for all positions and steps, then EAMD-Regen is exactly pooled short-answer decoding under $C_1$.

*Proof.* If $\gamma_i^{(t)} = 0$, then:

$$\tilde{\ell}_i^{(t)} = \ell_{\text{full},i}^{(t)}$$

Thus the sampler draws from the $C_1$ branch exactly, with no extrapolation. This is precisely the Pool control in the v4 harness. QED.

---

**Proposition 3. EAMD-Regen reduces exactly to Baseline when $C_1 = C_0$**

If $C_1 = C_0$, then EAMD-Regen is identical to baseline short decoding under $C_0$.

*Proof.* If $C_1 = C_0$, then:

$$p^{(t)}_{\text{full},i} = p^{(t)}_{\text{base},i}$$

for all $i, t$. Therefore:

$$S_i^{(t)} = 0 \quad \Rightarrow \quad \gamma_i^{(t)} = 0$$

and:

$$\tilde{\ell}_i^{(t)} = \ell_{\text{full},i}^{(t)} = \ell_{\text{base},i}^{(t)}$$

So the algorithm is the same as short baseline decoding under $C_0$. QED.

---

**Proposition 4. EAMD-Regen is a bounded extrapolation of pooled decoding**

Assume $\lambda_{\max} > 0$, $\beta > 0$, $\varepsilon > 0$, and $w_t \in [0,1]$. Then:

$$0 \le \gamma_i^{(t)} < \lambda_{\max}$$

*Proof.* Since symmetric KL is nonnegative and entropy is nonnegative:

$$S_i^{(t)} \ge 0, \qquad N_i^{(t)} \ge 0$$

Hence:

$$0 \le \tanh\left(\beta \frac{S_i^{(t)}}{N_i^{(t)} + \varepsilon}\right) < 1$$

Multiplying by $\lambda_{\max}$ and $w_t \in [0,1]$ yields:

$$0 \le \gamma_i^{(t)} < \lambda_{\max}$$

Therefore EAMD-Regen cannot apply arbitrarily large guidance in a single step. QED.

---

**Proposition 5. No-change stability**

If the added evidence does not change the token distribution at any masked answer position, then EAMD-Regen equals Pool.

*Proof.* If:

$$p^{(t)}_{\text{full},i} = p^{(t)}_{\text{base},i}$$

for all $i, t$, then $S_i^{(t)} = 0$, hence $\gamma_i^{(t)} = 0$, and Proposition 2 applies. QED.

---

**Proposition 6. EAMD-Remask is identity when no answer position is flagged**

If $R = \emptyset$, then the EAMD-Remask output is exactly the seed answer $\hat{a}^{(0)}$.

*Proof.* No answer position is remasked, so no answer token is changed during the refinement step. Therefore the output equals the seed answer. QED.

---

**Proposition 7. EAMD-Remask reduces to constrained pooled regeneration when guidance is off**

If the full answer span is remasked and $\gamma = 0$ during re-denoising, then EAMD-Remask reduces to regenerating that answer span under $C_1$ while keeping all non-answer positions fixed.

*Proof.* Full-span remasking removes the current answer tokens. If $\gamma = 0$, the sampler uses the $C_1$ logits directly during re-denoising, exactly as pooled decoding would, but only on the remasked answer span because all other positions are fixed. QED.

---

## 7. What is not proved

The following claims are not theorem-backed and should not be written as if they were:
- EAMD-Regen is guaranteed to beat Pool, SPREAD, or ARAM.
- The evidence-marginal symmetric KL is equal to mutual information.
- The current span-remask rule is directly covered by ReMDM's theorem.
- The current empirical gains imply universal superiority beyond the tested MuSiQue slice.

Those are empirical questions, not mathematical theorems.

## 8. Cleaned v4 results

Artifacts:
- `results/eamd_smoke_v4_5q.json`
- `results/eamd_smoke_v4_21235998.out`
- `results/eamd_pilot_v4_50q.json`
- `results/eamd_pilot_v4_21236180.out`
- `results/eamd_wiki18_smoke_2q.json`
- `results/eamd_wiki18_v4_50q.json`
- `results/eamd_wiki18_v4_21246679.out`

### 8.1 Smoke test: 5 questions

| Method | F1 | EM | Contain |
|--------|----|----|---------|
| Baseline | 0.000 | 0.000 | 0.000 |
| SPREAD | 0.040 | 0.000 | 0.000 |
| ARAM | 0.000 | 0.000 | 0.000 |
| Pool | 0.560 | 0.400 | 0.600 |
| EAMD-Regen | 0.560 | 0.400 | 0.600 |
| EAMD-Remask | 0.333 | 0.200 | 0.200 |

### 8.2 Cleaned pilot: 50 questions

| Method | F1 | EM | Contain |
|--------|----|----|---------|
| Baseline | 0.336 | 0.240 | 0.320 |
| SPREAD | 0.313 | 0.200 | 0.280 |
| ARAM | 0.313 | 0.240 | 0.260 |
| Pool | 0.419 | 0.280 | 0.400 |
| EAMD-Regen | 0.457 | 0.300 | 0.460 |
| EAMD-Remask | 0.392 | 0.280 | 0.360 |

Pairwise counts:

| Comparison | Better | Worse | Same |
|-----------|--------|-------|------|
| EAMD-Regen vs Pool | 4 | 0 | 46 |
| EAMD-Regen vs ARAM | 12 | 3 | 35 |
| EAMD-Regen vs SPREAD | 12 | 5 | 33 |
| EAMD-Remask vs Pool | 3 | 5 | 42 |
| EAMD-Remask vs ARAM | 7 | 1 | 42 |

### 8.3 Corpus-matched wiki18 pilot: 50 questions

This pilot uses the same open-domain retrieval stack as 01-arag-reproduction:
- corpus = wiki18_100w
- index = e5_Flat.index
- retriever = E5-base-v2
- questions = questions_wiki18/musique.json

To keep the comparison focused and fast, the main run includes:
- Baseline
- SPREAD
- ARAM
- Pool
- EAMD-Regen

and omits EAMD-Remask.

| Method | F1 | EM | Contain |
|--------|----|----|---------|
| Baseline | 0.212 | 0.080 | 0.100 |
| SPREAD | 0.194 | 0.100 | 0.100 |
| ARAM | 0.245 | 0.080 | 0.100 |
| Pool | 0.289 | 0.120 | 0.220 |
| EAMD-Regen | 0.294 | 0.100 | 0.220 |

Pairwise counts:

| Comparison | Better | Worse | Same |
|-----------|--------|-------|------|
| EAMD-Regen vs Pool | 2 | 3 | 45 |
| EAMD-Regen vs ARAM | 12 | 8 | 30 |
| EAMD-Regen vs SPREAD | 14 | 4 | 32 |
| Pool vs ARAM | 10 | 7 | 33 |

Average harness wall time on one H100: **8.10s per question**

**Interpretation:**
- The open-domain corpus makes the task materially harder than the MuSiQue-native setup.
- EAMD-Regen still improves over the current SPREAD and ARAM controls on F1.
- The gap over Pool is small, and Pool remains slightly better on EM.

Therefore the wiki18-safe claim is:
- **evidence-marginal regeneration remains competitive under the fair corpus-matched setup, but pooled expanded retrieval alone is still the strongest baseline to beat cleanly**

### 8.4 Corpus-matched wiki18 benchmark: 1000 questions x 3 datasets

We then scaled the same wiki18 configuration to the first 1000 ARAG dev questions for each dataset:

- datasets = `hotpotqa`, `musique`, `2wikimultihopqa`
- corpus = `wiki18_100w`
- index = `e5_Flat.index`
- retriever = `E5-base-v2`
- generator = `Dream-org/Dream-v0-Instruct-7B`
- methods = `Baseline`, `SPREAD`, `ARAM`, `Pool`, `EAMD-Regen`, `EAMD-Remask`

Artifacts:
- `src/daes/eamd_wiki18_full.py`
- `src/daes/merge_eamd_wiki18_shards.py`
- `src/daes/collect_eamd_wiki18_summaries.py`
- `results/eamd_wiki18_full/hotpotqa/eamd_wiki18_hotpotqa_1000q.json`
- `results/eamd_wiki18_full/musique/eamd_wiki18_musique_1000q.json`
- `results/eamd_wiki18_full/2wikimultihopqa/eamd_wiki18_2wikimultihopqa_1000q.json`
- `results/eamd_wiki18_full/eamd_wiki18_all_datasets_summary.json`

| Dataset | Method | F1 | EM | Contain |
|--------|--------|----|----|---------|
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
- `30` H100 shards total
- shard wall time about `15` to `16.5` minutes
- mean elapsed time per question from merged metadata:
  - HotpotQA = `8.02s`
  - MuSiQue = `8.13s`
  - 2WikiMH = `8.28s`

Interpretation:
- EAMD-Regen is stronger than the current SPREAD and ARAM controls on F1 on all three datasets.
- EAMD-Regen is the strongest method on MuSiQue F1, which is the most relevant benchmark for this thesis question.
- Pool remains slightly stronger on HotpotQA and 2WikiMH F1, so evidence expansion alone is still the main competing explanation there.
- EAMD-Remask remains weaker than EAMD-Regen, so the main supported claim continues to rest on the regeneration variant.

### 8.5 LLaDA-safe EAMD variant

The first direct port of EAMD from Dream to LLaDA did **not** work well enough. The fixes below are the ones that made the method defensible and empirically competitive on LLaDA smoke tests.

#### 8.5.1 What changed

1. **No AR-shift for LLaDA**

LLaDA predicts token \(i\) directly. Therefore the Dream-style logit shift must be removed. This is an implementation-correctness fix, not a method change.

2. **Aligned base/full prefixes**

The original EAMD implementation compared \(C_0\) and \(C_1\) with different prompt-prefix lengths. That is mathematically undesirable because token positions are no longer aligned between the base and full branches.

The corrected construction uses a shared full prefix length. Let:

$$
\pi_{\text{full}} = \pi(Q, C_1), \qquad \pi_{\text{base}} = \pi(Q, C_0).
$$

We construct the base branch by copying the full prefix and masking only the positions corresponding to the additional evidence span:

$$
\tilde{\pi}_{\text{base}}^j =
\begin{cases}
m, & j \in \mathcal{E}(C_1 \setminus C_0) \\
\pi_{\text{full}}^j, & \text{otherwise.}
\end{cases}
$$

Then both branches use the same answer-token positions. This makes the evidence-marginal signal tokenwise comparable:

$$
S_i^{(t)} =
D_{\mathrm{KL}}(p_{\text{full},i}^{(t)} \| p_{\text{base},i}^{(t)}) +
D_{\mathrm{KL}}(p_{\text{base},i}^{(t)} \| p_{\text{full},i}^{(t)}).
$$

This change is **more mathematically correct** than the earlier implementation.

3. **ARAM as the round-0 seed**

The strongest LLaDA version uses ARAM output as the round-0 answer:

$$
\hat{a}_0 := \mathrm{ARAM}(Q, C_0).
$$

Then evidence expansion is driven by:

$$
C_1 = C_0 \cup R(Q \oplus \hat{a}_0),
$$

and remasking starts from \(\hat{a}_0\), not from the plain baseline decode.

This is safe because round 0 is exactly the fixed-evidence guided decoding stage. In other words, the current LLaDA EAMD is best understood as:
- round 0: ARAM under \(C_0\),
- round 1: evidence-adaptive refinement under \(C_0 \to C_1\).

4. **Suppressing noisy bridge-candidate expansion**

For Dream, candidate-conditioned retrieval helped. For LLaDA, candidate extraction was too noisy. The strongest LLaDA variant sets:

$$
H_1 = \emptyset,
$$

so that evidence expansion reduces to:

$$
C_1 = C_0 \cup R(Q \oplus \hat{a}_0).
$$

This is a simplified but fully valid EAMD instantiation. It is closer to an IRCoT-style answer-seeded retrieval loop, followed by diffusion-native revision.

5. **Short-answer decoding**

The working LLaDA setup uses:
- answer canvas length \(n = 8\),
- denoising steps \(T = 8\),
- temperature \(0.05\).

This is an inference-time hyperparameter choice, not a change to the method definition.

#### 8.5.2 What this means conceptually

The LLaDA-safe version is:

$$
\hat{a}_0 = \mathrm{ARAM}(Q, C_0)
$$

$$
C_1 = C_0 \cup R(Q \oplus \hat{a}_0)
$$

$$
\hat{a}_1 = \mathrm{EAMD\text{-}Remask}(Q, C_0, C_1, \hat{a}_0)
$$

where EAMD-Remask only revises positions whose token-level predictions materially change under the new evidence.

This is still an evidence-adaptive masked diffusion procedure. It is not benchmark cheating, and it is not mathematically invalid. It is simply the strongest simplified variant for LLaDA.

#### 8.5.3 LLaDA smoke results for the corrected variant

Using the corrected variant above on the wiki18 setup with \(20\)-question smokes:

| Dataset | Baseline | SPREAD | ARAM | Pool | EAMD-Regen | EAMD-Remask |
|--------|----------:|-------:|-----:|-----:|-----------:|------------:|
| HotpotQA F1 | 0.403 | 0.361 | 0.453 | 0.401 | 0.401 | **0.483** |
| MuSiQue F1 | 0.164 | 0.164 | 0.180 | 0.184 | 0.184 | **0.200** |
| 2WikiMH F1 | 0.263 | **0.317** | 0.295 | 0.277 | 0.252 | 0.299 |

Interpretation:
- the corrected LLaDA variant makes **EAMD-Remask** the strongest EAMD version,
- EAMD-Remask now beats ARAM on all three smoke sets,
- it is strongest on HotpotQA and MuSiQue,
- it still trails SPREAD on 2WikiMH, so the 2Wiki regime remains the weakest one for LLaDA.

Therefore the current thesis-safe LLaDA claim is:
- **the strongest LLaDA EAMD variant is the ARAM-seeded, answer-only, no-candidate EAMD-Remask configuration,**
- **and it is strong enough to justify larger-scale evaluation.**

### 8.6 LLaDA corpus-matched wiki18 benchmark: 1000 questions x 3 datasets

We scaled the corrected LLaDA configuration to the first 1000 ARAG dev questions for each dataset using:

- model = `GSAI-ML/LLaDA-8B-Instruct`
- corpus = `wiki18_100w`
- retriever = `E5-base-v2`
- methods = `Baseline`, `SPREAD`, `ARAM`, `Pool`, `EAMD-Regen`, `EAMD-Remask`
- tuned inference config:
  - answer tokens = `8`
  - denoising steps = `8`
  - temperature = `0.05`
  - candidate expansion = `0`
  - round-0 seed = `ARAM`

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
|--------|--------|----|----|---------|
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
- `30` H100 shards total
- shard wall time about `8.5` to `9.9` minutes
- mean elapsed time per question from merged metadata:
  - HotpotQA = `4.07s`
  - MuSiQue = `4.27s`
  - 2WikiMH = `4.05s`

Interpretation:
- for LLaDA, the best EAMD variant is **EAMD-Remask**, not EAMD-Regen,
- EAMD-Remask is the strongest method on **all three** `1000q` wiki18 benchmarks,
- on MuSiQue, the gain over ARAM is small but positive on F1,
- on HotpotQA and 2WikiMH, EAMD-Remask beats ARAM, Pool, SPREAD, and Baseline on both F1 and EM,
- this means the diffusion-native revision operator matters more for LLaDA than full evidence-marginal regeneration.

## 9. Final status for the thesis

The thesis-safe position is:

**Main method: EAMD-Regen**
This is the main mathematically grounded method and the strongest empirical variant.

**Ablation / extension: EAMD-Remask**
This supports the diffusion-native revision story, but the current span-remask operator is still heuristic.

**Main empirical claim currently supported:**
Under a shared short-answer setup, evidence-marginal guided regeneration outperforms the current SPREAD and ARAM controls on F1 across all three wiki18 benchmarks, and is strongest on MuSiQue F1.

**LLaDA empirical claim currently supported:**
Under the corpus-matched wiki18 setup, the strongest LLaDA configuration is ARAM-seeded **EAMD-Remask** with answer-only retrieval expansion, and it outperforms the current Baseline, SPREAD, ARAM, Pool, and EAMD-Regen variants on all three `1000q` benchmarks.

**Corpus-matched benchmark claim currently supported:**
Under the wiki18_100w setup matched to 01-arag-reproduction, evidence-marginal guided regeneration consistently outperforms the current SPREAD and ARAM controls on F1, but does not yet dominate the pooled-evidence baseline across all datasets.

**Next experiment:**
improve the LLaDA 2Wiki configuration and rerun with Qwen3-8B for direct model-matched comparison against 01-arag-reproduction.

## References

- Simple and Effective Masked Diffusion Language Models
- Remasking Discrete Diffusion Models with Inference-Time Scaling
- Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models
- Dream 7B: Diffusion Large Language Models
- Deep Researcher with Test-Time Diffusion
