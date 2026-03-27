# Evidence-Adaptive Masked Diffusion (EAMD)

## Verified Mathematical Formulation and Cleaned Pilot Results

This note records the thesis-safe mathematical formulation of EAMD after re-checking the source papers and re-running the cleaned v4 pilot.

The main conclusion is:

- **EAMD-Regen** is the main mathematically grounded method.
- **EAMD-Remask** is a diffusion-native extension, but its current span-remask operator is still heuristic.
- The cleaned 50-question MuSiQue pilot supports `EAMD-Regen > Pool > ARAM/SPREAD` under a shared short-answer setup.

## 1. What is source-backed vs what is ours

### 1.1 Source-backed components

1. **Absorbing-state masked diffusion**  
   Source: [MDLM, arXiv:2406.07524](https://arxiv.org/abs/2406.07524), [Dream 7B, arXiv:2508.15487](https://arxiv.org/abs/2508.15487)  
   Safe claim:
   - discrete masked diffusion uses an absorbing mask state,
   - noising is applied independently across token positions,
   - denoising predicts token distributions from partially masked sequences.

2. **Inference-time remasking is principled in a generalized masked diffusion model**  
   Source: [ReMDM, arXiv:2503.00307](https://arxiv.org/abs/2503.00307)  
   Safe claim:
   - remasking can be introduced by a generalized posterior family,
   - the generalized process can preserve the same marginals as classical masked diffusion under the paper's constraints,
   - remasking schedules can be token-dependent.

3. **Adaptive retrieval guidance from conditional vs prior branches**  
   Source: [ARAM, arXiv:2603.17677](https://arxiv.org/abs/2603.17677)  
   Safe claim:
   - guidance can be driven by the distributional shift between a context-conditioned branch and a prior branch,
   - stronger guidance is appropriate when contextual signal is informative and low-noise.

4. **Iterative retrieval during diffusion-like refinement exists, but not for per-token QA**  
   Source: [TTD-DR, arXiv:2507.16075](https://arxiv.org/abs/2507.16075)  
   Safe claim:
   - retrieval can be updated during an iterative diffusion-style refinement process,
   - that paper operates at the draft/report level, not token-level QA denoising.

### 1.3 Retrieval-corpus caveat for the current experiments

The cleaned v4 pilot in this repository uses:

- questions from the ARAG MuSiQue split,
- retrieval over the **MuSiQue native paragraph corpus**,
- retriever = `E5-base-v2`.

This makes the comparison internally fair because all compared methods use the same retriever and the same corpus.

But it is not a corpus-faithful reproduction of the original papers:

- **SPREAD** reports `NV-Embed-v2`, document chunks of 2,000 characters, and top-5 retrieval, but the paper source checked here does not clearly specify one single named backing corpus such as `wiki18_100w`.
- **ARAM** explicitly uses `bge-large` and retrieves from **MS MARCO 2.1** with top-3 retrieval.

Therefore the current theorem-safe and experiment-safe claim is:

- EAMD-Regen outperforms our SPREAD / ARAM / Pool controls **under a matched MuSiQue-native retrieval setup**.

It is **not yet** a claim of superiority under the original papers' exact retrieval corpora.

### 1.2 Our construction

What is new here is **not** claimed to be already proved by prior work:

1. using **successive evidence sets** `C0 -> C1` rather than `context vs no-context`,
2. defining guidance from the **evidence-marginal shift**
   `p_theta(. | x_t, Q, C1) - p_theta(. | x_t, Q, C0)`,
3. applying that guidance to **short-answer regeneration**,
4. adding a **span remask** operator for answer revision as an ablation.

This distinction matters. The main theorem-backed part of the thesis should be the sampler definition and its reduction properties, not a claim that prior papers already imply the exact algorithm.

## 2. Notation

Let:

- `V` be the vocabulary,
- `m` be the absorbing mask token,
- `Q` be the question,
- `C` be a set of retrieved passages,
- `n` be the fixed short-answer canvas length,
- `x_t in (V union {m})^n` be the answer canvas at denoising step `t`,
- `p_theta(. | x_t, Q, C, t)` be the denoiser distribution over answer tokens.

We use a single refinement round in the current experiments:

- `C0 = R(Q)`
- `C1 = C0 union DeltaC1`

where `R(.)` is the retriever and:

$$
\Delta C_1
=
R(Q \oplus \hat a_0)
\cup
\bigcup_{h \in H_1} R(Q \oplus h),
$$

with:

- `hat a_0` = baseline short answer under `C0`,
- `H_1` = top bridge-entity candidates extracted from the Dream token distribution.

By construction:

$$
C_0 \subseteq C_1.
$$

## 3. Forward masking model

We use the standard absorbing-mask forward process:

$$
q(x_t^i = m \mid x_0^i) = \mu_t,
\qquad
q(x_t^i = x_0^i \mid x_0^i) = 1 - \mu_t,
$$

with:

$$
0 = \mu_0 \le \mu_1 \le \cdots \le \mu_T = 1.
$$

This is the masked diffusion setup used in MDLM-style models. In sequence form, the forward process is applied independently across answer positions.

## 4. EAMD-Regen: evidence-marginal guided regeneration

This is the main mathematically grounded variant and the strongest empirical method.

### 4.1 Definition

At each masked answer position `i` and denoising step `t`, define:

$$
p^{(t)}_{\text{base},i}
:=
p_\theta(\cdot \mid x_t, Q, C_0, t),
\qquad
p^{(t)}_{\text{full},i}
:=
p_\theta(\cdot \mid x_t, Q, C_1, t).
$$

Define the evidence-marginal signal:

$$
S_i^{(t)}
:=
D_{\mathrm{KL}}\!\left(p^{(t)}_{\text{full},i}\,\|\,p^{(t)}_{\text{base},i}\right)
+
D_{\mathrm{KL}}\!\left(p^{(t)}_{\text{base},i}\,\|\,p^{(t)}_{\text{full},i}\right).
$$

Define the uncertainty term:

$$
N_i^{(t)} := H\!\left(p^{(t)}_{\text{full},i}\right).
$$

Let `w_t in [0,1]` be a denoising-time schedule. In the current implementation:

$$
w_t = \frac{t}{T},
$$

so guidance increases later in denoising.

Define:

$$
\gamma_i^{(t)}
:=
\lambda_{\max}
\tanh\!\left(
\beta \frac{S_i^{(t)}}{N_i^{(t)} + \varepsilon}
\right)
w_t.
$$

Let `ell_full,i^(t)` and `ell_base,i^(t)` be the full- and base-evidence logits. Then the guided logits are:

$$
\tilde \ell_i^{(t)}
:=
\ell_{\text{full},i}^{(t)}
+
\gamma_i^{(t)}
\left(
\ell_{\text{full},i}^{(t)} - \ell_{\text{base},i}^{(t)}
\right).
$$

This is exactly the v4 implementation:

- if `gamma = 0`, use pooled decoding under `C1`,
- if `gamma > 0`, amplify the direction in logit space that new evidence prefers over old evidence.

### 4.2 Interpretation

`S_i^(t)` measures how much **newly added evidence** changes the denoiser's belief at token `i`.

This differs from ARAM:

- ARAM compares `context vs prior/no-context`,
- EAMD-Regen compares `updated evidence C1 vs previous evidence C0`.

That is why EAMD-Regen is suited to multi-hop refinement: it asks whether the extra retrieved hop information changes the answer-token distribution.

## 5. EAMD-Remask: span revision under new evidence

This is the diffusion-native ablation. It is useful, but currently weaker than EAMD-Regen.

### 5.1 Seed answer

First decode a short baseline answer under `C0`:

$$
\hat a^{(0)} = \mathrm{Decode}_{C_0}(Q).
$$

Let `J` be the committed answer span up to the first EOS token.

### 5.2 Evidence divergence on committed answer tokens

For each committed position `j in J`, compute:

$$
p_{\text{old},j}
:=
p_\theta(\cdot \mid \hat a^{(0)}, Q, C_0),
\qquad
p_{\text{new},j}
:=
p_\theta(\cdot \mid \hat a^{(0)}, Q, C_1).
$$

Define:

$$
D_j
:=
D_{\mathrm{KL}}(p_{\text{new},j}\,\|\,p_{\text{old},j})
+
D_{\mathrm{KL}}(p_{\text{old},j}\,\|\,p_{\text{new},j}).
$$

Let:

$$
y_j := \arg\max p_{\text{new},j},
\qquad
c_j := \hat a^{(0)}_j.
$$

### 5.3 Current span-remask rule

The current v4 span-remask heuristic is:

$$
R =
\begin{cases}
J, & \exists j \in J \text{ such that } y_j \neq c_j \text{ or } D_j > \delta, \\
\emptyset, & \text{otherwise}.
\end{cases}
$$

Then positions in `R` are remasked and re-denoised using the same evidence-marginal guidance as EAMD-Regen, while positions outside `R` stay fixed.

### 5.4 Status of this rule

This rule is **empirically sensible** but still **heuristic**:

- it is inspired by ReMDM,
- it is not yet derived from a theorem specific to this exact span-remask operator.

So the correct thesis stance is:

- EAMD-Remask is a diffusion-native extension and ablation,
- EAMD-Regen is the main mathematically grounded contribution.

## 6. Propositions and proofs

### Proposition 1. Evidence monotonicity

$$
C_0 \subseteq C_1.
$$

**Proof.** By definition,

$$
C_1 = C_0 \cup \Delta C_1.
$$

Therefore every element of `C0` is also an element of `C1`. QED.

### Proposition 2. EAMD-Regen reduces exactly to Pool

If `gamma_i^(t) = 0` for all positions and steps, then EAMD-Regen is exactly pooled short-answer decoding under `C1`.

**Proof.** If `gamma_i^(t) = 0`, then:

$$
\tilde \ell_i^{(t)} = \ell_{\text{full},i}^{(t)}.
$$

Thus the sampler draws from the `C1` branch exactly, with no extrapolation. This is precisely the `Pool` control in the v4 harness. QED.

### Proposition 3. EAMD-Regen reduces exactly to Baseline when `C1 = C0`

If `C1 = C0`, then EAMD-Regen is identical to baseline short decoding under `C0`.

**Proof.** If `C1 = C0`, then:

$$
p^{(t)}_{\text{full},i} = p^{(t)}_{\text{base},i}
$$

for all `i,t`. Therefore:

$$
S_i^{(t)} = 0
\quad \Rightarrow \quad
\gamma_i^{(t)} = 0,
$$

and:

$$
\tilde \ell_i^{(t)} = \ell_{\text{full},i}^{(t)} = \ell_{\text{base},i}^{(t)}.
$$

So the algorithm is the same as short baseline decoding under `C0`. QED.

### Proposition 4. EAMD-Regen is a bounded extrapolation of pooled decoding

Assume `lambda_max > 0`, `beta > 0`, `eps > 0`, and `w_t in [0,1]`. Then:

$$
0 \le \gamma_i^{(t)} < \lambda_{\max}.
$$

**Proof.** Since symmetric KL is nonnegative and entropy is nonnegative:

$$
S_i^{(t)} \ge 0, \qquad N_i^{(t)} \ge 0.
$$

Hence:

$$
0 \le
\tanh\!\left(
\beta \frac{S_i^{(t)}}{N_i^{(t)} + \varepsilon}
\right)
< 1.
$$

Multiplying by `lambda_max` and `w_t in [0,1]` yields:

$$
0 \le \gamma_i^{(t)} < \lambda_{\max}.
$$

Therefore EAMD-Regen cannot apply arbitrarily large guidance in a single step. QED.

### Proposition 5. No-change stability

If the added evidence does not change the token distribution at any masked answer position, then EAMD-Regen equals Pool.

**Proof.** If:

$$
p^{(t)}_{\text{full},i} = p^{(t)}_{\text{base},i}
$$

for all `i,t`, then `S_i^(t)=0`, hence `gamma_i^(t)=0`, and Proposition 2 applies. QED.

### Proposition 6. EAMD-Remask is identity when no answer position is flagged

If `R = emptyset`, then the EAMD-Remask output is exactly the seed answer `hat a^(0)`.

**Proof.** No answer position is remasked, so no answer token is changed during the refinement step. Therefore the output equals the seed answer. QED.

### Proposition 7. EAMD-Remask reduces to constrained pooled regeneration when guidance is off

If the full answer span is remasked and `gamma = 0` during re-denoising, then EAMD-Remask reduces to regenerating that answer span under `C1` while keeping all non-answer positions fixed.

**Proof.** Full-span remasking removes the current answer tokens. If `gamma = 0`, the sampler uses the `C1` logits directly during re-denoising, exactly as pooled decoding would, but only on the remasked answer span because all other positions are fixed. QED.

## 7. What is **not** proved

The following claims are **not** theorem-backed and should not be written as if they were:

1. `EAMD-Regen` is guaranteed to beat `Pool`, `SPREAD`, or `ARAM`.
2. The evidence-marginal symmetric KL is equal to mutual information.
3. The current span-remask rule is directly covered by ReMDM's theorem.
4. The current empirical gains imply universal superiority beyond the tested MuSiQue slice.

Those are empirical questions, not mathematical theorems.

## 8. Cleaned v4 results

Artifacts:

- `results/eamd_smoke_v4_5q.json`
- `results/eamd_smoke_v4_21235998.out`
- `results/eamd_pilot_v4_50q.json`
- `results/eamd_pilot_v4_21236180.out`

### 8.1 Smoke test: 5 questions

| Method | F1 | EM | Contain |
| --- | ---: | ---: | ---: |
| Baseline | 0.000 | 0.000 | 0.000 |
| SPREAD | 0.040 | 0.000 | 0.000 |
| ARAM | 0.000 | 0.000 | 0.000 |
| Pool | 0.560 | 0.400 | 0.600 |
| EAMD-Regen | 0.560 | 0.400 | 0.600 |
| EAMD-Remask | 0.333 | 0.200 | 0.200 |

### 8.2 Cleaned pilot: 50 questions

| Method | F1 | EM | Contain |
| --- | ---: | ---: | ---: |
| Baseline | 0.336 | 0.240 | 0.320 |
| SPREAD | 0.313 | 0.200 | 0.280 |
| ARAM | 0.313 | 0.240 | 0.260 |
| Pool | 0.419 | 0.280 | 0.400 |
| **EAMD-Regen** | **0.457** | **0.300** | **0.460** |
| EAMD-Remask | 0.392 | 0.280 | 0.360 |

Pairwise counts:

| Comparison | Better | Worse | Same |
| --- | ---: | ---: | ---: |
| EAMD-Regen vs Pool | 4 | 0 | 46 |
| EAMD-Regen vs ARAM | 12 | 3 | 35 |
| EAMD-Regen vs SPREAD | 12 | 5 | 33 |
| EAMD-Remask vs Pool | 3 | 5 | 42 |
| EAMD-Remask vs ARAM | 7 | 1 | 42 |

## 9. Final status for the thesis

The thesis-safe position is:

1. **Main method**: EAMD-Regen  
   This is the main mathematically grounded method and the strongest empirical variant.

2. **Ablation / extension**: EAMD-Remask  
   This supports the diffusion-native revision story, but the current span-remask operator is still heuristic.

3. **Main empirical claim currently supported**:  
   Under a shared short-answer setup, evidence-marginal guided regeneration outperforms the current SPREAD, ARAM, and pooled-evidence baselines on a 50-question MuSiQue pilot.

4. **Next experiment**:  
   scale `EAMD-Regen` to `200q` or `500q` under the same v4 harness before making any broader benchmark claim.

## References

- [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307)
- [Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models](https://arxiv.org/abs/2603.17677)
- [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
- [Deep Researcher with Test-Time Diffusion](https://arxiv.org/abs/2507.16075)
