# EAMD v4: Trajectory-Consistent Contradiction-Aware Guidance

## 0. Status of Claims

### Theorem-backed in this document

1. The multi-round trajectory posterior is the exact optimizer of a per-round trust-region objective.
2. The global closed form
   $$
   q_R(x) \propto p_0(x)\exp\!\Big(\sum_{r=1}^R \alpha_r \Delta_r(x)\Big)
   $$
   follows exactly by induction.
3. The contradiction score defined from the cosine between the local innovation and the accumulated trajectory direction is bounded in $[0,1]$.
4. The guidance scale
   $$
   \alpha_{r,j,t}^* = \frac{w_t S_{r,j}}{N_{r,j} + \kappa c_{r,j} + \varepsilon}
   $$
   is the exact optimizer of a local quadratic contradiction-aware utility.
5. The aggregated-logit update
   $$
   \bar\ell_{r,j} = \bar\ell_{r-1,j} + \alpha_{r,j,t}(\ell_{r,j} - \ell_{r-1,j})
   $$
   is exact up to an additive normalizing constant.
6. When $R=1$, the trajectory update recovers single-round EAMD after the reparameterization $\alpha_1 = 1 + \gamma$.
7. No memoryless iterative guidance rule that depends only on $(p_{r-1}, p_r)$ can reproduce the trajectory posterior for all histories.
8. Contradiction-aware trajectory memory adds no extra model forward passes beyond the underlying pairwise guided iterative decode.

### Assumption-backed in this document

1. The local quadratic utility in Section 4 is a second-order certainty-equivalent surrogate, not an oracle EM/F1 objective.
2. The efficiency comparison to IRCoT assumes the same per-round retrieval budget and a batched old/new-context dLLM forward pass.
3. The tokenwise factorized update in Section 5 is the standard local denoising approximation used in v2 and v3; it is exact at the one-step categorical level, not for the full joint sequence.

### Not claimed

1. That the contradiction-aware scale is oracle-optimal for EM or F1.
2. That trajectory-consistent EAMD is automatically faster in wall-clock time than every AR system under every implementation.
3. That iterative ARAM or iterative SPREAD cannot be augmented with trajectory memory; only that a naive memoryless iterative wrapper cannot replicate the v4 posterior.

## 1. Notation and Setup

Let $q$ be the question, and let
$$
C_0 \subseteq C_1 \subseteq \cdots \subseteq C_R
$$
be the nested evidence sets produced by iterative retrieval. Define the evidence increment
$$
\Delta C_r := C_r \setminus C_{r-1}, \qquad r \ge 1.
$$

Let
$$
p_r(x) := p_\theta(x \mid q, C_r)
$$
be the dLLM answer distribution under evidence $C_r$. As in `EAMD_MATH_V2.md`, define the log evidence ratio
$$
\Delta_r(x) := \log p_r(x) - \log p_{r-1}(x), \qquad r \ge 1.
$$

At token position $j$ and denoising step $t$, let
$$
\ell_{r,j} \in \mathbb R^{|\mathcal V|}
$$
be the logits under $C_r$, so that
$$
p_{r,j}(v) = \frac{\exp(\ell_{r,j}(v))}{\sum_{u \in \mathcal V}\exp(\ell_{r,j}(u))}.
$$

We reuse the diffusion reliability weight from Proposition 4.2 of `EAMD_MATH_V2.md`:
$$
w_t := 1 - \mu_t \in [0,1].
$$

## 2. Trajectory Posterior

### Definition 2.1. Trajectory posterior

Set
$$
q_0 := p_0.
$$
For each round $r \ge 1$, define the trajectory posterior $q_r$ as
$$
q_r(x)
:=
\frac{q_{r-1}(x)\exp(\alpha_r \Delta_r(x))}{Z_r},
$$
where
$$
Z_r := \sum_x q_{r-1}(x)\exp(\alpha_r \Delta_r(x))
$$
and $\alpha_r \ge 0$ is the round-$r$ guidance scale.

### Theorem 2.2. Multi-round trust-region update

For each round $r \ge 1$, the trajectory posterior $q_r$ is the unique optimizer of
$$
\max_{q \in \Delta(\mathcal X)}
\Big\{
\mathbb E_q[\Delta_r(X)] - \alpha_r^{-1} D_{\mathrm{KL}}(q \,\|\, q_{r-1})
\Big\}.
$$

#### Proof

Fix $r$. Consider the Lagrangian
$$
\mathcal L(q,\lambda)
=
\sum_x q(x)\Delta_r(x)
- \alpha_r^{-1}\sum_x q(x)\log\frac{q(x)}{q_{r-1}(x)}
+ \lambda\Big(\sum_x q(x)-1\Big).
$$
Taking the derivative with respect to $q(x)$ gives
$$
\frac{\partial \mathcal L}{\partial q(x)}
=
\Delta_r(x)
- \alpha_r^{-1}\Big(\log\frac{q(x)}{q_{r-1}(x)} + 1\Big)
+ \lambda.
$$
Setting this to zero yields
$$
\log\frac{q(x)}{q_{r-1}(x)}
=
\alpha_r\Delta_r(x) + \alpha_r(\lambda-1),
$$
hence
$$
q(x)
=
q_{r-1}(x)\exp(\alpha_r\Delta_r(x))\exp(\alpha_r(\lambda-1)).
$$
Normalizing over $x$ gives
$$
q(x)
=
\frac{q_{r-1}(x)\exp(\alpha_r\Delta_r(x))}{\sum_{x'} q_{r-1}(x')\exp(\alpha_r\Delta_r(x'))}
=
\frac{q_{r-1}(x)\exp(\alpha_r\Delta_r(x))}{Z_r}.
$$

Uniqueness follows because the objective is strictly concave in $q$: the expectation term is linear, and $-D_{\mathrm{KL}}(q\|q_{r-1})$ is strictly concave on the simplex. QED.

### Corollary 2.3. Global trajectory form

For every $R \ge 1$,
$$
q_R(x)
\propto
p_0(x)\exp\!\Big(\sum_{r=1}^R \alpha_r \Delta_r(x)\Big).
$$

#### Proof

By Definition 2.1,
$$
q_1(x) \propto q_0(x)\exp(\alpha_1\Delta_1(x))
=
p_0(x)\exp(\alpha_1\Delta_1(x)).
$$
Assume the formula holds for $R-1$. Then
$$
q_R(x)
\propto
q_{R-1}(x)\exp(\alpha_R\Delta_R(x))
\propto
p_0(x)\exp\!\Big(\sum_{r=1}^{R-1}\alpha_r\Delta_r(x)\Big)\exp(\alpha_R\Delta_R(x)),
$$
which is exactly
$$
q_R(x)
\propto
p_0(x)\exp\!\Big(\sum_{r=1}^{R}\alpha_r\Delta_r(x)\Big).
$$
QED.

### Remark 2.4. Why this is the right multi-round object

Theorem 2.2 is the trajectory analogue of Theorem 3.1 in `EAMD_MATH_V2.md`. In v2, one maximizes a one-step evidence-amplification objective relative to a single update $C_0 \to C_1$. In v4, one performs the same trust-region operation repeatedly, but with the previous trajectory posterior $q_{r-1}$ as the reference measure. The consequence is that the algorithm carries explicit evidence history rather than forgetting earlier rounds.

## 3. Contradiction Geometry

At token position $j$, define the local innovation vector
$$
d_{r,j} := \ell_{r,j} - \ell_{r-1,j}
$$
and the accumulated trajectory direction
$$
g_{r-1,j} := \bar\ell_{r-1,j} - \ell_{0,j},
$$
where $\bar\ell_{r-1,j}$ are the aggregated logits whose existence is justified in Section 5 below.

### Definition 3.1. Contradiction score

The contradiction score at round $r$, position $j$, is
$$
c_{r,j}
:=
\Bigg[
-\frac{\langle d_{r,j}, g_{r-1,j}\rangle}
{\|d_{r,j}\|_2 \, \|g_{r-1,j}\|_2 + \varepsilon_c}
\Bigg]_+,
$$
where $[u]_+ := \max\{u,0\}$ and $\varepsilon_c > 0$ avoids division by zero.

### Proposition 3.2. Range and extremal behavior

For all $r,j$,
$$
0 \le c_{r,j} \le 1.
$$
Moreover:

1. if $\langle d_{r,j}, g_{r-1,j}\rangle \ge 0$, then $c_{r,j} = 0$;
2. if $d_{r,j}$ and $g_{r-1,j}$ are perfectly anti-aligned and nonzero, then $c_{r,j} \to 1$ as $\varepsilon_c \downarrow 0$.

#### Proof

For nonzero vectors, the cosine term satisfies
$$
-1 \le \frac{\langle d_{r,j}, g_{r-1,j}\rangle}{\|d_{r,j}\|_2\|g_{r-1,j}\|_2} \le 1.
$$
Therefore
$$
-1 \le
-\frac{\langle d_{r,j}, g_{r-1,j}\rangle}{\|d_{r,j}\|_2\|g_{r-1,j}\|_2 + \varepsilon_c}
\le 1.
$$
Applying $[\cdot]_+$ yields $0 \le c_{r,j} \le 1$.

If $\langle d_{r,j}, g_{r-1,j}\rangle \ge 0$, the quantity inside $[\cdot]_+$ is nonpositive, so $c_{r,j}=0$.

If $d_{r,j}$ and $g_{r-1,j}$ are perfectly anti-aligned, then
$$
\langle d_{r,j}, g_{r-1,j}\rangle = -\|d_{r,j}\|_2\|g_{r-1,j}\|_2,
$$
so
$$
c_{r,j}
=
\frac{\|d_{r,j}\|_2\|g_{r-1,j}\|_2}{\|d_{r,j}\|_2\|g_{r-1,j}\|_2+\varepsilon_c}
\to 1
\qquad (\varepsilon_c \downarrow 0).
$$
QED.

### Remark 3.3. What the contradiction score detects

The score $c_{r,j}$ is zero when the new evidence update points in the same direction as the evidence trajectory accumulated so far. It becomes large only when the local innovation tries to undo or reverse the trajectory direction. This is exactly the failure mode that a purely local $C_{r-1}$-versus-$C_r$ rule misses.

## 4. Contradiction-Aware Guidance Scale

At position $j$, define the token random variable $Y_{r,j} \sim p_{r,j}$. Let the scalar local log-ratio be
$$
s_{r,j}(v) := \log p_{r,j}(v) - \log p_{r-1,j}(v).
$$
Define
$$
S_{r,j} := \mathbb E[s_{r,j}(Y_{r,j})],
\qquad
N_{r,j} := \mathrm{Var}(s_{r,j}(Y_{r,j})).
$$

### Proposition 4.1. Signal equals forward KL

For every $r,j$,
$$
S_{r,j} = D_{\mathrm{KL}}(p_{r,j} \,\|\, p_{r-1,j}) \ge 0.
$$

#### Proof

By definition,
$$
S_{r,j}
=
\sum_{v \in \mathcal V} p_{r,j}(v)\big(\log p_{r,j}(v) - \log p_{r-1,j}(v)\big),
$$
which is exactly $D_{\mathrm{KL}}(p_{r,j}\|p_{r-1,j})$. Nonnegativity is the standard nonnegativity of KL divergence. QED.

### Theorem 4.2. Local contradiction-aware optimal scale

Fix round $r$, token $j$, and denoising step $t$. Consider the local utility
$$
U_{r,j,t}(\alpha)
:=
w_t \alpha S_{r,j}
- \frac{1}{2}\alpha^2\big(N_{r,j} + \kappa c_{r,j} + \varepsilon\big),
\qquad
\alpha \ge 0,
$$
where $\kappa \ge 0$ and $\varepsilon > 0$.

Then $U_{r,j,t}$ is strictly concave in $\alpha$, and its unique maximizer is
$$
\alpha_{r,j,t}^*
=
\frac{w_t S_{r,j}}{N_{r,j} + \kappa c_{r,j} + \varepsilon}.
$$

#### Proof

Differentiate:
$$
\frac{dU_{r,j,t}}{d\alpha}
=
w_t S_{r,j}
- \alpha\big(N_{r,j} + \kappa c_{r,j} + \varepsilon\big).
$$
Setting the derivative to zero yields
$$
\alpha^*
=
\frac{w_t S_{r,j}}{N_{r,j} + \kappa c_{r,j} + \varepsilon}.
$$
The second derivative is
$$
\frac{d^2U_{r,j,t}}{d\alpha^2}
=
-\big(N_{r,j} + \kappa c_{r,j} + \varepsilon\big) < 0,
$$
so the utility is strictly concave and the stationary point is the unique maximizer. QED.

### Corollary 4.3. Monotonicity of the optimal scale

The optimal scale $\alpha_{r,j,t}^*$ is:

1. nondecreasing in the reliability weight $w_t$;
2. nondecreasing in the signal $S_{r,j}$;
3. nonincreasing in the noise $N_{r,j}$;
4. nonincreasing in the contradiction score $c_{r,j}$.

#### Proof

Immediate from the closed form in Theorem 4.2 by partial differentiation, using the positivity of the denominator. QED.

### Remark 4.4. Interpretation

Theorem 4.2 is the multi-round contradiction-aware analogue of the signal-to-noise logic in ARAM. The difference is that v4 treats contradiction with the accumulated trajectory as an additional uncertainty term rather than folding everything into a single one-shot context-versus-prior comparison.

## 5. Aggregated Logits

Suppose that at token position $j$,
$$
q_{r-1,j}(v) = \mathrm{softmax}(\bar\ell_{r-1,j})(v).
$$

### Theorem 5.1. Exact tokenwise aggregated-logit update

Let
$$
q_{r,j}(v)
\propto
q_{r-1,j}(v)\exp\!\big(\alpha_{r,j,t} s_{r,j}(v)\big),
$$
where $s_{r,j}(v)=\log p_{r,j}(v)-\log p_{r-1,j}(v)$. Then
$$
q_{r,j}
=
\mathrm{softmax}\!\Big(
\bar\ell_{r-1,j} + \alpha_{r,j,t}(\ell_{r,j} - \ell_{r-1,j})
\Big).
$$
Equivalently, the aggregated logits satisfy
$$
\bar\ell_{r,j}
=
\bar\ell_{r-1,j} + \alpha_{r,j,t}(\ell_{r,j} - \ell_{r-1,j})
$$
up to an additive constant vector $c\mathbf 1$, which softmax removes.

#### Proof

Write
$$
q_{r,j}(v)
\propto
q_{r-1,j}(v)\exp\!\big(\alpha_{r,j,t}(\log p_{r,j}(v)-\log p_{r-1,j}(v))\big).
$$
Because $q_{r-1,j}=\mathrm{softmax}(\bar\ell_{r-1,j})$, there exists a scalar $a$ such that
$$
\log q_{r-1,j}(v) = \bar\ell_{r-1,j}(v) - a.
$$
Likewise, there exist scalars $b_r,b_{r-1}$ such that
$$
\log p_{r,j}(v) = \ell_{r,j}(v) - b_r,
\qquad
\log p_{r-1,j}(v) = \ell_{r-1,j}(v) - b_{r-1}.
$$
Substituting,
$$
\log q_{r,j}(v)
=
\bar\ell_{r-1,j}(v) - a
+ \alpha_{r,j,t}\big(\ell_{r,j}(v)-\ell_{r-1,j}(v)\big)
- \alpha_{r,j,t}(b_r-b_{r-1})
- \log \widetilde Z_{r,j},
$$
where $\widetilde Z_{r,j}$ is the normalization constant. The terms
$$
- a - \alpha_{r,j,t}(b_r-b_{r-1}) - \log \widetilde Z_{r,j}
$$
do not depend on $v$, so they are absorbed by softmax. Therefore
$$
q_{r,j}
=
\mathrm{softmax}\!\Big(
\bar\ell_{r-1,j} + \alpha_{r,j,t}(\ell_{r,j} - \ell_{r-1,j})
\Big).
$$
QED.

### Remark 5.2. No extra forward pass

Theorem 5.1 matters algorithmically: once $\ell_{r-1,j}$ and $\ell_{r,j}$ are already available from the batched old/new-context forward pass, the contradiction-aware trajectory update requires only arithmetic on existing logits. It does not require a third model evaluation.

## 6. Single-Round Recovery and Connection to ARAM

### Corollary 6.1. Recovery of single-round EAMD

If $R=1$, then
$$
q_1(x)
\propto
p_0(x)\exp(\alpha_1(\log p_1(x)-\log p_0(x))).
$$
Writing $\alpha_1 = 1 + \gamma$, this becomes
$$
q_1(x)
\propto
p_1(x)\exp(\gamma(\log p_1(x)-\log p_0(x))),
$$
which is exactly the single-round EAMD form of Theorem 3.1 in `EAMD_MATH_V2.md`.

#### Proof

By Corollary 2.3 with $R=1$,
$$
q_1(x)
\propto
p_0(x)\exp(\alpha_1(\log p_1(x)-\log p_0(x))).
$$
Set $\alpha_1=1+\gamma$. Then
$$
q_1(x)
\propto
p_0(x)\exp((1+\gamma)\log(p_1(x)/p_0(x)))
=
p_0(x)\frac{p_1(x)^{1+\gamma}}{p_0(x)^{1+\gamma}}
=
p_1(x)\Big(\frac{p_1(x)}{p_0(x)}\Big)^\gamma,
$$
which is
$$
q_1(x)
\propto
p_1(x)\exp(\gamma(\log p_1(x)-\log p_0(x))).
$$
QED.

### Proposition 6.2. Structural connection to ARAM

In the single-round setting $R=1$, if

1. $p_0$ is chosen as the prior or no-context branch,
2. $c_{1,j}=0$,
3. the scale is chosen by the local signal-to-noise rule of Theorem 4.2,

then v4 reduces to a single-update SNR-guided contradiction-free rule with the same structural form as ARAM: guidance strength increases with contextual signal and decreases with uncertainty.

#### Proof

Under $R=1$, Corollary 6.1 shows that v4 reduces exactly to a one-step evidence-ratio guidance rule. Under $c_{1,j}=0$, Theorem 4.2 gives
$$
\alpha_{1,j,t}^* = \frac{w_t S_{1,j}}{N_{1,j}+\varepsilon},
$$
which is precisely a signal-over-noise structure modulated by denoising reliability. This is structurally identical to the ARAM principle of increasing guidance when the contextual branch is informative and decreasing it when uncertainty is high. The statement is structural rather than an identity of estimator formulas, because ARAM and v4 need not use the same observable estimators of signal and noise. QED.

## 7. Why Naive Iterative ARAM/SPREAD Cannot Replicate v4

### Definition 7.1. Memoryless iterative guidance rule

A memoryless iterative guidance rule is any rule of the form
$$
\widetilde q_r = F_r(p_{r-1}, p_r),
$$
where the output at round $r$ depends only on the current pair $(p_{r-1}, p_r)$ and not on earlier history except through that pair.

### Theorem 7.2. Non-replicability of the trajectory posterior by memoryless wrappers

There is no family of memoryless rules $F_r$ that reproduces the v4 trajectory posterior $q_r$ for every possible evidence trajectory $(p_0,\dots,p_r)$, unless all earlier update weights are identically zero.

#### Proof

It is enough to consider $r=2$. Fix any $p_1$, $p_2$, and any $\alpha_2 > 0$. Construct two histories:

- History A: $(p_0^{A}, p_1, p_2)$,
- History B: $(p_0^{B}, p_1, p_2)$,

with $p_0^{A} \neq p_0^{B}$ and $\alpha_1 > 0$.

By Corollary 2.3,
$$
q_1^{A}(x) \propto p_0^{A}(x)\exp(\alpha_1(\log p_1(x)-\log p_0^{A}(x))),
$$
$$
q_1^{B}(x) \propto p_0^{B}(x)\exp(\alpha_1(\log p_1(x)-\log p_0^{B}(x))).
$$
Because $p_0^{A} \neq p_0^{B}$ and $\alpha_1>0$, these two distributions are unequal for a generic choice of $p_1$.

Now update to round $2$:
$$
q_2^{A}(x) \propto q_1^{A}(x)\exp(\alpha_2(\log p_2(x)-\log p_1(x))),
$$
$$
q_2^{B}(x) \propto q_1^{B}(x)\exp(\alpha_2(\log p_2(x)-\log p_1(x))).
$$
Since the multiplicative factor $\exp(\alpha_2(\log p_2-\log p_1))$ is the same in both histories, any difference between $q_1^A$ and $q_1^B$ persists into a difference between $q_2^A$ and $q_2^B$ after renormalization, unless $q_1^A=q_1^B$, which is not true in general.

However, the current pair $(p_1,p_2)$ is identical in both histories. Therefore any memoryless rule $F_2(p_1,p_2)$ must return the same output for histories A and B, while the true v4 trajectory posterior requires different outputs. So no such universal memoryless rule exists.

The only degenerate exception is when all earlier update weights are zero, in which case the history is never used. QED.

### Corollary 7.3. Implication for iterative ARAM and iterative SPREAD

Any iterative wrapper around a single-round method that recomputes guidance only from the current round and does not store the trajectory posterior $q_{r-1}$ cannot in general reproduce v4.

#### Proof

Immediate from Theorem 7.2. QED.

### Remark 7.4. Honest scope of the non-replicability theorem

Theorem 7.2 does **not** say that ARAM or SPREAD could never be extended to include trajectory memory. It says only that a naive iterative wrapper that remains memoryless at each round cannot reproduce the v4 posterior. That is the relevant fairness statement for baseline design.

## 8. Practical Algorithm

### Algorithm 1. Trajectory-consistent contradiction-aware EAMD

Inputs:

- question $q$
- evidence sets $C_0 \subseteq \cdots \subseteq C_R$
- denoising steps $t=T,\dots,1$
- batched logits $(\ell_{r-1,j}, \ell_{r,j})$ at each step

State:

- trajectory logits $\bar\ell_{0,j} := \ell_{0,j}$

For each round $r=1,\dots,R$:

1. Retrieve new evidence and form $C_r$.
2. For each denoising step $t$ and active answer position $j$:
   - compute $p_{r-1,j}, p_{r,j}$;
   - form the local innovation vector $d_{r,j} = \ell_{r,j} - \ell_{r-1,j}$;
   - form the trajectory direction $g_{r-1,j} = \bar\ell_{r-1,j} - \ell_{0,j}$;
   - compute the contradiction score $c_{r,j}$ from Definition 3.1;
   - compute the signal $S_{r,j} = D_{\mathrm{KL}}(p_{r,j}\|p_{r-1,j})$;
   - compute the noise $N_{r,j} = \mathrm{Var}_{p_{r,j}}[\log p_{r,j}(Y)-\log p_{r-1,j}(Y)]$;
   - set
     $$
     \alpha_{r,j,t}
     =
     \frac{w_t S_{r,j}}{N_{r,j} + \kappa c_{r,j} + \varepsilon};
     $$
   - update the aggregated logits
     $$
     \bar\ell_{r,j}
     =
     \bar\ell_{r-1,j} + \alpha_{r,j,t}(\ell_{r,j} - \ell_{r-1,j}).
     $$
3. Decode from $\bar\ell_r$.
4. Continue to the next retrieval round using the newly decoded answer / bridge hypotheses.

### Remark 8.1. Contradiction-triggered re-retrieval

The core v4 guidance law above is theorem-backed. A practical contradiction-triggered re-retrieval rule can then be layered on top: if contradiction mass concentrates on bridge-bearing positions, query expansion should branch on the competing bridge hypotheses. That policy choice is downstream of the posterior update and is not needed for Theorems 2.2 through 7.2.

## 9. Efficiency Accounting

### Proposition 9.1. Exact model-call counts for trajectory memory

Suppose an iterative dLLM decode uses $T$ denoising steps per round and $R$ retrieval rounds, and suppose the old/new-context branches are evaluated in a single batched forward pass at each step. Then:

1. vanilla iterative decoding without guidance uses $RT$ model forward calls;
2. pairwise guided iterative decoding also uses $RT$ model forward calls under batched old/new evaluation;
3. trajectory-consistent contradiction-aware EAMD uses the same $RT$ model forward calls;
4. the trajectory-memory, contradiction, and scale computations add only post-logit arithmetic and no extra model calls.

#### Proof

At each round and denoising step, one batched model call returns the logits for the old and new evidence branches. This is one model forward call, not two separate model calls. There are $T$ such calls per round and $R$ rounds, so the total is $RT$. The trajectory update in Sections 3 to 5 consumes only the already computed logits, requiring dot products, variances, and vector additions, but no extra model invocation. QED.

### Corollary 9.2. Comparison to IRCoT-style AR decoding

Let an AR iterative retriever-generator perform $R$ rounds, with $L_r$ generated tokens in round $r$. Then the number of AR model calls is
$$
\sum_{r=1}^R L_r,
$$
and the generation is strictly sequential across tokens and rounds. By contrast, v4 uses exactly $RT$ dLLM model calls under a fixed-step denoising schedule, with old/new evidence branches batched into the same call.

#### Proof

In an AR decoder, each new token requires one model call conditioned on all preceding tokens, so round $r$ needs $L_r$ sequential calls. Summing over rounds gives $\sum_r L_r$. Proposition 9.1 gives the dLLM count $RT$. The batching statement follows directly from the old/new-context batched evaluation assumption. QED.

### Remark 9.3. What the efficiency result does and does not prove

Proposition 9.1 and Corollary 9.2 are exact call-count statements. They do **not** by themselves prove lower wall-clock latency than every IRCoT implementation, because wall-clock depends on hardware kernels, retrieval latency, and batching efficiency. What they do prove is that trajectory consistency and contradiction handling add no extra model calls beyond the underlying pairwise guided iterative decode.

## 10. Summary

The mathematically clean v4 picture is:

1. Maintain a trajectory posterior $q_r$, not just the latest raw posterior $p_r$.
2. Admit new evidence only through a reliability-weighted local innovation term.
3. Penalize updates that contradict the accumulated trajectory.
4. Decode from aggregated logits $\bar\ell_r$, which are exact tokenwise natural parameters of $q_r$.
5. Preserve the single-round EAMD limit when $R=1$.
6. Obtain a formal non-replicability result for naive memoryless iterative wrappers.

This is the training-free trajectory-consistent contradiction-aware extension of EAMD.

## References

- `EAMD_MATH_V2.md`
- `EAMD_MATH_V3_DIR.md`
- ARAM: https://arxiv.org/abs/2603.17677
- SPREAD: https://arxiv.org/abs/2601.11342
