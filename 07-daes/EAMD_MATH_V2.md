# EAMD v2: Evidence-Geodesic Inference for Iterative Evidence Refinement

This note replaces the heuristic EAMD v1 story with a theorem-safe formulation.

The central change is conceptual:
- **ARAM** compares `context` vs `no-context` and addresses retrieval-prior conflict.
- **EAMD** compares **two evidence sets** $C_0 \to C_1$, where $C_1 = C_0 \cup \Delta C_1$, and addresses **iterative evidence refinement**.

The right object for EAMD is therefore not "context helps more than no-context" but:

> how the answer posterior changes when we replace stale evidence $C_0$ by refined evidence $C_1$.

This document is organized so that every statement is either:
- proved,
- bounded with explicit assumptions, or
- marked as a conjecture.

## 0. Status of Claims

### Theorem-backed in this document

1. The inference-time EAMD guided distribution is the exact optimizer of a KL trust-region variational problem.
2. The natural evidence information gain for $C_0 \to C_1$ is the forward KL
   $D_KL(p_1 || p_0)$, and its expectation over retrieval randomness equals a conditional mutual information.
3. The symmetric KL (Jeffreys divergence) arises as the change in expected evidence score between the $C_1$ and $C_0$ branches.
4. Multi-round refinement monotonically improves the **exact Bayesian answer log-likelihood** whenever the added evidence contributes non-zero conditional mutual information.
5. A principled remask probability follows from a local variational decision problem on top of the ReMDM generalized posterior family.

### Approximation-backed in this document

1. A closed-form scalar guidance rule can be obtained from a second-order expansion of the exact trust-region solution.
2. A retrieval-recall improvement bound can be proved under an explicit latent-bridge coverage model.

### Conjectural in this document

1. Monotonic improvement of downstream QA metrics such as EM/F1 for the approximate model.
2. A model-mismatch statement that the best oracle posterior can lie beyond $p_1$ on the $e$-geodesic from $p_0$ to $p_1$.

## 1. Notation and Preliminaries

We follow the notation of `EAMD_MATHEMATICAL_FORMULATION.md`

Let:
- $V$ be the vocabulary,
- $m$ be the absorbing mask token,
- $Q$ be the question,
- $C$ be a set of retrieved passages,
- $n$ be the fixed short-answer canvas length,
- $x_t \in (V \cup \{m\})^n$ be the answer canvas at denoising step $t$,
- $X_0$ denote the random clean answer sequence,
- $p_theta(. | x_t, Q, C, t)$ be the denoiser distribution over answer tokens.

For a fixed answer position $i$, we write the token posterior as

$$
p_{theta,C}^{(t,i)}(x)
:= p_theta(X_0^i = x | x_t, Q, C, t),
qquad x in V.
$$

For EAMD we use two evidence sets:

$$
C_0 = R(Q),
qquad
C_1 = C_0 \cup \Delta C_1,
$$

with

$$
\Delta C_1 = R(Q \oplus \hat a_0) cup \bigcup_{h in H_1} R(Q \oplus h).
$$

Here:
- $\hat a_0$ is the current answer hypothesis under $C_0$,
- $H_1$ is the bridge-candidate set extracted from the dLLM posterior,
- $R(.)$ is the retriever.

By construction, $C_0 \subseteq C_1$.

### 1.1 Absorbing-state masked diffusion

As in MDLM and Dream, the forward noising process is absorbing-mask diffusion:

$$
q(x_t^i = m | x_0^i) = \mu_t,
\qquad
q(x_t^i = x_0^i | x_0^i) = 1 - \mu_t,
$$

with $0 = \mu_0 <= ... <= \mu_T = 1$.

Equivalently, if $\alpha_t := 1 - \mu_t$, a token survives the noising process with probability $\alpha_t$ and is masked otherwise.

### 1.2 MDLM ELBO and the local denoising objective

For evidence $C$, the standard masked-diffusion ELBO is

$$
\mathcal L_{\mathrm{ELBO}}(theta; Q,C)
= \mathbb E_{q(X_0)} \mathbb E_{q(X_{1:T}|X_0)}
\Big[
\log p_theta(X_T)
+ \sum_{t=1}^T \log p_theta(X_{t-1} | X_t, Q, C, t)
- \log q(X_t | X_{t-1}, X_0)
\Big].
$$

For absorbing-state discrete diffusion, MDLM shows that this objective reduces, up to constants and timestep reweighting, to reconstruction of the clean token at masked positions. Therefore the inference object relevant to EAMD is the family of local posteriors $p_{theta,C}^{(t,i)}$.

We now fix one denoising step $t$ and one answer position $i$, and abbreviate

$$
p_0(x) := p_{theta,C_0}^{(t,i)}(x),
\qquad
p_1(x) := p_{theta,C_1}^{(t,i)}(x).
$$

Everything below is local in $(t,i)$; the full sampler applies the construction independently at each masked answer position and denoising step.

## 2. Evidence Information Gain for $C_0 \to C_1$

Define the **evidence score ratio**

$$
r(x) := \log \frac{p_1(x)}{p_0(x)},
\qquad x \in V.
$$

This is the exact analog of ARAM's log-likelihood ratio, but with $(p_cond, p_prior)$ replaced by $(p_1, p_0)$.

### Proposition 2.1. Forward evidence information gain

The natural information gain contributed by the refined evidence $\Delta C_1$ at token $(t,i)$ is

$$
\mathrm{IG}_{t,i}(\Delta C_1)
:= D_{\mathrm{KL}}(p_1 || p_0)
= \mathbb E_{p_1}[r(X)].
$$

#### Proof

By definition of KL divergence,

$$
D_{\mathrm{KL}}(p_1 || p_0)
= \sum_{x \in V} p_1(x) \log \frac{p_1(x)}{p_0(x)}
= \mathbb E_{p_1}[r(X)].
$$

This is exactly the expected log-likelihood ratio in favor of refined evidence over stale evidence. QED.

### Proposition 2.2. Conditional mutual information identity

Treat $\Delta C_1$ as a random variable produced by the retriever conditioned on $(Q, C_0)$. Then

$$
I(X_0^i; \Delta C_1 | x_t, Q, C_0)
= \mathbb E_{\Delta C_1}
\left[
D_{\mathrm{KL}}\bigl(p_{theta,C_1}^{(t,i)} || p_{theta,C_0}^{(t,i)}\bigr)
\right].
$$

#### Proof

Fix $(x_t,Q,C_0)$. By the definition of conditional mutual information,

$$
I(X_0^i; \Delta C_1 | x_t, Q, C_0)
= \mathbb E_{\Delta C_1}
\left[
D_{\mathrm{KL}}
\left(
P(X_0^i | x_t,Q,C_0,\Delta C_1)
\Big\|
P(X_0^i | x_t,Q,C_0)
\right)
\right].
$$

Since $C_1 = C_0 \cup \Delta C_1$, we identify

$$
P(X_0^i | x_t,Q,C_0,\Delta C_1) = p_{theta,C_1}^{(t,i)},
\qquad
P(X_0^i | x_t,Q,C_0) = p_{theta,C_0}^{(t,i)}.
$$

Substituting gives the result. QED.

This proposition is the key distinction from ARAM:
- ARAM's information gain is about $context$ versus $prior/no-context$.
- EAMD's information gain is about the **incremental value of refined evidence** $\Delta C_1$ relative to the already-available evidence $C_0$.

### Proposition 2.3. Why symmetric KL appears

Define the Jeffreys divergence

$$
J(p_1,p_0)
:= D_{\mathrm{KL}}(p_1 || p_0) + D_{\mathrm{KL}}(p_0 || p_1).
$$

Then

$$
J(p_1,p_0) = \mathbb E_{p_1}[r(X)] - \mathbb E_{p_0}[r(X)].
$$

#### Proof

Using the definition of $r$,

$$
\mathbb E_{p_1}[r(X)] = D_{\mathrm{KL}}(p_1 || p_0),
$$

and

$$
\mathbb E_{p_0}[r(X)]
= \sum_x p_0(x) \log \frac{p_1(x)}{p_0(x)}
= - D_{\mathrm{KL}}(p_0 || p_1).
$$

Subtracting gives

$$
\mathbb E_{p_1}[r(X)] - \mathbb E_{p_0}[r(X)]
= D_{\mathrm{KL}}(p_1 || p_0) + D_{\mathrm{KL}}(p_0 || p_1)
= J(p_1,p_0).
$$

QED.

Therefore:
- the **fundamental** information quantity is the forward KL $D_{KL}(p_1 || p_0)$, because it is the exact conditional mutual information integrand,
- while the **symmetric** KL appears because it measures how differently the two branches score the same evidence-ratio statistic.

This is the first-principles justification for using symmetric KL as a scalar guidance signal: it is not an arbitrary symmetrization, but the gap in expected evidence score between the refined branch and the stale branch.

### Relation to ARAM Eq. (5)-(13)

EAMD inherits ARAM's information-theoretic template but changes the objects being compared. In ARAM Eq. (5)-(8), the score ratio is between $p_cond$ and $p_prior$; in EAMD the corresponding ratio is $r(x)=log(p_1(x)/p_0(x))$, where $p_0$ and $p_1$ are the stale- and refined-evidence posteriors. ARAM's DV program in Eq. (9)-(13) analyzes how strongly one should trust retrieved context over the prior. EAMD instead solves a trust-region posterior-transport problem on the evidence-refinement pair $(C_0,C_1)$. The resulting family $q_gamma propto p_1 exp(gamma r)$ is therefore an evidence-refinement analog of ARAM's adaptive guidance, but it is derived for iterative evidence updates rather than retrieval-prior conflict.

## 3. Variational Objective for EAMD Guidance

We now derive the EAMD guidance distribution from an explicit optimization problem.

The object we want is:
- remain close to the updated-evidence posterior $p_1$, because $p_1$ is the denoiser's direct response to $C_1$,
- but emphasize tokens whose odds increased the most from $C_0$ to $C_1$.

This leads to a KL trust-region objective.

### Theorem 3.1. Trust-region evidence amplification

For a fixed token position $(t,i)$, consider the optimization problem

$$
\max_{q in \Delta(V)} \; \mathbb E_q[r(X)]
\quad
\text{subject to}
\quad
D_{\mathrm{KL}}(q || p_1) <= \varepsilon,
$$

where $\Delta(V)$ is the probability simplex over $V$ and $\varepsilon >= 0$ is a trust-region radius.

Then the unique optimizer is

$$
q_\gamma(x)
= \frac{p_1(x) \exp(\gamma r(x))}{Z(\gamma)}
= \frac{p_1(x)^{1+\gamma} p_0(x)^{-\gamma}}{Z(\gamma)},
$$

for some unique $\gamma >= 0$ chosen so that the KL constraint is active whenever $\varepsilon > 0$. The logit form is

$$
\log q_\gamma(x)
= \ell_1(x) + \gamma\bigl(\ell_1(x)-\ell_0(x)\bigr) - \log Z(\gamma),
$$

where $\ell_0 = \log p_0$ and $\ell_1 = \log p_1$ up to additive constants.

#### Proof

The feasible set $\{q in \Delta(V): D_{KL}(q||p_1) <= \varepsilon\}$ is compact and convex, and the objective $\mathbb E_q[r]$ is continuous and linear in $q$. Therefore an optimizer exists.

Form the Lagrangian

$$
\mathcal J(q,\eta,\nu)
= \sum_{x \in V} q(x) r(x)
- \eta\Big(\sum_x q(x)\log\frac{q(x)}{p_1(x)} - \varepsilon\Big)
+ \nu\Big(\sum_x q(x)-1\Big),
$$

with $\eta >= 0$ and normalization multiplier $\nu$.

For any interior optimum, differentiating with respect to $q(x)$ gives

$$
r(x) - \eta\Big(\log q(x)-\log p_1(x)+1\Big) + \nu = 0.
$$

Rearranging,

$$
\log q(x) = \log p_1(x) + \frac{1}{\eta} r(x) + \frac{\nu-\eta}{\eta}.
$$

Hence

$$
q(x) = \frac{p_1(x)\exp(\gamma r(x))}{Z(\gamma)},
\qquad \gamma := \frac{1}{\eta} >= 0.
$$

Using $r(x)=\log(p_1(x)/p_0(x))$,

$$
q_\gamma(x)
\propto p_1(x)\exp\Big(\gamma\log\frac{p_1(x)}{p_0(x)}\Big)
= p_1(x)^{1+\gamma} p_0(x)^{-\gamma}.
$$

Uniqueness follows from strict convexity of $D_{KL}(q||p_1)$ in $q$. The stated logit form is immediate by taking logs. QED.

### Proposition 3.1A. Why maximizing $E_q[r]$ is the right QA objective

Assume there exists an oracle answer distribution $p^*$ at token $(t,i)$ that lies on the evidence geodesic generated by $(p_0,p_1)$, i.e.,

$$
p^*(x) = q_{\gamma^*}(x)
= \frac{p_1(x)\exp(\gamma^* r(x))}{Z(\gamma^*)}
$$

for some $\gamma^* >= 0$. Then for any candidate distribution $q$,

$$
D_{\mathrm{KL}}(q || p^*)
= D_{\mathrm{KL}}(q || p_1) - \gamma^* \mathbb E_q[r(X)] + A(\gamma^*),
$$

where $A(\gamma^*) = \log Z(\gamma^*)$. Consequently:

1. maximizing $\gamma^* \mathbb E_q[r(X)] - D_{\mathrm{KL}}(q||p_1)$ is exactly equivalent to minimizing $D_{\mathrm{KL}}(q||p^*)$,
2. on every trust-region shell $D_{\mathrm{KL}}(q||p_1)=\varepsilon$, maximizing $\mathbb E_q[r(X)]$ is exactly equivalent to minimizing $D_{\mathrm{KL}}(q||p^*)$,
3. therefore Theorem 3.1 is the primal trust-region form of oracle answer-risk minimization under a proximity constraint to $p_1$.

#### Proof

Because $p^*(x)=p_1(x)\exp(\gamma^* r(x)-A(\gamma^*))$,

$$
\log p^*(x) = \log p_1(x) + \gamma^* r(x) - A(\gamma^*).
$$

Hence

$$
D_{\mathrm{KL}}(q||p^*)
= \sum_x q(x)\log\frac{q(x)}{p^*(x)}
= \sum_x q(x)\log\frac{q(x)}{p_1(x)} - \gamma^* \sum_x q(x)r(x) + A(\gamma^*)
= D_{\mathrm{KL}}(q||p_1) - \gamma^* \mathbb E_q[r(X)] + A(\gamma^*).
$$

This proves item 1. If $D_{\mathrm{KL}}(q||p_1)=\varepsilon$ is fixed, the first and third terms are constants, so minimizing $D_{\mathrm{KL}}(q||p^*)$ is equivalent to maximizing $\mathbb E_q[r(X)]$, proving item 2. Item 3 follows because Theorem 3.1 solves the constrained maximization of $\mathbb E_q[r(X)]$ over the KL ball around $p_1$, whose active-boundary solution is therefore the I-projection toward the oracle geodesic target. QED.

### Remark 3.1B. Approximate oracle robustness

If the oracle is only approximately geodesic, i.e.

$$
\log p^*(x) = \log p_1(x) + \gamma^* r(x) + \delta(x) - A,
\qquad \|\delta\|_\infty <= \zeta,
$$

then

$$
\big| D_{\mathrm{KL}}(q||p^*) - \big(D_{\mathrm{KL}}(q||p_1) - \gamma^* \mathbb E_q[r(X)] + A\big) \big|
<= \zeta.
$$

So maximizing $\mathbb E_q[r]$ under the trust region still optimizes the oracle KL objective up to additive model-mismatch $\zeta$.

### Corollary 3.2. EAMD guidance is an information-geometric $e$-geodesic extrapolation

Let

$$
q_\beta(x) \propto p_0(x)^{1-\beta} p_1(x)^\beta,
\qquad \beta \in [0,\infty).
$$

Then $q_\beta$ is the exponential-geodesic between $p_0$ and $p_1$, with:
- $\beta = 0$ giving $p_0$,
- $\beta = 1$ giving $p_1$,
- $\beta > 1$ extrapolating beyond $p_1$ in the evidence-ratio direction.

The EAMD family above is exactly $q_\beta$ with $\beta = 1+\gamma$.

#### Proof

Substitute $\beta = 1+\gamma$ into Theorem 3.1:

$$
q_\gamma(x) \propto p_1(x)^{1+\gamma} p_0(x)^{-\gamma}
= p_0(x)^{1-(1+\gamma)} p_1(x)^{1+\gamma}.
$$

This is precisely the exponential-geodesic continuation beyond $p_1$. QED.

This gives a clean interpretation of EAMD:
- ARAM is a context-vs-prior adaptive guidance rule,
- EAMD is **posterior transport along the evidence-refinement geodesic** generated by $C_0 \to C_1$.

### Theorem 3.3. Existence and characterization of the oracle-optimal guidance scale

Let $\pi$ be an oracle target distribution over the correct token at $(t,i)$ under ideal refined evidence. Define

$$
F(\gamma) := \mathbb E_{\pi}[\log q_\gamma(X)].
$$

Then:

1. $F(\gamma)$ is concave in $\gamma$.
2. If $\mathrm{Var}_{q_\gamma}(r) > 0$, then $F$ is strictly concave.
3. Any maximizer $\gamma^*$ satisfies the moment-matching equation

$$
\mathbb E_{\pi}[r(X)] = \mathbb E_{q_{\gamma^*}}[r(X)].
$$

4. If $\mathbb E_{\pi}[r(X)] >= \mathbb E_{p_1}[r(X)]$, then $\gamma^* >= 0$. If the inequality is strict, then $\gamma^* > 0$.

#### Proof

By Theorem 3.1,

$$
\log q_\gamma(x) = \log p_1(x) + \gamma r(x) - A(\gamma),
\qquad
A(\gamma) := \log \sum_x p_1(x) e^{\gamma r(x)}.
$$

Hence

$$
F(\gamma)
= \mathbb E_{\pi}[\log p_1(X)] + \gamma \mathbb E_{\pi}[r(X)] - A(\gamma).
$$

The first term is constant in $\gamma$. Since $A$ is the log-partition function of a finite exponential family, it is convex, with

$$
A'(\gamma) = \mathbb E_{q_\gamma}[r(X)],
\qquad
A''(\gamma) = \mathrm{Var}_{q_\gamma}(r(X)) >= 0.
$$

Therefore

$$
F''(\gamma) = -A''(\gamma) = -\mathrm{Var}_{q_\gamma}(r(X)) <= 0,
$$

so $F$ is concave, and strictly concave whenever the variance is positive.

Differentiating,

$$
F'(\gamma) = \mathbb E_{\pi}[r(X)] - \mathbb E_{q_\gamma}[r(X)].
$$

Hence any maximizer must satisfy $F'(\gamma^*)=0$, which yields the stated moment-matching equation.

Finally, at $\gamma = 0$ we have $q_0 = p_1$, so

$$
F'(0) = \mathbb E_{\pi}[r(X)] - \mathbb E_{p_1}[r(X)].
$$

If this quantity is nonnegative, the maximizer of a concave function cannot lie to the left of $0$; if it is strictly positive, strict concavity implies $\gamma^* > 0$. QED.

This theorem is the clean answer to item (1) in the request:
- an optimal EAMD guidance scale exists,
- it is defined by an explicit objective,
- and it is positive exactly when the oracle posterior lies further in the evidence-ratio direction than $p_1$.

## 4. From Information Gain to Practical Guidance Strength

The exact trust-region solution gives a family $q_\gamma$, but we still need a scalar calibration rule.

### 4.1 Exact trust-region calibration

Define the log-partition function

$$
A(\gamma) := \log \mathbb E_{p_1}[e^{\gamma r(X)}].
$$

Then

$$
D_{\mathrm{KL}}(q_\gamma || p_1) = \gamma A'(\gamma) - A(\gamma).
$$

### Proposition 4.1. Monotonicity of the trust radius

Let

$$
D(\gamma) := D_{\mathrm{KL}}(q_\gamma || p_1).
$$

Then for $\gamma >= 0$,

$$
D'(\gamma) = \gamma \, \mathrm{Var}_{q_\gamma}(r(X)) >= 0.
$$

Hence $D(\gamma)$ is nondecreasing, and strictly increasing whenever $\mathrm{Var}_{q_\gamma}(r) > 0$.

#### Proof

Differentiate $D(\gamma)=\gamma A'(\gamma)-A(\gamma)$:

$$
D'(\gamma) = A'(\gamma) + \gamma A''(\gamma) - A'(\gamma) = \gamma A''(\gamma).
$$

Since $A''(\gamma)=\mathrm{Var}_{q_\gamma}(r)>=0$, the claim follows. QED.

Therefore, for every trust budget $\varepsilon >= 0$, there is a unique $\gamma_\varepsilon >= 0$ such that

$$
D_{\mathrm{KL}}(q_{\gamma_\varepsilon} || p_1) = \varepsilon,
$$

provided $r$ is not almost surely constant.

This is the exact EAMD guidance rule: choose the trust budget, then solve for the unique $\gamma$.

### 4.2 Denoising-time reliability from the forward noise schedule

The guidance strength should depend not only on the evidence update $(C_0,C_1)$ but also on how informative the current denoising state $x_t$ is. This dependence can be derived from the absorbing-mask channel itself.

### Proposition 4.2. Local denoising reliability equals the survival probability

Let $Y_t^i$ denote the direct channel output at token position $i$ under the absorbing-mask forward process:

$$
Y_t^i =
\begin{cases}
X_0^i, & \text{with probability } \alpha_t = 1-\mu_t,\\
m, & \text{with probability } \mu_t.
\end{cases}
$$

Then

$$
I(X_0^i ; Y_t^i) = \alpha_t H(X_0^i).
$$

Equivalently, if one writes the stepwise diffusion SNR as

$$
\mathrm{SNR}_t := \frac{\alpha_t}{1-\alpha_t} = \frac{1-\mu_t}{\mu_t},
$$

then the corresponding reliability weight is

$$
w_t := \frac{\mathrm{SNR}_t}{1+\mathrm{SNR}_t} = \alpha_t = 1-\mu_t.
$$

#### Proof

Condition on $Y_t^i$. If $Y_t^i != m$, then $Y_t^i = X_0^i$ and the token is known exactly, so $H(X_0^i | Y_t^i != m)=0$. If $Y_t^i = m$, the channel output reveals no symbol identity, so the posterior over $X_0^i$ equals the prior and $H(X_0^i | Y_t^i = m)=H(X_0^i)$. Therefore

$$
H(X_0^i | Y_t^i)
= \alpha_t \cdot 0 + \mu_t H(X_0^i)
= \mu_t H(X_0^i).
$$

Hence

$$
I(X_0^i ; Y_t^i)
= H(X_0^i) - H(X_0^i | Y_t^i)
= (1-\mu_t) H(X_0^i)
= \alpha_t H(X_0^i).
$$

The SNR identity is algebraic. QED.

This gives the principled denoising-time schedule:

$$
w_t := \alpha_t = 1-\mu_t.
$$

Early in denoising, $\mu_t$ is large and $w_t$ is small, so guidance should be weak. Late in denoising, $\mu_t$ is small and $w_t$ is large, so guidance can be stronger. If the noise schedule is linear in denoising time, $\alpha_t = t/T$, then this reduces exactly to the v1 linear schedule.

### 4.3 Local second-order approximation

The exact rule above is already rigorous. We now derive a closed-form approximation.

Let $m_1 := \mathbb E_{p_1}[r(X)] = D_{KL}(p_1||p_0)$ and $V_1 := \mathrm{Var}_{p_1}(r(X))$. Since

$$
A'(0) = m_1,
\qquad
A''(0) = V_1,
$$

a Taylor expansion around $\gamma = 0$ gives

$$
A(\gamma) = \gamma m_1 + \frac{\gamma^2}{2} V_1 + R_3(\gamma),
$$

where, if $|r(X)-m_1| <= B$ almost surely, the remainder satisfies the bound

$$
|R_3(\gamma)|
<= \frac{|\gamma|^3}{6} B^3 e^{|\gamma|B}.
$$

Substituting into $D(\gamma)=\gamma A'(\gamma)-A(\gamma)$ yields

$$
D(\gamma) = \frac{\gamma^2}{2} V_1 + \widetilde R_3(\gamma),
$$

with $\widetilde R_3(\gamma)=O(|\gamma|^3)$.

Therefore, for sufficiently small trust budgets,

$$
\gamma_\varepsilon
= \sqrt{\frac{2\varepsilon}{V_1}} + O(\varepsilon).
$$

#### Proof

The expansion of $A$ is standard Taylor's theorem with remainder because $A$ is smooth on a finite simplex. The derivative identities follow from exponential-family calculus. Expanding $D(\gamma)=\gamma A'(\gamma)-A(\gamma)$ and canceling the first-order term gives the stated result. QED.

### 4.4 Canonical trust-budget map from Theorem 3.3

Theorem 3.3 gives the exact oracle first-order condition

$$
\mathbb E_{\pi}[r] = \mathbb E_{q_{\gamma^*}}[r].
$$

To turn this into an observable rule, we need one extra assumption linking the unobserved oracle score gap to the observed evidence update.

### Assumption 4.3. Self-similar refinement

At a fixed denoising step and token position, the first-order oracle score excess above $p_1$ is proportional to the already observed evidence gain:

$$
\mathbb E_{\pi}[r] - \mathbb E_{p_1}[r]
= \kappa_t \, \mathrm{IG}_t + o(\mathrm{IG}_t),
\qquad \kappa_t >= 0,
$$

where $\mathrm{IG}_t := D_{\mathrm{KL}}(p_1||p_0)$. The **canonical self-consistent choice** is $\kappa_t = 1$, meaning that the next unobserved refinement step is expected to contribute the same first-order evidence score as the current observed refinement $C_0 \to C_1$.

### Proposition 4.4. Observable approximation to the oracle-optimal guidance scale

Under Assumption 4.3 and the finite-third-moment condition from Section 4.3,

$$
\gamma_t^* = \frac{\kappa_t \, \mathrm{IG}_t}{V_t} + O\!\left(\frac{\mathrm{IG}_t^2}{V_t^2}\right),
\qquad V_t := \mathrm{Var}_{p_1}(r).
$$

In particular, under the canonical choice $\kappa_t = 1$,

$$
\gamma_{t,\mathrm{can}} = \frac{\mathrm{IG}_t}{V_t}.
$$

#### Proof

By Theorem 3.3, $\gamma_t^*$ satisfies $\mathbb E_{\pi}[r] = \mathbb E_{q_{\gamma_t^*}}[r]$. By exponential-family calculus,

$$
\mathbb E_{q_\gamma}[r] = A'(\gamma) = A'(0) + \gamma A''(0) + O(\gamma^2) = \mathrm{IG}_t + \gamma V_t + O(\gamma^2).
$$

Under Assumption 4.3,

$$
\mathbb E_{\pi}[r] = \mathrm{IG}_t + \kappa_t \, \mathrm{IG}_t + o(\mathrm{IG}_t).
$$

Equating the two expansions and solving for $\gamma_t^*$ yields

$$
\gamma_t^* = \frac{\kappa_t \, \mathrm{IG}_t}{V_t} + O\!\left(\frac{\mathrm{IG}_t^2}{V_t^2}\right).
$$

QED.

Combining this with the denoising-time reliability weight from Proposition 4.2 gives the **canonical EAMD v2 scale**

$$
\gamma_t^{\mathrm{EAMD}} = w_t \frac{\mathrm{IG}_t}{V_t},
\qquad w_t = 1-\mu_t.
$$

Using the local trust-radius approximation from Section 4.3, this corresponds to the specific trust budget

$$
\varepsilon_t^{\mathrm{EAMD}}
= \frac{V_t}{2}\bigl(\gamma_t^{\mathrm{EAMD}}\bigr)^2
= \frac{w_t^2 \, \mathrm{IG}_t^2}{2V_t}
+ O\!\left(\frac{\mathrm{IG}_t^3}{V_t^2}\right).
$$

So the trust-budget map is no longer unspecified:

$$
g_t(\mathrm{IG}_t, V_t, \mu_t) := \frac{(1-\mu_t)^2 \, \mathrm{IG}_t^2}{2V_t}.
$$

This shows exactly when guidance should be strong or weak:
- **strong guidance** when refined evidence has large information gain $\mathrm{IG}_t$, the score variance $V_t$ is small, and the denoising state has low noise ($\mu_t$ small),
- **weak guidance** when the new evidence is redundant ($\mathrm{IG}_t \approx 0$), the score is unstable ($V_t$ large), or the denoising state is still highly masked ($\mu_t$ large).

### 4.5 Why symmetric KL is still useful

The exact information gain is the forward KL. The symmetric KL is useful because it is the branch-discrimination gap

$$
J(p_1,p_0) = \mathbb E_{p_1}[r] - \mathbb E_{p_0}[r].
$$

So the correct theorem-safe reading is:
- $D_KL(p_1||p_0)$ is the **information gain**,
- $J(p_1,p_0)$ is the **observable branch-separation signal**.

If one wants a single scalar signal for diagnostics or for a practical approximation, Jeffreys divergence is principled. But the core optimization and the exact trust-region rule do **not** require symmetrization.

### 4.6. v1 schedule as a special case of v2

The v1 rule used

$$
\hat\gamma_t^{\mathrm{v1}}
= \lambda_{\max}\tanh\!\left(\beta \frac{J_t}{H_t+\varepsilon}\right) w_t,
$$

where $J_t$ is the symmetric KL, $H_t = H(p_1)$, and $w_t=t/T$. Under the small-trust-budget approximation from Section 4.3, any proxy guidance scale $\hat\gamma_t$ corresponds to the trust budget

$$
\hat\varepsilon_t = \frac{V_t}{2} \hat\gamma_t^2.
$$

Therefore v1 is recovered by choosing the proxy budget

$$
\varepsilon_t^{\mathrm{v1}}
:= \frac{V_t}{2}\left[\lambda_{\max}\tanh\!\left(\beta \frac{J_t}{H_t+\varepsilon}\right) w_t\right]^2.
$$

In that sense v1 is a **special-case instantiation** of the v2 framework in which
- $J_t$ is used as a proxy for the exact information gain $\mathrm{IG}_t$,
- entropy $H_t$ is used as a proxy for the exact score-variance term $V_t$,
- and the linear schedule $w_t=t/T$ approximates the principled noise-derived reliability factor $w_t=1-\mu_t$ when the diffusion schedule is linear.

This validates the existing v1 experiments while making their heuristic status explicit.

## 5. Diffusion-Native Multi-Query Retrieval


The next question is why a dLLM can propose multiple bridge hypotheses in one forward pass.

### Proposition 5.1. One dLLM forward pass exposes the full bridge-token posterior

Fix a partially masked canvas $x_t$ and a bridge position $b$. A single masked-diffusion forward pass returns the full categorical distribution

$$
p_theta(X_0^b = x | x_t, Q, C_0, t),
\qquad x \in V,
$$

for **all** vocabulary items $x$ simultaneously.

#### Proof

By definition, the denoiser produces a logit vector over $V$ at every masked position. Applying softmax yields the full categorical distribution at that position. No additional model call is needed to obtain the top-$k$ bridge seeds. QED.

This proposition should be stated carefully. What is available in one pass is:
- the top-$k$ **seed hypotheses** for the bridge token,
- not necessarily the final multi-token bridge phrase, whose completion may require extra denoising steps.

That distinction is important and should be explicit in the paper.

### Proposition 5.2. Entropy lower-bounds hypothesis richness

Let $p_B$ be the posterior over bridge-token hypotheses at a chosen bridge position, and let

$$
H_B := - \sum_{x \in V} p_B(x) \log p_B(x).
$$

If $\mathrm{supp}(p_B)$ denotes the support of $p_B$, then

$$
|\mathrm{supp}(p_B)| >= e^{H_B}.
$$

#### Proof

For any discrete distribution, Shannon entropy is bounded by the logarithm of support size:

$$
H_B <= \log |\mathrm{supp}(p_B)|.
$$

Exponentiating both sides gives the result. QED.

Therefore large bridge-position entropy implies a large effective set of plausible bridge hypotheses. This is exactly the diversity resource that EAMD can exploit for candidate-driven retrieval.

This differs from SPREAD Sec. 4:
- SPREAD uses query-token semantic relevance to preserve alignment during denoising,
- EAMD uses posterior diversity at bridge positions to expand retrieval trajectories.

The two are orthogonal.

### Theorem 5.3. Candidate-driven retrieval improves a coverage lower bound

Assume there exists a latent bridge variable $B^*$ such that retrieving with query $Q \oplus b$ yields supporting evidence with probability $\rho(b)$ when $b = B^*$.

Let $K_k$ be the top-$k$ candidate set under $p_B$, and define the top-$k$ mass

$$
M_k := \sum_{b \in K_k} p_B(b).
$$

If we issue retrieval for every $b \in K_k$ and union the returned passages, the resulting support-recall satisfies

$$
\mathrm{Recall}_k
= \sum_{b \in K_k} p_B(b) \rho(b)
>= \rho_{\min}(K_k) \, M_k,
$$

where $\rho_{\min}(K_k) := \min_{b \in K_k} \rho(b)$.

#### Proof

Condition on the latent correct bridge $B^*$. Under the stated coverage model, support is retrieved exactly when the correct bridge lies in the queried set and the retriever succeeds for that bridge. Hence

$$
\mathrm{Recall}_k
= \sum_{b \in K_k} P(B^*=b) P(\text{support retrieved} | B^*=b)
= \sum_{b \in K_k} p_B(b) \rho(b).
$$

Since every $\rho(b) >= \rho_{\min}(K_k)$,

$$
\mathrm{Recall}_k
>= \rho_{\min}(K_k) \sum_{b \in K_k} p_B(b)
= \rho_{\min}(K_k) M_k.
$$

QED.

This theorem formalizes the key EAMD intuition:
- candidate expansion helps when the posterior mass over plausible bridges is spread across multiple hypotheses,
- and the retrieval union converts that extra posterior mass into extra support coverage.

### Remark 5.3A. Scope of the latent-bridge assumption

The latent bridge model in Theorem 5.3 is strongest for entity-chain multi-hop questions, where one bridge entity or relation template largely determines whether the second-hop retrieval succeeds. This is a good approximation for many HotpotQA and MuSiQue bridge questions. It is weaker for comparison, aggregation, or set-valued reasoning, where multiple bridge variables may be required. The theorem should therefore be read as a coverage guarantee for the bridge-entity regime, not as a universal statement about all multi-hop QA.

### Corollary 5.4. Monotonicity in the number of candidates

Under the same assumptions,

$$
\mathrm{Recall}_{k+1} >= \mathrm{Recall}_k.
$$

#### Proof

$K_k \subseteq K_{k+1}$ and each term in the sum is nonnegative, so adding a new candidate cannot decrease the sum. QED.

### Corollary 5.5. Independence approximation

If support events from different candidate queries are conditionally independent, with per-query recall $r_b$, then

$$
\mathrm{Recall}_k = 1 - \prod_{b \in K_k} (1-r_b),
$$

which is always at least $\max_{b \in K_k} r_b$.

#### Proof

Under conditional independence,

$$
P\Big(\bigcup_{b \in K_k} E_b\Big)
= 1 - P\Big(\bigcap_{b \in K_k} E_b^c\Big)
= 1 - \prod_{b \in K_k} (1-r_b).
$$

Since $\prod_b (1-r_b) <= 1-r_{b_0}$ for every $b_0$, the result follows. QED.

## 6. Iterative Evidence Refinement and Convergence

We now move from one refinement step to multiple rounds

$$
C_0 \to C_1 \to C_2 \to \cdots,
\qquad
C_{r+1} = C_r \cup \Delta C_{r+1}.
$$

Let

$$
p_r^*(x) := P(X_0 = x | x_t, Q, C_r)
$$

be the exact Bayesian answer posterior at round $r$.

### Theorem 6.1. Exact posterior uncertainty decreases monotonically

For every refinement round,

$$
H(X_0 | x_t, Q, C_{r+1})
= H(X_0 | x_t, Q, C_r)
- I(X_0; \Delta C_{r+1} | x_t, Q, C_r).
$$

In particular,

$$
H(X_0 | x_t, Q, C_{r+1}) <= H(X_0 | x_t, Q, C_r).
$$

#### Proof

Since $C_{r+1}$ is the pair $(C_r, \Delta C_{r+1})$ up to a deterministic union operation, the chain rule for conditional entropy gives

$$
H(X_0 | x_t,Q,C_r,\Delta C_{r+1})
= H(X_0 | x_t,Q,C_r)
- I(X_0;\Delta C_{r+1}|x_t,Q,C_r).
$$

Identifying $H(X_0 | x_t,Q,C_r,\Delta C_{r+1})$ with $H(X_0 | x_t,Q,C_{r+1})$ proves the identity. Since conditional mutual information is nonnegative, the inequality follows. QED.

### Corollary 6.2. Exact Bayesian answer log-likelihood is monotone

If $X_0 ~ p_r^*$, then the expected negative log-likelihood under the exact posterior satisfies

$$
\mathbb E_{p_{r+1}^*}[-\log p_{r+1}^*(X_0)]
<= \mathbb E_{p_r^*}[-\log p_r^*(X_0)].
$$

Moreover, the improvement equals the conditional mutual information from the added evidence.

#### Proof

For any posterior $p$,

$$
\mathbb E_p[-\log p(X)] = H(p).
$$

Applying Theorem 6.1 gives the claim. QED.

So exact Bayesian refinement is monotone in a strict information-theoretic sense.

### Theorem 6.3. Approximate-model improvement bound

Let $p_{theta,r}$ be the model posterior used at round $r$, and define its approximation error to the exact posterior as

$$
\kappa_r := D_{\mathrm{KL}}(p_r^* || p_{theta,r}).
$$

Then the cross-entropy difference across one refinement round is

$$
\mathrm{CE}(p_{r+1}^*, p_{theta,r+1})
- \mathrm{CE}(p_r^*, p_{theta,r})
= - I(X_0; \Delta C_{r+1} | x_t,Q,C_r) + \kappa_{r+1} - \kappa_r,
$$

where $\mathrm{CE}(p,q) := \mathbb E_p[-\log q(X)]$.

Hence a sufficient condition for monotone improvement of the approximate model is

$$
\kappa_{r+1} - \kappa_r < I(X_0; \Delta C_{r+1} | x_t,Q,C_r).
$$

#### Proof

By the identity $\mathrm{CE}(p,q)=H(p)+D_{KL}(p||q)$,

$$
\mathrm{CE}(p_r^*, p_{theta,r}) = H(p_r^*) + \kappa_r,
$$

and similarly for round $r+1$.

Subtracting yields

$$
\mathrm{CE}(p_{r+1}^*, p_{theta,r+1})
- \mathrm{CE}(p_r^*, p_{theta,r})
= \bigl(H(p_{r+1}^*) - H(p_r^*)\bigr) + \kappa_{r+1} - \kappa_r.
$$

Apply Theorem 6.1 to the entropy difference. QED.

This theorem gives the correct paper-safe statement:
- exact posterior quality is monotone,
- approximate model quality is monotone when the information gain from new evidence exceeds the increase in model approximation error.

### Theorem 6.4. Summability and vanishing of refinement gains

Define the per-round information gain

$$
I_r := I(X_0 ; \Delta C_{r+1} | x_t, Q, C_r).
$$

Then for every integer $R >= 1$,

$$
\sum_{r=0}^{R-1} I_r
= H(X_0 | x_t,Q,C_0) - H(X_0 | x_t,Q,C_R)
<= H(X_0 | x_t,Q,C_0).
$$

Consequently, the series $\sum_{r=0}^\infty I_r$ converges and, in particular,

$$
I_r \to 0 \qquad \text{as } r \to \infty.
$$

#### Proof

Apply Theorem 6.1 at rounds $r=0,1,\dots,R-1$ and sum:

$$
H(X_0|x_t,Q,C_{r+1}) = H(X_0|x_t,Q,C_r) - I_r.
$$

The sum telescopes to

$$
\sum_{r=0}^{R-1} I_r
= H(X_0|x_t,Q,C_0) - H(X_0|x_t,Q,C_R).
$$

Since conditional entropy is nonnegative, the right-hand side is at most $H(X_0|x_t,Q,C_0)$. Thus the partial sums are monotone increasing and uniformly bounded, so the infinite series converges. Any convergent series of nonnegative terms has terms tending to zero, hence $I_r -> 0$. QED.

### Corollary 6.5. Finite-corpus stabilization

Assume retrieval is deduplicated and all evidence passages are drawn from a finite corpus $\mathcal U$. If every nonempty refinement increment $\Delta C_{r+1}$ adds at least one previously unseen passage, then after at most $|\mathcal U|$ nontrivial rounds the evidence sets stabilize and all later information gains are zero.

#### Proof

Because the evidence sets are nested and each nonempty increment adds at least one unseen passage from a finite corpus, there can be at most $|\mathcal U|$ strict enlargements before no new passage remains. Once $C_{r+1}=C_r$, we have $\Delta C_{r+1}=\emptyset$, so Theorem 6.1 gives $I_r=0$. QED.

### Remark 6.6. What is and is not proved about convergence

Theorem 6.4 proves that the **information gain sequence** of iterative refinement is summable and vanishes. It does **not** prove that the full refinement operator is a contraction in any global metric. A Banach-style geometric convergence statement would require additional Lipschitz assumptions on the retriever, the denoiser, and the answer-to-query expansion map; those assumptions are not proved here.

### Conjecture 6.7. Monotonic EM/F1 improvement

Under calibrated retrieval, bounded approximation drift, and a stable answer extractor from the local posterior to the decoded short answer, EM/F1 should improve monotonically with rounds until the conditional mutual information of added evidence becomes negligible.

This is **not proved** here. Theorem 6.3 proves only a surrogate cross-entropy condition, and Theorem 6.4 proves only that the information gain sequence vanishes.

## 7. A Principled Remask Rule from ReMDM


ReMDM shows that one can introduce remasking through a generalized posterior while preserving the same marginals as classical masked diffusion, provided the remask probabilities stay within the valid $sigma_t$ bounds. We now derive a local EAMD remask criterion on top of that family.

Fix a committed answer position $j$. Let

$$
p_{\mathrm{old},j}(x) := p_theta(X_0^j=x | \hat a^{(0)}, Q, C_0),
\qquad
p_{\mathrm{new},j}(x) := p_theta(X_0^j=x | \hat a^{(0)}, Q, C_1).
$$

### Theorem 7.1. The exact local value of refreshing a token is a forward KL

Consider the local variational free energy under the updated evidence posterior:

$$
\mathcal F_j(q; p_{\mathrm{new},j}) := D_{\mathrm{KL}}(q || p_{\mathrm{new},j}).
$$

If the current committed variational factor is $q = p_{\mathrm{old},j}$ and refreshing the token allows us to reset the factor to the optimum $q = p_{\mathrm{new},j}$, then the exact free-energy gain from remasking position $j$ is

$$
\Delta_j^{\mathrm{refresh}}
= \mathcal F_j(p_{\mathrm{old},j}; p_{\mathrm{new},j})
- \mathcal F_j(p_{\mathrm{new},j}; p_{\mathrm{new},j})
= D_{\mathrm{KL}}(p_{\mathrm{old},j} || p_{\mathrm{new},j}).
$$

#### Proof

By definition,

$$
\mathcal F_j(p_{\mathrm{new},j}; p_{\mathrm{new},j})
= D_{\mathrm{KL}}(p_{\mathrm{new},j} || p_{\mathrm{new},j}) = 0.
$$

Therefore the gain from replacing the stale factor $p_old$ with the optimal factor $p_new$ is exactly

$$
D_{\mathrm{KL}}(p_{\mathrm{old},j} || p_{\mathrm{new},j}).
$$

QED.

This is the exact theorem-safe remask criterion. It is **not** the symmetric KL.

### Proposition 7.2. Symmetric divergence is a conservative surrogate

Define

$$
D_j^{\mathrm{sym}}
:= D_{\mathrm{KL}}(p_{\mathrm{old},j} || p_{\mathrm{new},j})
+ D_{\mathrm{KL}}(p_{\mathrm{new},j} || p_{\mathrm{old},j}).
$$

Then

$$
D_j^{\mathrm{sym}} >= \Delta_j^{\mathrm{refresh}}.
$$

#### Proof

The second KL term is nonnegative, so

$$
D_j^{\mathrm{sym}}
= D_{\mathrm{KL}}(p_{\mathrm{old},j} || p_{\mathrm{new},j})
+ D_{\mathrm{KL}}(p_{\mathrm{new},j} || p_{\mathrm{old},j})
>= D_{\mathrm{KL}}(p_{\mathrm{old},j} || p_{\mathrm{new},j}).
$$

QED.

So a symmetric criterion is defensible only as a conservative robustification, not as the exact refresh value.

### Theorem 7.3. Optimal remask probability with a ReMDM prior

Let $\rho_j \in [0,1]$ be the probability of remasking token $j$. Let $\pi_j \in (0,1)$ be a base remask prior inherited from the global ReMDM schedule, and let $c_j >= 0$ be the computational/stability cost of remasking token $j$.

Consider the local decision objective

$$
\max_{0 <= \rho <= 1}
\Big[
\rho \, \Delta_j^{\mathrm{refresh}} - c_j \rho
- \tau \, D_{\mathrm{KL}}\bigl(\mathrm{Bern}(\rho) || \mathrm{Bern}(\pi_j)\bigr)
\Big],
$$

with temperature $\tau > 0$.

Then the unique optimizer is

$$
\rho_j^*
= \sigma\left(
\operatorname{logit}(\pi_j)
+ \frac{\Delta_j^{\mathrm{refresh}} - c_j}{\tau}
\right),
$$

where $\sigma(u)=1/(1+e^{-u})$.

#### Proof

Write the objective as

$$
G(\rho)
= \rho(\Delta_j^{\mathrm{refresh}} - c_j)
- \tau\Big[
\rho \log\frac{\rho}{\pi_j}
+ (1-\rho)\log\frac{1-\rho}{1-\pi_j}
\Big].
$$

Differentiate:

$$
G'(\rho)
= \Delta_j^{\mathrm{refresh}} - c_j
- \tau\left[
\log\frac{\rho}{\pi_j} - \log\frac{1-\rho}{1-\pi_j}
\right].
$$

Setting $G'(\rho)=0$ gives

$$
\log\frac{\rho}{1-\rho}
= \log\frac{\pi_j}{1-\pi_j}
+ \frac{\Delta_j^{\mathrm{refresh}} - c_j}{\tau}.
$$

Applying the logistic map yields the stated solution. Strict concavity follows because the Bernoulli KL term is strictly convex in $\rho$, so the maximizer is unique. QED.

### Corollary 7.4. Valid ReMDM remask probability

If the ReMDM generalized posterior requires $0 <= \sigma_j <= \sigma_{t,\max}$, define

$$
\sigma_j^* := \min\{\sigma_{t,\max}, \rho_j^*\}.
$$

Then the remask rule is valid within the ReMDM posterior family.

#### Proof

By construction $\rho_j^* \in [0,1]$. Clipping enforces the ReMDM admissibility constraint. Theorem 3.1 of ReMDM guarantees that any admissible remask schedule defines a valid generalized posterior with the same one-step marginals as the original absorbing-mask process. QED.

This gives the correct theorem-safe remask story:
- the exact value of revising token $j$ is the forward KL $D_KL(p_old || p_new)$,
- the optimal remask probability is logistic in that value after combining it with the ReMDM prior and a remask cost,
- symmetric divergence is optional and conservative, not fundamental.

## 8. Final EAMD v2 Sampler Summary

At each masked answer position $(t,i)$:

1. Compute the stale- and refined-evidence posteriors

$$
p_0 = p_{theta,C_0}^{(t,i)},
\qquad
p_1 = p_{theta,C_1}^{(t,i)}.
$$

2. Form the evidence score ratio

$$
r(x) = \log \frac{p_1(x)}{p_0(x)}.
$$

3. Measure the exact evidence information gain

$$
\mathrm{IG}_{t,i} = D_{\mathrm{KL}}(p_1 || p_0).
$$

4. Compute the denoising-time reliability weight from the diffusion noise schedule

$$
w_t = 1-\mu_t.
$$

5. Compute the canonical local guidance scale

$$
\gamma_{t,i}^{\mathrm{EAMD}} = w_t \frac{\mathrm{IG}_{t,i}}{V_{t,i}},
\qquad V_{t,i} = \mathrm{Var}_{p_1}(r).
$$

Equivalently, the canonical trust budget is

$$
\varepsilon_{t,i}^{\mathrm{EAMD}}
= \frac{w_t^2 \, \mathrm{IG}_{t,i}^2}{2V_{t,i}}
+ O\!\left(\frac{\mathrm{IG}_{t,i}^3}{V_{t,i}^2}\right).
$$

6. Use the guided logits

$$
\tilde \ell(x) = \ell_1(x) + \gamma_{t,i}^{\mathrm{EAMD}}(\ell_1(x)-\ell_0(x)).
$$

Equivalently, one may solve the exact trust-region equation

$$
D_{\mathrm{KL}}(q_{\gamma_{t,i}} || p_1) = \varepsilon_{t,i}^{\mathrm{EAMD}},
\qquad
q_\gamma(x) \propto p_1(x) e^{\gamma r(x)}.
$$

7. If one wants the original v1 heuristic, choose instead the proxy budget

$$
\varepsilon_t^{\mathrm{v1}}
= \frac{V_t}{2}\left[\lambda_{\max}\tanh\!\left(\beta \frac{J_t}{H_t+\varepsilon}\right) w_t\right]^2,
$$

which recovers $\gamma_t^{\mathrm{v1}} = \lambda_{\max}\tanh(\beta J_t/(H_t+\varepsilon)) w_t$ under the local approximation.

At a committed answer token $j$, the exact refresh value is

$$
\Delta_j^{\mathrm{refresh}} = D_{\mathrm{KL}}(p_{\mathrm{old},j} || p_{\mathrm{new},j}),
$$

and the optimal remask probability is

$$
\rho_j^*
= \sigma\left(
\operatorname{logit}(\pi_j)
+ \frac{\Delta_j^{\mathrm{refresh}} - c_j}{\tau}
\right),
$$

followed by clipping to the valid ReMDM range.

## 9. What This Formulation Proves, and What It Does Not Prove

### Proved

1. EAMD guidance is the optimizer of a trust-region variational objective.
2. Under the evidence-geodesic oracle assumption, that objective is exactly the primal form of oracle answer-risk minimization.
3. The correct information gain for $C_0 \to C_1$ is $D_KL(p_1||p_0)$.
4. Symmetric KL is justified as a branch-separation statistic, not as an arbitrary heuristic.
5. Exact Bayesian iterative refinement is monotone in answer log-likelihood, and the per-round information gains are summable and vanish.
6. A principled remask probability follows from a local variational objective under ReMDM.

### Not proved

1. That the self-similar refinement approximation $\kappa_t = 1$ is exact for every model and dataset.
2. That the oracle-optimal posterior always lies beyond $p_1$ on the evidence geodesic.
3. That the full refinement operator is a global contraction.
4. That EM/F1 are monotone for the approximate model.
5. That candidate expansion always improves retrieval in practice; the bound requires the latent-bridge coverage assumptions in Theorem 5.3.

### Recommended paper-safe claim

> EAMD is a variational posterior-transport method for iterative evidence refinement in masked diffusion language models. It replaces stale evidence $C_0$ by refined evidence $C_1$, measures the exact information gain $D_KL(p_1||p_0)$, calibrates a denoising-time-aware trust budget $\varepsilon_t \asymp (1-\mu_t)^2 D_KL(p_1||p_0)^2 / (2\,\mathrm{Var}_{p_1}(r))$, transports the posterior along the evidence geodesic under that KL trust region, and uses a ReMDM-consistent remask policy whose optimal probability is logistic in the local refresh value $D_KL(p_old||p_new)$.

## References


- ARAM: *Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models*, arXiv:2603.17677.
- SPREAD: *Unlocking the Potentials of Retrieval-Augmented Generation for Diffusion Language Models: A Semantic Drift Perspective*, arXiv:2601.11342.
- ReMDM: *Remasking Discrete Diffusion Models with Inference-Time Scaling*, arXiv:2503.00307.
- MDLM: *Simple and Effective Masked Diffusion Language Models*, arXiv:2406.07524.
- Dream-7B: *Dream 7B: Diffusion Large Language Models*.
