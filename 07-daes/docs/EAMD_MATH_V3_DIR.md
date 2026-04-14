# EAMD v3: Denoising-Interleaved Retrieval (DIR)

This note extends `EAMD_MATH_V2.md` from one-shot evidence refinement
`C_0 -> C_1` to a finite-horizon denoising-interleaved retrieval process.

The central change is conceptual:
- `EAMD v2` studies how the posterior changes when stale evidence `C_0` is
  replaced by refined evidence `C_1`.
- `EAMD v3 / DIR` studies how evidence is updated *during one diffusion
  trajectory* at a finite set of retrieval checkpoints.

Throughout, every statement is either:
- proved,
- proved under explicit assumptions, or
- marked as a conjecture.

## 0. Status of Claims

### Theorem-backed in this document

1. The checkpoint evidence-gain identity is an exact conditional-entropy
   identity and remains valid for partially committed denoising states.
2. The finite-horizon answer-score increments telescope exactly.
3. Under an explicit query-channel bound, retrieval is not utility-improving
   below a reliability threshold `w_min`.
4. Under the latent-bridge coverage model of Theorem 5.3 in
   `EAMD_MATH_V2.md`, the DIR query-construction policy admits a recall lower
   bound, a top-`k` advantage bound over top-1 retrieval, and dominance over
   question-only retrieval under explicit sufficient conditions.
5. The exact refresh value of a committed token remains a forward KL after a
   mid-trajectory evidence update, and the time-dependent remask rule remains
   valid in the ReMDM generalized-posterior family.
6. Under a separable local cross-entropy surrogate, remasking the top-`m`
   divergent committed positions is optimal.
7. DIR terminates under a finite checkpoint budget, and also under an adaptive
   policy with deduplicated retrieval from a finite corpus.
8. The model-call counts of vanilla dLLM, DIR, ARAM, and IRCoT follow exactly
   from their inference procedures.
9. The asynchronous DIR variant is theorem-equivalent to synchronous DIR when
   all quantities are evaluated at the evidence-injection state.

### Assumption-backed in this document

1. The usefulness threshold for checkpoint retrieval uses an explicit
   query-channel efficiency bound.
2. The top-`m` remask optimality result uses a mean-field separability
   assumption, stated explicitly.
3. The coordinate-ascent interpretation (informal analogy) of DIR (Remark 3.3)
   is an analogy, not a formal variational bound.
4. The entropy-triggered parallel retrieval result uses an explicit
   entropy-to-mass calibration assumption.
5. The contrastive evidence bonus is justified only under an explicit
   lexical-support approximation to the true evidence score ratio.

### Implementation-level additions (not theorem-backed)

1. Ghost denoising step (Remark 6.7): evaluate new evidence without changing
   the canvas, preventing flickering from noisy retrieval. This is the +R cost
   in Proposition 8.1.
2. Bridge-span sensitivity (Remark 6.8): entity-aware masking that commits
   function words before bridge entities; adaptive retrieval trigger when bridge
   confidence exceeds threshold.
3. Recovery Rate (Definition 8.8): fraction of wrong tokens corrected after
   remask-and-re-denoise; primary ablation diagnostic.
4. Contrastive evidence guidance (Proposition 8.10): first-order surrogate of
   the true evidence score ratio under an explicit lexical-support
   approximation.

### Conjectural in this document

1. Monotonic improvement of downstream EM/F1 under DIR.
2. Uniform superiority of DIR over AR methods in wall-clock latency; this is an
   empirical claim, not a theorem.

## 1. Notation

We inherit the notation of `EAMD_MATH_V2.md`.

Let:
- `Q` be the question,
- `A = X_0 \in V^n` be the clean short-answer sequence,
- `m` be the absorbing mask token,
- `x_t \in (V \cup {m})^n` be the denoising canvas at reverse step `t`,
- `M_t := { j : x_t^j != m }` be the committed positions,
- `t_0 > t_1 > ... > t_R >= 0` be the DIR retrieval checkpoints,
- `C_r` be the evidence set available immediately before checkpoint `r`,
- `\Delta C_{r+1}` be the passages added at checkpoint `r`,
- `C_{r+1} := C_r \cup \Delta C_{r+1}`.

The checkpoint state is

$$
s_r := (x_{t_r}, M_{t_r}, Q, C_r).
$$

Because `M_{t_r}` is a deterministic function of `x_{t_r}`, we sometimes omit
it from the conditioning set.

For a fixed checkpoint `r` and answer position `j`, define the local posterior
under evidence `C` by

$$
p_{C,r}^{(j)}(x)
:= p_theta(A^j = x \mid x_{t_r}, Q, C, t_r),
\qquad x \in V.
$$

Define the checkpoint answer posterior by

$$
p_r(a) := p_theta(A = a \mid s_r).
$$

The diffusion reliability weight from Proposition 4.2 of
`EAMD_MATH_V2.md` is

$$
w_t := 1 - \mu_t.
$$

## 2. Checkpoint Evidence-Gain Identity

The exact one-round entropy identity from Theorem 6.1 of
`EAMD_MATH_V2.md` extends directly to DIR checkpoints.

### Theorem 2.1. Checkpoint evidence-gain identity

At checkpoint `r`,

$$
H(A \mid x_{t_r}, Q, C_{r+1})
=
H(A \mid x_{t_r}, Q, C_r)
-
I(A ; \Delta C_{r+1} \mid x_{t_r}, Q, C_r).
$$

Equivalently, because `M_{t_r} = f(x_{t_r})`,

$$
H(A \mid x_{t_r}, M_{t_r}, Q, C_{r+1})
=
H(A \mid x_{t_r}, M_{t_r}, Q, C_r)
-
I(A ; \Delta C_{r+1} \mid x_{t_r}, M_{t_r}, Q, C_r).
$$

#### Proof

Since `C_{r+1}` is the deterministic union of `(C_r, \Delta C_{r+1})`, the
chain rule for conditional entropy gives

$$
H(A \mid x_{t_r}, Q, C_r, \Delta C_{r+1})
=
H(A \mid x_{t_r}, Q, C_r)
-
I(A ; \Delta C_{r+1} \mid x_{t_r}, Q, C_r).
$$

Identifying

$$
H(A \mid x_{t_r}, Q, C_r, \Delta C_{r+1})
=
H(A \mid x_{t_r}, Q, C_{r+1})
$$

proves the first identity.

For the second identity, note that `M_{t_r}` is measurable with respect to
`x_{t_r}`. Adding a deterministic function of a conditioned variable does not
change the sigma-algebra of the conditioning set, so the same equality holds
with `M_{t_r}` included explicitly. QED.

### Corollary 2.2. Nonnegativity of checkpoint gains

For every checkpoint,

$$
G_r := I(A ; \Delta C_{r+1} \mid x_{t_r}, Q, C_r) >= 0,
$$

and therefore

$$
H(A \mid x_{t_r}, Q, C_{r+1}) <= H(A \mid x_{t_r}, Q, C_r).
$$

#### Proof

Conditional mutual information is nonnegative. Apply Theorem 2.1. QED.

### Remark 2.3. Partially committed states are not a problem

The theorem does not require `x_{t_r}` to be fully denoised. It holds for any
partially committed canvas because `x_{t_r}` is simply part of the conditioning
context. No independence assumption on committed versus masked positions is
needed.

## 3. Finite-Horizon Evidence-Ascent Objective

DIR is a finite-horizon control problem over checkpoint states.

### 3.1 Admissible actions

At checkpoint `r`, the controller chooses an action

$$
a_r := (h_r, S_r, \rho_r),
$$

where:
- `h_r` is a query hypothesis drawn from a policy
  `\pi_h(. | s_r)`,
- `S_r \subseteq R(Q \oplus h_r)` is the selected subset of retrieved passages,
- `\rho_r = (\rho_{j,r})_{j \in M_{t_r}} \in [0,1]^{|M_{t_r}|}` is the remask
  probability vector applied after the evidence update.

The evidence update is

$$
\Delta C_{r+1} := S_r \setminus C_r,
\qquad
C_{r+1} := C_r \cup \Delta C_{r+1}.
$$

After remasking according to `\rho_r`, denoising resumes until the next
checkpoint.

### 3.2 Per-round utility

Define the checkpoint evidence gain

$$
G_r := I(A ; \Delta C_{r+1} \mid s_r).
$$

Let `cost_ret(\Delta C_{r+1}) >= 0` be the retrieval / context-update cost and
`cost_rev(\rho_r) >= 0` be the remask / revision cost. The DIR per-round utility
is

$$
U_r := G_r - \lambda_{ret} \, cost_{ret}(\Delta C_{r+1})
          - \lambda_{rev} \, cost_{rev}(\rho_r).
$$

For a policy `pi = (\pi_h, \pi_S, \pi_rho)` and finite horizon `R`, the control
objective is

$$
J(pi) := \mathbb E_pi \left[ \sum_{r=0}^{R-1} U_r \right].
$$

This is the finite-horizon evidence-ascent objective for DIR.

### Theorem 3.1. Telescoping answer-score identity

For any completed answer `a \in V^n`,

$$
\sum_{r=0}^{R-1} \bigl[ \log p_{r+1}(a) - \log p_r(a) \bigr]
= \log p_R(a) - \log p_0(a).
$$

#### Proof

The sum is telescoping:

$$
\begin{aligned}
\sum_{r=0}^{R-1} \bigl[ \log p_{r+1}(a) - \log p_r(a) \bigr]
&= (\log p_1(a)-\log p_0(a))
 + (\log p_2(a)-\log p_1(a))
 + \cdots \\
&\quad + (\log p_R(a)-\log p_{R-1}(a)) \\
&= \log p_R(a)-\log p_0(a).
\end{aligned}
$$

QED.

### Corollary 3.2. Cumulative evidence score

Define the checkpoint score increment

$$
g_r(a) := \log p_{r+1}(a) - \log p_r(a).
$$

Then the cumulative answer-level DIR score is

$$
g_{0:R-1}(a) := \sum_{r=0}^{R-1} g_r(a)
= \log p_R(a) - \log p_0(a).
$$

This is the answer-level analogue of the local evidence score ratio from
Section 2 of `EAMD_MATH_V2.md`.

#### Proof

Immediate from Theorem 3.1. QED.

### Remark 3.3. Evolving trajectory and ELBO surgery interpretation

Several subtleties of the telescoping identity deserve explicit comment.

1. **Fixed-answer telescoping.**
   Theorem 3.1 holds for a *fixed* answer \(a\). In the actual DIR process the
   canvas evolves between checkpoints: the answer decoded at checkpoint \(r\) may
   differ from the one decoded at checkpoint \(r+1\). The identity is still exact
   for any single \(a\), but the operationally relevant answer changes across
   rounds.

2. **Post-hoc evaluation.**
   For the final decoded answer \(a_R\), the cumulative score
   \(g_{0:R-1}(a_R) = \log p_R(a_R) - \log p_0(a_R)\) measures the total
   evidence support accrued for that particular answer across the entire
   trajectory. This is the quantity reported in evaluation.

3. **Coordinate-ascent / ELBO surgery interpretation.**
   DIR alternates between two operations: (i) updating \(a\) by denoising the
   canvas \(x_t\) under fixed evidence \(C_r\), and (ii) updating \(C\) by
   retrieval under a fixed (or partially committed) canvas. This is loosely analogous to
   coordinate ascent in variational inference. However, this is an informal
   analogy, not a formal ELBO decomposition: we do not claim that retrieval
   optimises a specific rate term. The rigorous claim is only that each
   evidence update reduces conditional entropy (Theorem 2.1) and each
   denoising phase reduces canvas uncertainty under fixed evidence.

4. **Non-stationarity caveat.**
   Because the per-round increments \(g_r(a)\) are evaluated at *different*
   answer states \(p_r\), they are not increments of a single stationary
   objective. The telescoping sum is exact, but interpreting it as monotonic
   improvement of a single scalar requires fixing \(a\) to a single reference
   answer (e.g., \(a_R\)).

## 4. Optimal Checkpoint Placement

The reliability factor `w_t = 1 - \mu_t` from Proposition 4.2 of
`EAMD_MATH_V2.md` determines whether the current denoising state contains enough
stable signal to justify a retrieval update.

### Lemma 4.1. Conditional absorbing-mask information identity

Fix a checkpoint step `t` and answer position `j`. Let `Y_t^j` denote the
absorbing-mask observation of `A^j` at step `t`. Then, conditional on `(Q,C_r)`,

$$
I(A^j ; Y_t^j \mid Q, C_r)
=
(1-\mu_t) \, H(A^j \mid Q, C_r)
=
w_t \, H(A^j \mid Q, C_r).
$$

#### Proof

Condition on `(Q,C_r)`. With probability `1-\mu_t`, the token survives and
`Y_t^j = A^j`, so the conditional entropy is `0`. With probability `\mu_t`, the
token is masked and carries no token identity, so the conditional entropy is
`H(A^j | Q,C_r)`. Therefore

$$
H(A^j \mid Y_t^j, Q, C_r)
= \mu_t \, H(A^j \mid Q, C_r).
$$

Hence

$$
\begin{aligned}
I(A^j ; Y_t^j \mid Q, C_r)
&= H(A^j \mid Q, C_r) - H(A^j \mid Y_t^j, Q, C_r) \\
&= (1-\mu_t) H(A^j \mid Q, C_r).
\end{aligned}
$$

QED.

### Assumption 4.2. Query-channel efficiency bound

At checkpoint `r`, let `J_r \subseteq M_{t_r}` be the set of token positions used
by the query-construction policy. Assume there exists `\kappa_r >= 0` such that
for every admissible query hypothesis `h_r` measurable with respect to `s_r`,

$$
I(A ; \Delta C_{r+1} \mid s_r)
<=
 \kappa_r \, I(A ; h_r \mid Q, C_r).
$$

This assumption states that the retrieval channel cannot create more answer
information than a constant-factor multiple of the information already encoded
in the query hypothesis.

### Proposition 4.3. Reliability-gated usefulness bound

Under Assumption 4.2,

$$
G_r <= \kappa_r \, w_{t_r} \, B_r,
\qquad
B_r := \sum_{j \in J_r} H(A^j \mid Q, C_r).
$$

Consequently, if every non-null retrieval update has cost at least
`c_{ret}^{min} > 0`, then any policy whose checkpoint utility is

$$
U_r = G_r - \lambda_{ret} \, cost_{ret}(\Delta C_{r+1}) - \lambda_{rev} \, cost_{rev}(\rho_r)
$$

cannot have positive retrieval utility whenever

$$
w_{t_r} <= w_{min,r}
:= \frac{\lambda_{ret} c_{ret}^{min}}{\kappa_r B_r}.
$$

In particular, retrieval is only potentially worthwhile when `w_{t_r} > w_{min,r}`.

#### Proof

By Assumption 4.2,

$$
G_r <= \kappa_r I(A ; h_r \mid Q, C_r).
$$

Because `h_r` is constructed from `(Q, C_r)` and the observed token variables at
positions `J_r`, data processing gives

$$
I(A ; h_r \mid Q, C_r) <= I(A ; Y_{t_r}^{J_r} \mid Q, C_r).
$$

Using the chain rule and conditioning reduction,

$$
\begin{aligned}
I(A ; Y_{t_r}^{J_r} \mid Q, C_r)
&= \sum_{j \in J_r} I(A ; Y_{t_r}^j \mid Y_{t_r}^{<j}, Q, C_r) \\
&<= \sum_{j \in J_r} I(A ; Y_{t_r}^j \mid Q, C_r).
\end{aligned}
$$

Since `Y_{t_r}^j` depends on `A` only through `A^j`, another data-processing step
yields

$$
I(A ; Y_{t_r}^j \mid Q, C_r)
<= I(A^j ; Y_{t_r}^j \mid Q, C_r).
$$

Apply Lemma 4.1 to each term:

$$
I(A ; h_r \mid Q, C_r)
<= w_{t_r} \sum_{j \in J_r} H(A^j \mid Q, C_r)
= w_{t_r} B_r.
$$

Hence `G_r <= \kappa_r w_{t_r} B_r`.

If `\Delta C_{r+1}` is nonempty, then `cost_ret(\Delta C_{r+1}) >= c_{ret}^{min}`
and `cost_rev(\rho_r) >= 0`, so

$$
U_r <= \kappa_r w_{t_r} B_r - \lambda_{ret} c_{ret}^{min}.
$$

Therefore `U_r <= 0` whenever

$$
w_{t_r} <= \frac{\lambda_{ret} c_{ret}^{min}}{\kappa_r B_r}.
$$

QED.

### Remark 4.4. Practical schedule

Proposition 4.3 does not prescribe exact checkpoint times. It proves a
necessary usefulness condition: retrieve only after the denoising state has
reliability above a threshold. This justifies schedules that avoid very early
checkpoints and support adaptive rules based on `w_t`, confidence, and entropy.

## 5. Query Construction from a Partial Denoising State

DIR needs a query policy that uses both committed tokens and posterior mass at
uncertain bridge positions.

### 5.1 Working-memory summarizer and query policy

Fix checkpoint `r`.

For a committed position `j \in M_{t_r}`, define the confidence score

$$
conf_{r,j} := 1 - \frac{H(p_{C_r,r}^{(j)})}{\log |V|}.
$$

Fix a confidence threshold `\tau_c \in (0,1)`. The high-confidence committed set
is

$$
K_r := \{ j \in M_{t_r} : conf_{r,j} >= \tau_c \}.
$$

Let `B_r` be a designated set of bridge-sensitive positions, possibly outside
`M_{t_r}`. For each `b \in B_r`, let

$$
Top_k(b,r) := \{ u_{b,1}, ..., u_{b,k} \}
$$

be the top-`k` tokens under `p_{C_r,r}^{(b)}`.

Define the DIR working-memory summarizer

$$
\phi_r := \phi(x_{t_r}, M_{t_r}, C_r)
:=
\bigl(s_r^{conf}, B_r, \{Top_k(b,r)\}_{b \in B_r}\bigr),
$$

where `s_r^{conf}` is the concatenation of committed tokens at positions `K_r`
in answer order.

A query hypothesis is then a tuple

$$
h = (s_r^{conf}, u_{b_1,i_1}, ..., u_{b_m,i_m}),
$$

where each `u_{b_l,i_l}` is a top-`k` bridge candidate.

Define the DIR query policy by

$$
\pi(h \mid s_r)
\propto
\mathbf 1\{h \text{ is compatible with } s_r^{conf}\}
\prod_{l=1}^m p_{C_r,r}^{(b_l)}(u_{b_l,i_l}).
$$

Operationally, retrieval uses the working-memory query

$$
Q \oplus \phi_r \oplus h
$$

rather than the question alone.

### Proposition 5.1. One-checkpoint posterior support

The DIR working-memory summarizer `\phi_r` and query policy `\pi(\cdot \mid s_r)` are
measurable with respect to the checkpoint state `s_r` and require no additional
model call beyond the checkpoint denoiser outputs.

#### Proof

The committed span `s_r^{conf}` is a deterministic function of `(x_{t_r}, C_r)`,
the bridge set `B_r` is chosen from the checkpoint state, and each top-`k`
candidate set `Top_k(b,r)` is a deterministic function of the checkpoint
posterior `p_{C_r,r}^{(b)}` already produced by the denoiser. Hence both
`\phi_r` and `\pi(\cdot \mid s_r)` are measurable with respect to `s_r` and require no
extra model forward pass. QED.

### Theorem 5.2. Coverage lower bound for DIR queries

Assume the latent-bridge coverage model of Theorem 5.3 in
`EAMD_MATH_V2.md`. Let `K_r^*` be the candidate set issued by the DIR query
policy at checkpoint `r`, with posterior mass

$$
M_r^{DIR} := \sum_{b \in K_r^*} p_B(b \mid s_r).
$$

If retrieving with query `Q \oplus h(b)` succeeds with probability `\rho_r(b)`
when the latent bridge is `b`, then the support-recall of the unioned DIR
retrieval satisfies

$$
Recall_r^{DIR}
>= \rho_{min,r}^{DIR} \, M_r^{DIR},
\qquad
\rho_{min,r}^{DIR} := \min_{b \in K_r^*} \rho_r(b).
$$

#### Proof

This is exactly Theorem 5.3 of `EAMD_MATH_V2.md`, with the bridge posterior now
conditioned on the checkpoint state `s_r` rather than on a one-shot state. QED.

### Proposition 5.3. Top-`k` advantage over top-1 retrieval

Let `K_{r,1}` be the top-1 bridge candidate set and `K_{r,k}` the top-`k`
bridge candidate set under the checkpoint posterior. Define the additional mass

$$
\Delta M_{r,k:1}
:=
\sum_{b \in K_{r,k} \setminus K_{r,1}} p_B(b \mid s_r)
$$

and the additional-candidate success floor

$$
\rho_{add,r}^{(k)}
:=
\min_{b \in K_{r,k} \setminus K_{r,1}} \rho_r(b).
$$

Then, under the same latent-bridge model as Theorem 5.2,

$$
Recall_{r,k}^{DIR} - Recall_{r,1}^{DIR}
>=
\rho_{add,r}^{(k)} \, \Delta M_{r,k:1}.
$$

#### Proof

Applying Theorem 5.2 to `K_{r,k}` and `K_{r,1}` gives

$$
Recall_{r,k}^{DIR}
=
\sum_{b \in K_{r,k}} p_B(b \mid s_r)\rho_r(b),
$$

$$
Recall_{r,1}^{DIR}
=
\sum_{b \in K_{r,1}} p_B(b \mid s_r)\rho_r(b).
$$

Subtracting,

$$
Recall_{r,k}^{DIR} - Recall_{r,1}^{DIR}
=
\sum_{b \in K_{r,k} \setminus K_{r,1}} p_B(b \mid s_r)\rho_r(b).
$$

Every term in the sum is at least
`\rho_{add,r}^{(k)} p_B(b \mid s_r)`, so

$$
Recall_{r,k}^{DIR} - Recall_{r,1}^{DIR}
>=
\rho_{add,r}^{(k)}
\sum_{b \in K_{r,k} \setminus K_{r,1}} p_B(b \mid s_r)
=
\rho_{add,r}^{(k)} \Delta M_{r,k:1}.
$$

QED.

### Assumption 5.4. Entropy-to-mass calibration

Fix a bridge position `b`. Assume there exists a nondecreasing function
`\delta_k : [0,\log |V|] -> [0,1]` such that whenever the bridge entropy

$$
H_{bridge,r}^{(b)} := H\!\left(p_{C_r,r}^{(b)}\right)
$$

exceeds a threshold `h`, the additional top-`k` mass above top-1 satisfies

$$
\Delta M_{r,k:1}^{(b)} >= \delta_k(h).
$$

This assumption is not automatic from entropy alone; it is a calibration
assumption linking entropy to retrievable posterior mass.

### Corollary 5.5. Entropy-triggered parallel retrieval

Under Assumption 5.4, if a bridge position `b` satisfies
`H_{bridge,r}^{(b)} >= h`, then

$$
Recall_{r,k}^{DIR} - Recall_{r,1}^{DIR}
>=
\rho_{add,r}^{(k)} \, \delta_k(h).
$$

Thus high bridge entropy justifies parallel top-`k` retrieval exactly when it
implies nontrivial additional candidate mass.

#### Proof

By Assumption 5.4, `\Delta M_{r,k:1}^{(b)} >= \delta_k(h)`. Apply
Proposition 5.3. QED.

### Corollary 5.6. Sufficient condition for improvement over question-only retrieval

Let question-only retrieval have bridge candidate set `K_r^Q` with lower bound

$$
Recall_r^Q >= \rho_{min,r}^Q \, M_r^Q,
\qquad
M_r^Q := \sum_{b \in K_r^Q} p_B(b \mid Q, C_r).
$$

If

$$
\rho_{min,r}^{DIR} \, M_r^{DIR} >= \rho_{min,r}^Q \, M_r^Q,
$$

then

$$
Recall_r^{DIR} >= Recall_r^Q.
$$

In particular, DIR weakly dominates question-only retrieval whenever the
checkpoint state shifts enough bridge posterior mass into the selected top-`k`
set and does not reduce the worst-case retrieval success probability too much.

#### Proof

By Theorem 5.2,

$$
Recall_r^{DIR} >= \rho_{min,r}^{DIR} M_r^{DIR}.
$$

By assumption,

$$
\rho_{min,r}^{DIR} M_r^{DIR} >= \rho_{min,r}^Q M_r^Q.
$$

Using the stated lower bound for question-only retrieval yields

$$
Recall_r^{DIR} >= Recall_r^Q.
$$

QED.

### Remark 5.7. What is and is not proved

Theorem 5.2, Proposition 5.3, Corollary 5.5, and Corollary 5.6 are coverage
statements under the latent-bridge model of `EAMD_MATH_V2.md`. They do not
prove universal dominance for all multi-hop QA instances, and the
entropy-triggered result depends on the explicit calibration assumption.

## 6. Evidence-Conditional Remasking Within a Trajectory

We now extend Section 7 of `EAMD_MATH_V2.md` from one-shot evidence updates to
mid-trajectory checkpoint updates.

Fix checkpoint `r` and a committed position `j \in M_{t_r}`. Define

$$
p_{old,j}(x) := p_theta(A^j = x \mid x_{t_r}, Q, C_r, t_r),
$$

$$
p_{new,j}(x) := p_theta(A^j = x \mid x_{t_r}, Q, C_{r+1}, t_r).
$$

### Theorem 6.1. Exact checkpoint refresh value

The exact local value of refreshing token `j` after the evidence update
`C_r -> C_{r+1}` is

$$
\Delta_{j,r}^{refresh}
:= D_{KL}(p_{old,j} || p_{new,j}).
$$

#### Proof

This is Theorem 7.1 of `EAMD_MATH_V2.md` applied with `C_0 := C_r` and
`C_1 := C_{r+1}` at the fixed checkpoint state `x_{t_r}`. QED.

### Proposition 6.2. Time-dependent remask prior

Let `\sigma_{t,max}` be the admissibility ceiling from the ReMDM generalized
posterior family, and assume `\sigma_{t,max}` is nonincreasing along reverse
steps. Fix a base coefficient `bar \pi_j in (0,1]` and define

$$
\pi_{j,t} := bar \pi_j \frac{\sigma_{t,max}}{\sigma_{t_0,max}}.
$$

Then:

1. `0 < \pi_{j,t} <= bar \pi_j <= 1`,
2. `\pi_{j,t}` is nonincreasing as `t -> 0`, and
3. remasking is easier earlier and harder later.

#### Proof

Because `\sigma_{t,max} <= \sigma_{t_0,max}` and both are nonnegative,

$$
0 < \frac{\sigma_{t,max}}{\sigma_{t_0,max}} <= 1.
$$

Multiplying by `bar \pi_j in (0,1]` gives item 1.

If `\sigma_{t,max}` is nonincreasing along reverse denoising, then the ratio
`\sigma_{t,max} / \sigma_{t_0,max}` is also nonincreasing, proving item 2.
Item 3 is a restatement of item 2 in probabilistic terms. QED.

### Theorem 6.3. Optimal checkpoint remask probability

With the time-dependent prior `\pi_{j,t_r}` from Proposition 6.2 and checkpoint
revision cost `c_{j,r} >= 0`, consider the local objective

$$
max_{0 <= rho <= 1}
\Big[
 rho \, \Delta_{j,r}^{refresh} - c_{j,r} \, rho
 - \tau_r \, D_{KL}(Bern(rho) || Bern(\pi_{j,t_r}))
\Big].
$$

Its unique optimizer is

$$
\rho_{j,r}^*
=
\sigma\left(
\logit(\pi_{j,t_r}) + \frac{\Delta_{j,r}^{refresh} - c_{j,r}}{\tau_r}
\right).
$$

#### Proof

This is Theorem 7.3 of `EAMD_MATH_V2.md` with the base prior replaced by the
checkpoint prior `\pi_{j,t_r}`. Strict concavity is unchanged, so the optimizer
is unique. QED.

### Corollary 6.4. ReMDM validity at checkpoints

Define

$$
\sigma_{j,r}^* := min\{ \sigma_{t_r,max}, \rho_{j,r}^* \}.
$$

Then the checkpoint remask rule is valid within the ReMDM generalized-posterior
family.

#### Proof

This is Corollary 7.4 of `EAMD_MATH_V2.md` applied at checkpoint `r`. The clip
ensures admissibility under the ReMDM ceiling `\sigma_{t_r,max}`. QED.

### Assumption 6.5. Separable local cross-entropy surrogate

Let `S \subseteq M_{t_r}` be the set of committed positions remasked at
checkpoint `r`. Assume the post-update answer cross-entropy admits the local
surrogate

$$
CE_r(S) = CE_r(\emptyset) - \sum_{j \in S} (\Delta_{j,r}^{refresh} - c_{j,r}),
$$

and suppose at most `m` positions can be remasked.

This is the standard mean-field separability assumption: each remasked token
contributes additively to the surrogate objective.

### Theorem 6.6. Top-`m` divergent remasking is optimal

Under Assumption 6.5, among all remask sets `S \subseteq M_{t_r}` with
`|S| <= m`, the set minimizing `CE_r(S)` is obtained by selecting the `m`
largest positive margins

$$
margin_{j,r} := \Delta_{j,r}^{refresh} - c_{j,r}.
$$

Equivalently, if the costs are equal across positions, the optimal policy is to
remask the top-`m` positions with largest divergence
`\Delta_{j,r}^{refresh}`.

#### Proof

By Assumption 6.5,

$$
CE_r(S) = CE_r(\emptyset) - \sum_{j \in S} margin_{j,r}.
$$

Thus minimizing `CE_r(S)` is equivalent to maximizing
`sum_{j \in S} margin_{j,r}` subject to `|S| <= m`.

Let `S^*` be the set of indices corresponding to the `m` largest positive
margins. Suppose some feasible set `S` differs from `S^*`. Then there exists
`i \in S^* \setminus S` and `j \in S \setminus S^*` with
`margin_{i,r} >= margin_{j,r}`. Replacing `j` by `i` does not decrease the
objective. Repeating this exchange argument yields `S^*`. If a margin is
nonpositive, adding that index cannot improve the objective, so only positive
margins should be selected. QED.

### Remark 6.7. Ghost denoising step

Before applying the remask rule of Theorem 6.3, it is useful to perform one
additional forward pass with the *new* evidence \(C_{r+1}\) while keeping the
canvas \(x_{t_r}\) unchanged. Concretely: evaluate \(p_{\theta}(A^j \mid
x_{t_r}, Q, C_{r+1}, t_r)\) for all committed positions \(j \in M_{t_r}\),
but do *not* update the canvas. This "ghost" step serves two purposes:

1. It produces the updated posteriors \(p_{new,j}\) needed to compute the
   refresh value \(\Delta_{j,r}^{refresh}\) in Theorem 6.1.
2. It prevents *flickering*: if the new evidence is noisy or partially
   irrelevant, committing immediately could introduce errors that cascade
   through subsequent denoising steps. By evaluating without committing, the
   remask decision is informed by \(C_{r+1}\) but the canvas remains stable
   until remasking explicitly permits revision.

This ghost pass is precisely the \(+R\) extra forward-pass cost counted in
Proposition 8.1 (one per checkpoint). It is not a heuristic but a necessary
computation for the principled remask rule.

### Remark 6.8. Bridge-span sensitivity

In multi-hop QA, *bridge entities* (entities that connect the first hop to the
second) are disproportionately important for retrieval success. An
implementation-level refinement is entity-aware masking:

1. **Function-word commitment.** Commit high-confidence function words
   (determiners, prepositions, copulae) before bridge entities. These tokens
   carry little retrieval signal and their early commitment reduces canvas
   entropy without sacrificing query quality.

2. **Bridge-entity retrieval trigger.** When the confidence score
   \(conf_{r,j}\) of a bridge-entity position \(j\) first exceeds a threshold
   \(\tau_{bridge}\), trigger a retrieval checkpoint. This adaptive rule
   supplements the reliability-gated schedule of Proposition 4.3 with an
   entity-level signal.

These are implementation-level design choices, not theorems. They are motivated
by the observation that bridge entities drive the recall gap between DIR and
question-only retrieval (Corollary 5.6).

## 7. Convergence and Termination of the DIR Loop

The exact entropy telescoping result of Theorem 6.4 in `EAMD_MATH_V2.md` was
stated for repeated refinements at a fixed denoising state. In DIR the
checkpoint states vary with `r`, so the same telescoping argument does not apply
verbatim. We therefore prove the convergence facts that DIR actually needs:
finite-horizon summability and finite-corpus stabilization.

### Proposition 7.1. Finite-horizon summability

For any fixed checkpoint budget `R < \infty`,

$$
\sum_{r=0}^{R-1} G_r < \infty.
$$

#### Proof

Each `G_r` is a conditional mutual information and therefore nonnegative and
finite on a finite vocabulary and finite evidence set. A finite sum of finite
nonnegative numbers is finite. QED.

### Corollary 7.2. Finite-horizon termination

A DIR sampler with a predetermined finite checkpoint set
`{t_0, t_1, ..., t_R}` terminates after at most `R` retrieval updates.

#### Proof

By construction, retrieval is invoked only at the prescribed checkpoints. There
are at most `R` such checkpoints. QED.

### Theorem 7.3. Finite-corpus stabilization for adaptive DIR

Assume:
- retrieval is deduplicated,
- all evidence passages are drawn from a finite corpus `U`, and
- every nonempty update `\Delta C_{r+1}` adds at least one previously unseen
  passage.

Then an adaptive DIR controller can perform at most `|U|` nontrivial retrieval
updates. Consequently:

1. `\Delta C_{r+1} = \emptyset` for all sufficiently large `r`,
2. `G_r = 0` for all sufficiently large `r`,
3. the gain sequence `(G_r)` is summable, and
4. `G_r -> 0`.

#### Proof

Because the evidence sets are nested and each nonempty update adds at least one
new passage, there can be at most `|U|` strict enlargements before no unseen
passage remains. After stabilization, `C_{r+1} = C_r`, so
`\Delta C_{r+1} = \emptyset`.

Applying Theorem 2.1 with `\Delta C_{r+1} = \emptyset` gives

$$
G_r = I(A ; \emptyset \mid x_{t_r}, Q, C_r) = 0.
$$

Hence only finitely many terms of `(G_r)` are nonzero. Therefore the series is
summable and the terms converge to `0`. QED.

### Remark 7.4. Honest scope of the convergence result

Theorem 7.3 proves termination and vanishing gains. It does *not* prove global
contraction of the full DIR operator, nor monotonic improvement of EM/F1.
Those remain outside the current theorem set.

## 8. Efficiency Accounting

We now count model forward passes under the standard implementation of each
method.

### Proposition 8.1. Exact model-call counts

Assume:
- a vanilla dLLM denoiser uses one forward pass per reverse step,
- DIR reuses the current checkpoint forward pass and performs exactly one extra
  forward pass after each retrieval update to obtain `p_new`,
- ARAM evaluates both the context-conditioned and prior branches at every step,
- IRCoT performs `T_{AR}` autoregressive token-generation forward passes per
  retrieval round.

Then the model-call counts are:

1. Vanilla dLLM: `T` forward passes.
2. DIR: `T + R` forward passes.
3. ARAM: `2T` forward passes.
4. IRCoT: `R T_{AR}` forward passes.

#### Proof

1. Vanilla dLLM performs one denoiser call at each of `T` reverse steps.
2. DIR performs the same `T` denoiser calls, plus one additional re-evaluation
   at each of the `R` checkpoints after updating the evidence.
3. ARAM evaluates two branches per step, so the count is `2T`.
4. IRCoT regenerates autoregressively at each round; if a round emits
   `T_{AR}` tokens, it requires `T_{AR}` forward passes, hence `R T_{AR}` over
   `R` rounds.

QED.

### Corollary 8.2. Utility condition for compute justification

Let `c_model > 0` be the per-forward-pass compute cost. Relative to vanilla
dLLM, DIR has extra compute cost `R c_model`. Relative to vanilla, the expected
net benefit is positive whenever

$$
\sum_{r=0}^{R-1} \mathbb E[G_r]
>
R c_{model}
+ \lambda_{ret} \sum_{r=0}^{R-1} \mathbb E[cost_{ret}(\Delta C_{r+1})]
+ \lambda_{rev} \sum_{r=0}^{R-1} \mathbb E[cost_{rev}(\rho_r)].
$$

#### Proof

This is immediate from the finite-horizon objective `J(pi)` after isolating the
`R` extra model calls of DIR relative to vanilla dLLM. QED.

### Remark 8.3. What efficiency is and is not claimed

Proposition 8.1 is an exact call-count statement. It does not by itself prove
that DIR is faster wall-clock than AR methods; actual latency depends on
batching, retrieval backend, and hardware utilization.

### Definition 8.4. Asynchronous DIR

In the asynchronous DIR variant, checkpoint `r` is split into two times:

- a retrieval-launch step `\ell_r`, at which the retriever is launched using
  the current working-memory query `Q \oplus \phi_r \oplus h_r`, and
- an evidence-injection step `\tau_r^{sync} <= \ell_r`, at which the retriever
  has returned, the new evidence is injected, and the ghost pass / remask
  update is actually performed.

Between `\ell_r` and `\tau_r^{sync}`, denoising may continue under the stale
evidence `C_r`. All mathematical quantities associated with the evidence update
are defined at the synchronization state

$$
\tilde s_r := (x_{\tau_r^{sync}}, M_{\tau_r^{sync}}, Q, C_r).
$$

### Proposition 8.5. Synchronous-equivalence at the injection state

Let asynchronous DIR be defined as in Definition 8.4. Then Theorem 2.1,
Theorem 6.1, Theorem 6.3, and Proposition 7.1 apply verbatim when the checkpoint
state `s_r` is replaced by the synchronization state `\tilde s_r`.

#### Proof

Each of the listed results is stated conditionally on the state at which the
evidence update `C_r -> C_{r+1}` is actually applied. Their proofs depend only
on that local state and on the updated evidence, not on the earlier
retrieval-launch step. Substituting `\tilde s_r` for `s_r` therefore leaves all
proofs unchanged. QED.

### Corollary 8.6. Overlap bound for asynchronous retrieval

Let `L_r` be the retrieval latency launched at step `\ell_r`, let `D_r` be the
denoising wall-clock time available between `\ell_r` and `\tau_r^{sync}`, and
let `c_{sync}` be the synchronization overhead of injection plus the ghost pass.
Then the additional wall-clock overhead of checkpoint `r` satisfies

$$
Overhead_r <= \max\{0, L_r - D_r\} + c_{sync}.
$$

In particular, if `L_r <= D_r`, the retrieval latency is fully hidden except
for the synchronization overhead.

#### Proof

During the interval of length `D_r`, retrieval and denoising overlap. Only the
uncovered portion of retrieval latency can add extra wall-clock time, which is
at most `\max\{0, L_r - D_r\}`. Adding the deterministic synchronization overhead
gives the result. QED.

### Remark 8.7. Asynchronous retrieval

The \(+R\) extra forward passes in Proposition 8.1 assume synchronous execution:
denoising halts while retrieval completes. In practice, the retriever runs on
CPU while the denoiser occupies the GPU. Under Definition 8.4 and Corollary
8.6, the wall-clock latency of DIR approaches that of vanilla dLLM whenever
retrieval latency is hidden behind denoising computation.

### Definition 8.8. Recovery Rate

For checkpoint \(r\), let \(W_r \subseteq M_{t_r}\) be the set of committed
positions whose current token disagrees with the ground-truth answer, and let
\(W_r^+ \subseteq W_r\) be the subset that become correct after the
remask-and-re-denoise step. Define the **Recovery Rate** at checkpoint \(r\) as

$$
RR_r :=
\begin{cases}
\frac{|W_r^+|}{|W_r|}, & |W_r| > 0, \\
1, & W_r = \emptyset.
\end{cases}
$$

This is the primary ablation diagnostic for DIR: it measures the fraction of
wrong tokens that are corrected by a single retrieval-remask cycle. A high
\(RR_r\) indicates that the evidence update and remask rule are effectively
targeting erroneous positions.

### Assumption 8.9. Lexical-support approximation

Fix checkpoint `r` and position `j`. Let

$$
s_{j,r}(v) := \mathbf 1[v \text{ appears in } \Delta C_{r+1} \setminus C_r].
$$

Assume the true evidence score ratio at that position admits the approximation

$$
r_{j,r}(v)
:=
\log \frac{p_{new,j}(v)}{p_{old,j}(v)}
=
\beta_{j,r} + \alpha_r s_{j,r}(v) + \varepsilon_{j,r}(v),
$$

where `\beta_{j,r}` is a token-independent offset and `\varepsilon_{j,r}(v)` is
an approximation error.

### Proposition 8.10. Contrastive evidence bonus as a first-order surrogate

Under Assumption 8.9, maximizing the adjusted logits

$$
\tilde \ell_{j,r}(v) := \ell_{new,j}(v) + \alpha_r s_{j,r}(v)
$$

is equivalent, up to the additive constant `\beta_{j,r}`, to maximizing the
first-order surrogate

$$
\log p_{new,j}(v) + r_{j,r}(v) - \varepsilon_{j,r}(v).
$$

#### Proof

Since `\ell_{new,j}(v)` differs from `\log p_{new,j}(v)` only by a
token-independent log-normalizer, adding `\alpha_r s_{j,r}(v)` yields

$$
\tilde \ell_{j,r}(v)
=
\log p_{new,j}(v) + \alpha_r s_{j,r}(v) + const.
$$

Substitute the decomposition from Assumption 8.9:

$$
\alpha_r s_{j,r}(v)
=
r_{j,r}(v) - \beta_{j,r} - \varepsilon_{j,r}(v).
$$

Hence

$$
\tilde \ell_{j,r}(v)
=
\log p_{new,j}(v) + r_{j,r}(v) - \varepsilon_{j,r}(v) + const',
$$

where `const'` is token-independent. Since token-independent constants do not
change the argmax, the claim follows. QED.

### Remark 8.11. Contrastive evidence guidance

After the evidence update \(C_r \to C_{r+1}\), the *new* evidence
\(\Delta C_{r+1}\) may contain tokens that substantially shift the posterior
at certain positions. A heuristic implementation strategy is contrastive
evidence guidance: up-weight the contribution of tokens that appear in
\(\Delta C_{r+1}\) but not in \(C_r\) when computing the denoiser logits.
Concretely, for each vocabulary token \(v\) at position \(j\), add a bonus
\(\alpha \cdot \mathbf{1}[v \in \Delta C_{r+1} \setminus C_r]\) to the logit
before softmax.

Proposition 8.10 does not prove that the lexical-support approximation is
accurate; it only shows that, when the approximation is accurate, the
contrastive bonus is the corresponding first-order surrogate.

## 9. Comparison with TTD-DR

This section is descriptive rather than theorem-based.

TTD-DR is the closest conceptual prior because it also combines diffusion-style
refinement with retrieval updates during inference. The differences are:

1. Granularity:
   - TTD-DR operates at draft / report refinement level.
   - DIR operates at token / short-answer denoising checkpoints.

2. Task:
   - TTD-DR targets open-ended deep-research generation.
   - DIR targets finite-horizon multi-hop QA with explicit evidence updates.

3. Revision mechanism:
   - TTD-DR revises drafts heuristically at the text level.
   - DIR uses ReMDM-valid token remasking with the exact KL refresh value from
     Theorem 6.1 and the checkpoint rule of Theorem 6.3.

4. Objective:
   - TTD-DR is oriented toward open-ended research refinement.
   - DIR is derived from a finite-horizon evidence-ascent objective
     `J(pi) = E[sum_r U_r]` with explicit retrieval and revision costs.

These differences should be stated explicitly in the paper. The novelty claim
should be *intra-trajectory, token-level evidence refinement with principled
remasking for multi-hop QA*, not merely `retrieval during diffusion`.

## 10. Final DIR Summary

DIR is the mathematically defensible extension of EAMD to iterative multi-hop
retrieval inside a single diffusion trajectory.

The theorem-safe backbone is:
- Theorem 2.1: checkpoint evidence-gain identity.
- Theorem 3.1: telescoping answer-score identity.
- Proposition 4.3: reliability-gated usefulness condition.
- Theorem 5.2, Proposition 5.3, Corollary 5.5, and Corollary 5.6: checkpoint
  query coverage, top-`k` advantage, entropy-triggered parallel retrieval, and
  question-only dominance conditions.
- Theorem 6.1 through Theorem 6.6: exact refresh value, checkpoint remask rule,
  and top-`m` remask optimality.
- Theorem 7.3: finite-corpus stabilization and termination.
- Proposition 8.1, Proposition 8.5, Corollary 8.6, and Proposition 8.10:
  exact model-call counts, asynchronous injection-state equivalence, overlap
  bound, and the contrastive-evidence surrogate.

## References

- EAMD v2: `/projects/prjs1800/msc-thesis/07-daes/EAMD_MATH_V2.md`
- ARAM: https://arxiv.org/abs/2603.17677
- SPREAD: https://arxiv.org/abs/2601.11342
- DLLM-Searcher: https://arxiv.org/abs/2602.07035
- IRCoT: https://arxiv.org/abs/2212.10509
- TTD-DR: https://arxiv.org/abs/2507.16075
