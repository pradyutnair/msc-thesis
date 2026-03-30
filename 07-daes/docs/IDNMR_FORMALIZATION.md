# iDNMR: Iterative Diffusion-Native Multi-Query Retrieval

## 0. Status of Claims

### Theorem-backed in this document

1. The diffusion bridge-proposal distribution induces a well-defined posterior over bridge hypotheses by pushforward.
2. Under a fixed candidate budget `k`, posterior `TopK` bridge extraction maximizes retained bridge mass among all size-`k` candidate sets.
3. Distribution-based bridge extraction therefore weakly dominates answer-conditioned extraction at the bridge-coverage level, and strictly dominates whenever the answer-conditioned set differs from the posterior `TopK` set under a non-degenerate posterior.
4. Under a standard bridge-hit assumption on the retriever, the posterior mass of the candidate set yields a lower bound on next-hop retrieval success.
5. Under conditional independence of query hits, multi-query retrieval has the exact coverage formula
   $$
   1 - \prod_{b \in S}(1-\rho_r(b)),
   $$
   and therefore increases monotonically with additional nonzero-probability bridge queries.
6. The answer-support increments telescope exactly:
   $$
   \sum_{r=1}^{R}\big(\log p_r(a)-\log p_{r-1}(a)\big)=\log p_R(a)-\log p_0(a).
   $$
7. With a finite corpus and deduplicated evidence union, the evidence chain stabilizes in finitely many rounds; if the algorithm stops on zero evidence gain, it terminates in finitely many rounds.

### Assumption-backed in this document

1. The theoretical bridge posterior is the exact pushforward posterior over bridge strings. The implementation approximates `TopK` by truncated branching over high-probability initial tokens and short denoising completions.
2. The bridge-hit assumption says that if the correct bridge is queried, the retriever returns the required next-hop evidence with probability at least `η > 0`.
3. The product-form multi-query coverage theorem assumes conditional independence of per-query retrieval-hit events given the current round state.
4. The latent bridge variable is a task-level abstraction: multi-hop QA may admit multiple equivalent bridge strings; the theory works over equivalence classes after canonicalization.

### Not claimed

1. That the implemented branch extractor computes the exact `TopK` bridge posterior.
2. That iDNMR is universally faster than every AR system at equal quality.
3. That iDNMR is guaranteed to beat single-round `Pool` on every dataset; the theory guarantees coverage advantages under its assumptions, not universal empirical dominance.

## 1. Problem Setup and Notation

Let:

- `q` be the question.
- `D` be a finite retrieval corpus.
- `A` be the final answer random variable taking values in a discrete answer space `𝒜`.
- `C_r ⊆ D` be the evidence set after retrieval round `r`, with
  $$
  C_0 \subseteq C_1 \subseteq \cdots \subseteq C_R.
  $$
- `ΔC_{r+1} := C_{r+1} \setminus C_r` be the newly added evidence at round `r+1`.

Let the masked diffusion language model be parameterized by `θ`. As in the prior EAMD formalization, let
$$
p_r(a) := p_\theta(a \mid q, C_r)
$$
denote the answer distribution under evidence `C_r`.

For each round `r`, define a bridge hypothesis space `ℬ_r`. Elements `b ∈ ℬ_r` are short normalized text strings intended to retrieve the next-hop evidence needed to answer `q` from `C_r`. They may be named entities, relation-bearing fragments, or short bridge statements. In the exact theory, `ℬ_r` is the support of the normalized diffusion proposal variable defined below. In the implementation, `ℬ_r` is approximated by the finite set of unique candidate strings returned by the bridge extractor after string normalization and deduplication.

The key distinction between methods is:

- **Answer-conditioned extraction**: derive bridge candidates from a committed answer string `\hat a_r`.
- **Distribution-based extraction**: derive bridge candidates from the diffusion model's posterior proposal distribution before collapsing to a single committed bridge.

This document formalizes the latter.

## 2. Diffusion Bridge Posterior

### 2.1 Proposal canvas and pushforward distribution

At round `r`, define a short masked proposal canvas `Z_{r,T}` of length `L`, initialized as all-mask tokens. The diffusion model denoises this canvas conditional on `(q, C_r)`:
$$
Z_{r,0} \sim p_\theta(\cdot \mid Z_{r,T}, q, C_r).
$$

Let `U_r` be the final decoded proposal string obtained from `Z_{r,0}`. Let
$$
\psi : \Sigma^{\le L} \to \mathcal B_r
$$
be a normalization map that lowercases, trims punctuation or whitespace artifacts, and merges duplicate surface forms that the implementation treats as the same retrieval query string. Thus `ℬ_r` should be read concretely as a finite set of normalized candidate strings, not as an abstract semantic ontology. Define the latent bridge variable
$$
B_r^\star := \psi(U_r).
$$

### Definition 2.1. Bridge posterior

The diffusion-induced bridge posterior at round `r` is the pushforward distribution
$$
\pi_r(b)
:=
\mathbb P_\theta(B_r^\star = b \mid q, C_r)
=
\sum_{u:\psi(u)=b}\mathbb P_\theta(U_r=u \mid q, C_r),
\qquad b \in \mathcal B_r.
$$

This is the fundamental object used by iDNMR.

### 2.2 TopK bridge extraction

Fix a round `r` and a candidate budget `k ≥ 1`. Let
$$
S_r^{(k)} := \operatorname{TopK}(\pi_r, k)
$$
be any size-`k` set of bridge hypotheses with maximal posterior masses. More explicitly, if
$$
\pi_r(b_{(1)}) \ge \pi_r(b_{(2)}) \ge \cdots,
$$
then
$$
S_r^{(k)} = \{b_{(1)},\dots,b_{(k)}\}
$$
up to tie-breaking.

Define the retained bridge mass of a candidate set `S ⊆ ℬ_r` by
$$
\Pi_r(S) := \sum_{b \in S}\pi_r(b) = \mathbb P_\theta(B_r^\star \in S \mid q, C_r).
$$

### Proposition 2.2. TopK posterior optimality

Among all size-`k` subsets of `ℬ_r`, `S_r^{(k)}` maximizes retained bridge mass:
$$
\Pi_r(S_r^{(k)}) = \max_{S \subseteq \mathcal B_r,\ |S|=k}\Pi_r(S).
$$

#### Proof

Order the bridge hypotheses so that
$$
\pi_r(b_{(1)}) \ge \pi_r(b_{(2)}) \ge \cdots.
$$
Let `T` be any size-`k` subset of `ℬ_r`. Write its elements in decreasing posterior order as
$$
\pi_r(t_1) \ge \cdots \ge \pi_r(t_k).
$$
Then for every `i = 1,\dots,k`,
$$
\pi_r(t_i) \le \pi_r(b_{(i)}),
$$
because otherwise there would be more than `i-1` hypotheses with posterior mass at least `\pi_r(t_i)` outside the ordered top-`i` list, contradicting the ordering of `b_{(i)}`.

Summing over `i` gives
$$
\Pi_r(T)
=
\sum_{i=1}^k \pi_r(t_i)
\le
\sum_{i=1}^k \pi_r(b_{(i)})

=
\Pi_r(S_r^{(k)}).
$$
Hence `S_r^{(k)}` maximizes retained bridge mass. QED.

## 3. Why Distribution-Based Extraction Dominates Answer-Conditioned Extraction

Answer-conditioned extraction first collapses the model state to a committed answer estimate `\hat a_r`, then applies a deterministic bridge-extraction map
$$
\phi_r : \hat a_r \mapsto T_r \subseteq \mathcal B_r,
$$
where `|T_r| = k` matches the same candidate budget used by iDNMR.

The essential difference is:

- answer-conditioned extraction chooses a candidate set after committing to one answer string;
- iDNMR chooses the candidate set directly from posterior support over bridge hypotheses.

### Theorem 3.1. Distribution-based extraction weakly dominates any equal-budget answer-conditioned set

Let `T_r ⊆ ℬ_r` be any size-`k` candidate set produced by an answer-conditioned extractor at round `r`. Then
$$
\Pi_r(S_r^{(k)}) \ge \Pi_r(T_r).
$$
Moreover, the inequality is strict whenever `T_r` omits some bridge in the posterior top-`k` set and the omitted bridge has strictly larger posterior mass than some bridge included in `T_r`.

#### Proof

The weak inequality is immediate from Proposition 2.2 because `T_r` is a size-`k` subset of `ℬ_r`.

For strictness, suppose `T_r \neq S_r^{(k)}` and there exists `b^+ \in S_r^{(k)} \setminus T_r` and `b^- \in T_r \setminus S_r^{(k)}` such that
$$
\pi_r(b^+) > \pi_r(b^-).
$$
Then replacing `b^-` by `b^+` strictly increases retained mass:
$$
\Pi_r\big((T_r \setminus \{b^-\}) \cup \{b^+\}\big)

=
\Pi_r(T_r) - \pi_r(b^-) + \pi_r(b^+)
>
\Pi_r(T_r).
$$
Applying this exchange repeatedly transforms `T_r` into `S_r^{(k)}` while strictly increasing mass at each step. Therefore
$$
\Pi_r(S_r^{(k)}) > \Pi_r(T_r).
$$
QED.

### Corollary 3.2. Point-estimate extraction is a special weak case

If answer-conditioned extraction collapses to a single bridge estimate `\hat b_r` and we compare against a size-`k` posterior set `S_r^{(k)}` with `k > 1`, then
$$
\Pi_r(S_r^{(k)}) \ge \pi_r(\hat b_r),
$$
with strict inequality whenever posterior mass outside `\hat b_r` is positive on at least one other top-`k` bridge.

#### Proof

Take `T_r = \{\hat b_r\}` for `k=1`, or augment it arbitrarily to size `k` without changing the fact that `S_r^{(k)}` is optimal by Proposition 2.2. Strictness follows from Theorem 3.1 whenever the posterior is not degenerate on a single bridge. QED.

### Remark 3.3. Relation to the empirical iPool vs iDNMR gap

Theorem 3.1 does not claim that every answer-conditioned implementation is always empirically worse in EM/F1. It states the precise mechanism-level advantage: under a fixed candidate budget, posterior-support extraction captures at least as much bridge mass as any candidate set chosen after collapsing to a committed answer string. The iPool-to-iDNMR gain is the empirical manifestation of this coverage advantage.

### Remark 3.4. Practical gap beyond the theorem

Theorem 3.1 is stated at the bridge-mass level and therefore assumes that the answer-conditioned extractor outputs a valid size-`k` subset `T_r \subseteq \mathcal B_r`. In practice, answer-conditioned span extraction can fail more severely: it may return short spans around unstable committed tokens that are not task-useful bridges at all. The theorem does not rely on this stronger failure mode, but empirically it makes the iPool-to-iDNMR gap easier to observe. Put differently, Theorem 3.1 already shows that posterior-support extraction dominates even when the answer-conditioned extractor stays inside the bridge space; the real implementation gap can be larger because answer-conditioned extraction may leave that idealized space.

## 4. Multi-Query Retrieval Coverage

Let `E_{r+1}(S)` denote the event that the retrieval stage at round `r+1`, given candidate set `S`, returns at least one task-useful next-hop passage. We formalize retrieval quality in two ways.

### 4.1 Minimal bridge-hit lower bound

Assume there exists `η_r ∈ (0,1]` such that for every bridge `b`,
$$
\mathbb P(E_{r+1}(S) \mid B_r^\star = b,\ q,\ C_r) \ge \eta_r \,\mathbf 1\{b \in S\}.
$$
This says: if the true bridge is queried, the retriever returns useful next-hop evidence with probability at least `η_r`.

### Proposition 4.1. Posterior-mass lower bound on retrieval success

Under the bridge-hit assumption above,
$$
\mathbb P(E_{r+1}(S) \mid q, C_r)
\ge
\eta_r \,\Pi_r(S).
$$

#### Proof

By conditioning on `B_r^\star`,
$$
\mathbb P(E_{r+1}(S)\mid q,C_r)
=
\sum_{b \in \mathcal B_r}
\mathbb P(E_{r+1}(S)\mid B_r^\star=b,q,C_r)\,\pi_r(b).
$$
Applying the bridge-hit assumption yields
$$
\mathbb P(E_{r+1}(S)\mid q,C_r)
\ge
\sum_{b\in \mathcal B_r} \eta_r \mathbf 1\{b\in S\}\pi_r(b)
=
\eta_r \sum_{b\in S}\pi_r(b)
=
\eta_r \Pi_r(S).
$$
QED.

### Corollary 4.2. Retrieval-success dominance of iDNMR over answer-conditioned extraction

For any size-`k` answer-conditioned set `T_r`,
$$
\mathbb P(E_{r+1}(S_r^{(k)}) \mid q,C_r)
\ge
\eta_r \Pi_r(S_r^{(k)})
\ge
\eta_r \Pi_r(T_r).
$$
If the strict inequality conditions of Theorem 3.1 hold, then the lower bound for iDNMR is strictly larger.

#### Proof

Combine Proposition 4.1 with Theorem 3.1. QED.

### 4.2 Exact multi-query coverage under conditional independence

For a candidate set `S`, define the per-query hit probability
$$
\rho_r(b) := \mathbb P(E_{r+1}(\{b\}) \mid q, C_r).
$$

Assume the hit events for different bridge queries are conditionally independent given `(q,C_r)`. Then:

### Theorem 4.3. Multi-query coverage formula

For any finite candidate set `S`,
$$
\mathbb P(E_{r+1}(S)\mid q,C_r)
=
1 - \prod_{b \in S}(1-\rho_r(b)).
$$

#### Proof

Under conditional independence,
$$
\mathbb P(\text{no hit from any }b\in S \mid q,C_r)
=
\prod_{b\in S}\mathbb P(E_{r+1}(\{b\})^c \mid q,C_r)
=
\prod_{b\in S}(1-\rho_r(b)).
$$
Taking the complement gives
$$
\mathbb P(E_{r+1}(S)\mid q,C_r)
=
1-\prod_{b\in S}(1-\rho_r(b)).
$$
QED.

### Remark 4.4. Correlation caveat

Theorem 4.3 is exact only under conditional independence of per-query hit events. Real bridge queries are often correlated, especially when two candidates refer to nearby entities or paraphrastic variants, so the product formula should not be treated as an exact law outside that assumption. In overlap-heavy positive-correlation regimes, the independence expression is often an optimistic upper envelope for marginal gains from adding more queries. Theorem 4.3 is therefore best read as an exact idealized coverage formula plus a monotonicity intuition, while Proposition 4.1 remains the more assumption-robust lower bound used by the core argument.

### Corollary 4.5. Monotonicity in candidate-set size

If `S ⊆ S'` and `ρ_r(b) > 0` for at least one `b ∈ S' \setminus S`, then
$$
\mathbb P(E_{r+1}(S')\mid q,C_r) > \mathbb P(E_{r+1}(S)\mid q,C_r).
$$

#### Proof

By Theorem 4.3,
$$
1-\prod_{b\in S'}(1-\rho_r(b))

=
1-\Big(\prod_{b\in S}(1-\rho_r(b))\Big)\Big(\prod_{b\in S'\setminus S}(1-\rho_r(b))\Big).
$$
If at least one newly added `ρ_r(b)` is strictly positive, then
$$
\prod_{b\in S'\setminus S}(1-\rho_r(b)) < 1,
$$
so the no-hit probability strictly decreases and the hit probability strictly increases. QED.

## 5. The iDNMR Algorithm

At round `r`, iDNMR keeps two objects:

- the current evidence set `C_r`,
- the current answer estimate `\hat a_r`.

The answer estimate is used as one retrieval query, but the distinctive bridge mechanism is the posterior-support candidate set `S_r^{(k)}`.

### Algorithm 5.1. iDNMR

**Inputs**

- question `q`
- corpus `D`
- retriever `R`
- diffusion model `p_\theta`
- initial evidence `C_0`
- bridge budget `k`
- maximum rounds `R_max`
- stopping rule `Stop`

**Initialization**

1. Decode a seed answer `\hat a_0` under `C_0`.

**For** `r = 0,1,\dots,R_max-1`:

1. Compute or approximate the bridge posterior `\pi_r(b) = P_\theta(B_r^\star=b \mid q,C_r)`.
   Exact theory: `\pi_r` is the pushforward distribution from Definition 2.1.
   Implementation: `\pi_r` is approximated by truncated branching over high-probability initial tokens followed by short denoising completions and string deduplication.
2. Select the posterior-support candidate set
   $$
   S_r^{(k)} = \operatorname{TopK}(\pi_r, k).
   $$
3. Form the retrieval query family
   $$
   \mathcal Q_r = \{q \oplus \hat a_r\} \cup \{q \oplus b : b \in S_r^{(k)}\}.
   $$
4. Retrieve from `D` with all queries in `𝒬_r`, obtaining new evidence `ΔC_{r+1}`.
5. Update evidence by deduplicated union:
   $$
   C_{r+1} = C_r \cup \Delta C_{r+1}.
   $$
6. Decode the new answer estimate
   $$
   \hat a_{r+1} = \arg\max_a p_\theta(a \mid q, C_{r+1})
   $$
   or its deterministic diffusion-decoding analogue.
7. If `Stop(C_r, C_{r+1}, \hat a_r, \hat a_{r+1})` is true, terminate.

**Output**

- final answer `\hat a_R`
- evidence trajectory `(C_0,\dots,C_R)`

### Remark 5.2. Pool, iPool, and iDNMR in one language

- `Pool`: execute only round `0`, using one distribution-based expansion.
- `iPool`: execute multiple rounds, but replace Step 1 by an answer-conditioned bridge extractor.
- `iDNMR`: execute multiple rounds with posterior-support extraction at every round.

This separation is important: iDNMR is not "iterative Pool" in the generic sense. Its claim is that iterative retrieval only works when the bridge extractor remains posterior-based rather than collapsing to committed answer tokens.

## 6. Telescoping Answer Support and Finite-Horizon Convergence

Define
$$
p_r(a) := p_\theta(a \mid q, C_r)
$$
and the round-`r` answer-support increment
$$
g_r(a) := \log p_r(a) - \log p_{r-1}(a), \qquad r \ge 1.
$$

### Proposition 6.1. Telescoping answer support

For every fixed answer `a`,
$$
\sum_{r=1}^{R} g_r(a)
=
\log p_R(a) - \log p_0(a).
$$

#### Proof

By substitution,
$$
\sum_{r=1}^{R}\big(\log p_r(a)-\log p_{r-1}(a)\big)
=
(\log p_1(a)-\log p_0(a))
+(\log p_2(a)-\log p_1(a))
\cdots
+(\log p_R(a)-\log p_{R-1}(a)).
$$
All intermediate terms cancel, leaving
$$
\log p_R(a)-\log p_0(a).
$$
QED.

### Corollary 6.2. Interpretation

iDNMR can be viewed as building answer support by accumulating evidence-bearing retrieval rounds. Each successful round adds a positive increment for correct answers to the extent that `C_{r+1}` is more informative than `C_r`.

### Remark 6.2A. Telescoping is an accounting identity, not a success guarantee

Proposition 6.1 is exact but purely algebraic. By itself it does not imply that `g_r(a^\star) > 0` for the correct answer `a^\star`, nor that later rounds must help. Whether a particular increment is positive depends on retrieval quality: if `ΔC_{r+1}` is irrelevant or distracting, then `g_r(a^\star)` can be non-positive. The role of the telescoping identity is to show how answer support accumulates when retrieval succeeds, not to guarantee that every retrieval round is beneficial.

### Proposition 6.3. Monotone evidence chain

The evidence sets produced by iDNMR satisfy
$$
C_0 \subseteq C_1 \subseteq \cdots \subseteq C_R \subseteq D.
$$

#### Proof

By construction,
$$
C_{r+1} = C_r \cup \Delta C_{r+1},
$$
so `C_r ⊆ C_{r+1}` for every `r`. Since all retrieved passages come from `D`, we also have `C_r ⊆ D`. QED.

### Theorem 6.4. Finite stabilization of the evidence chain

Assume the corpus `D` is finite and evidence updates are deduplicated. Then the sequence `(C_r)` stabilizes after finitely many strict expansions. In particular, the number of rounds with `C_{r+1} \neq C_r` is at most `|D| - |C_0|`.

#### Proof

Every strict expansion `C_{r+1} \neq C_r` adds at least one passage from `D \setminus C_r`. Since `D` is finite, at most `|D|-|C_0|` such additions can occur. After that point, no strict evidence expansion is possible, hence `C_{r+1}=C_r`. QED.

### Theorem 6.5. Termination guarantee

Suppose iDNMR uses the stop rule
$$
\text{Stop} = \mathbf 1\{C_{r+1}=C_r\}
$$
or the slightly weaker practical rule
$$
\text{Stop} = \mathbf 1\{C_{r+1}=C_r \text{ and } \hat a_{r+1}=\hat a_r\}.
$$
Then iDNMR terminates in finitely many rounds.

#### Proof

Under the first stop rule, termination follows immediately from Theorem 6.4: once the evidence chain stabilizes, the algorithm stops.

Under the second rule, once `C_{r+1}=C_r`, the next decode is performed on the same evidence set. If the decoder is deterministic, then `\hat a_{r+1}=\hat a_r` and the algorithm stops at that round. More generally, if one allows one additional settling decode after evidence stabilization, termination occurs in at most one extra round beyond the first stabilized evidence state. QED.

### Remark 6.6. Why 2-3 rounds can help without contradicting finite termination

Theorems 6.4 and 6.5 are finite-horizon statements, not asymptotic claims that more rounds always help. Empirically, iDNMR can improve from one round to two or three rounds and then saturate. This is consistent with the theory: useful evidence increments can be nonzero for a few rounds and then decay to zero as the evidence chain stabilizes.

## 7. Connection to Prior Work

### 7.1 ARAM

ARAM is a single-evidence-set guidance method. Its central object is a guidance scale computed from the contrast between a context-conditioned branch and a prior/no-context branch. The retrieval set itself is fixed during decoding.

iDNMR differs in two ways:

1. Its primary object is not a guidance scale but a bridge posterior `\pi_r`.
2. It acts on retrieval coverage by selecting multiple bridge hypotheses and expanding evidence across rounds.

So ARAM is a **decoding-on-fixed-evidence** method; iDNMR is a **retrieval-expansion** method.

### 7.2 SPREAD

SPREAD reorders or prioritizes denoising based on query relevance under a single retrieved context. Like ARAM, it does not change the evidence set.

iDNMR is orthogonal: it changes which evidence is retrieved at each round by querying the retriever with posterior-support bridge hypotheses.

Thus SPREAD addresses **how to decode given evidence**, while iDNMR addresses **which evidence to acquire next**.

### 7.3 IRCoT

IRCoT performs iterative retrieval using a committed autoregressive reasoning trace. Its next query is therefore a point estimate derived from one generated chain.

iDNMR differs at the retrieval-query level:

1. it uses posterior support over multiple bridge hypotheses rather than a single committed reasoning trace;
2. it issues multiple bridge queries per round;
3. its theoretical advantage comes from retained bridge mass and multi-query coverage.

An autoregressive system could approximate this behavior by beam search or repeated sampling, but that is not the standard IRCoT mechanism. The formal novelty of iDNMR is the use of diffusion-induced posterior support as the retrieval object.

There is also an empirical confound in benchmark comparisons: IRCoT in our experiments uses an autoregressive reader that is stronger than the Dream-7B diffusion reader used for iDNMR. So the cleanest empirical ablation is not iDNMR versus IRCoT alone, but iDNMR versus iPool, where the retrieval framework, retriever, corpus, prompt, and backbone are held fixed and only the bridge-extraction mechanism changes.

### 7.4 Summary of the relation

- ARAM: adaptive guidance on fixed evidence.
- SPREAD: relevance-aware denoising on fixed evidence.
- IRCoT: iterative retrieval from a committed point-estimate reasoning path.
- iDNMR: iterative retrieval from posterior-support bridge sets induced by a diffusion proposal distribution.

This is the cleanest way to position the method: iDNMR is a retrieval-coverage method built from diffusion-native posterior support, not another guidance heuristic.

## 8. Practical Interpretation

The theory supports the following empirical reading:

1. If bridge extraction is answer-conditioned, later rounds may collapse onto one wrong committed bridge and hurt retrieval.
2. If bridge extraction remains posterior-based, later rounds can preserve multiple plausible bridge hypotheses and continue adding useful evidence.
3. Therefore iterative retrieval helps only when extraction quality is maintained across rounds.

That is exactly the distinction between iPool-like point-estimate extraction and iDNMR-style posterior-support extraction.

## 9. What This Formalization Does and Does Not Buy Us

What it buys:

- a principled bridge posterior object;
- a proof that `TopK` posterior extraction is optimal under a fixed bridge budget;
- a formal dominance statement over answer-conditioned point-estimate extraction;
- explicit retrieval-coverage bounds;
- a finite-horizon iterative retrieval framework with telescoping answer support.

What it does not buy automatically:

- statistical significance of a small empirical gap over single-round `Pool`;
- an explanation of why iDNMR should significantly beat `Pool` in every setting; the clean theorem-to-experiment link is iDNMR versus iPool, where the only changed ingredient is posterior-support extraction versus answer-conditioned extraction;
- guaranteed superiority on every dataset;
- a claim that diffusion guidance, rather than retrieval, is the main mechanism.

Those require empirical validation. The formal result is narrower and cleaner: **if the bridge posterior is the right object, then posterior-support retrieval is the optimal finite-budget way to preserve bridge mass, and iterative retrieval can help exactly when that preservation is maintained across rounds.**
