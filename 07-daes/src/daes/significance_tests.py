"""Paired bootstrap significance tests for iDNMR results.

Tests whether iDNMR significantly outperforms each baseline using
paired bootstrap resampling on per-question F1 scores.

Run: python src/daes/significance_tests.py
"""
import json, glob, os
import numpy as np
from collections import defaultdict

def load_per_question_f1(pattern, methods):
    """Load per-question F1 from sharded JSON files."""
    per_q = defaultdict(dict)  # {qid: {method: f1}}
    files = sorted(glob.glob(pattern))
    for f in files:
        d = json.load(open(f))
        for r in d.get("results", []):
            qid = r["id"]
            for m in methods:
                if m in r:
                    per_q[qid][m] = r[m]["f1"]
    return per_q


def paired_bootstrap(scores_a, scores_b, n_bootstrap=10000, seed=42):
    """Paired bootstrap test: is mean(a) > mean(b)?
    Returns: observed delta, p-value (one-sided), 95% CI.
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    assert len(b) == n

    observed_delta = a.mean() - b.mean()
    deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = a[idx].mean() - b[idx].mean()

    # One-sided p-value: P(delta <= 0 under bootstrap)
    p_value = (deltas <= 0).mean()
    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)

    return observed_delta, p_value, ci_lo, ci_hi


def main():
    results_dir = "/projects/prjs1800/msc-thesis/07-daes/results"

    idnmr_methods = ["baseline", "pool", "ipool", "idnmr", "idnmr_2round"]
    baseline_methods = ["spread", "aram", "ispread", "iaram"]

    datasets = ["musique", "hotpotqa", "2wikimultihopqa"]

    print("=" * 80)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTS (n=10000)")
    print("=" * 80)

    for dataset in datasets:
        print(f"\n{'='*40}")
        print(f"  {dataset.upper()}")
        print(f"{'='*40}")

        # Load iDNMR results
        idnmr_pattern = f"{results_dir}/idnmr/dream_{dataset}_idnmr_1k_s[0-4].json"
        per_q_idnmr = load_per_question_f1(idnmr_pattern, idnmr_methods)

        # Load baseline results
        bl_pattern = f"{results_dir}/baselines/dream_{dataset}_baselines_s*.json"
        per_q_bl = load_per_question_f1(bl_pattern, baseline_methods)

        # Merge: get questions that appear in both
        all_methods_scores = defaultdict(list)
        common_qids = sorted(set(per_q_idnmr.keys()))

        for qid in common_qids:
            for m in idnmr_methods:
                if m in per_q_idnmr[qid]:
                    all_methods_scores[m].append(per_q_idnmr[qid][m])

        # For baseline methods, match by index (same question order)
        bl_qids = sorted(set(per_q_bl.keys()))
        for qid in bl_qids:
            for m in baseline_methods:
                if m in per_q_bl[qid]:
                    all_methods_scores[m].append(per_q_bl[qid][m])

        # Test iDNMR against each other method
        idnmr_scores = all_methods_scores.get("idnmr", [])
        n_idnmr = len(idnmr_scores)

        print(f"\n  iDNMR n={n_idnmr}, mean F1={np.mean(idnmr_scores):.4f}")
        print(f"  {'Comparison':<25s} {'Delta':>8s} {'p-value':>10s} {'95% CI':>20s} {'Sig?':>6s}")
        print(f"  {'-'*71}")

        # Compare against methods from the same runner (paired on same questions)
        for m in ["baseline", "pool", "ipool", "idnmr_2round"]:
            other = all_methods_scores.get(m, [])
            if len(other) != n_idnmr:
                print(f"  {m:<25s} SKIPPED (n={len(other)} != {n_idnmr})")
                continue
            delta, pval, ci_lo, ci_hi = paired_bootstrap(idnmr_scores, other)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"  iDNMR vs {m:<14s} {delta:>+8.4f} {pval:>10.4f} [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}] {sig:>6s}")

        # Compare against baseline methods (different runner, match by question count)
        for m in baseline_methods:
            other = all_methods_scores.get(m, [])
            n_common = min(len(idnmr_scores), len(other))
            if n_common < 100:
                print(f"  {m:<25s} SKIPPED (n={n_common} too small)")
                continue
            delta, pval, ci_lo, ci_hi = paired_bootstrap(
                idnmr_scores[:n_common], other[:n_common]
            )
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"  iDNMR vs {m:<14s} {delta:>+8.4f} {pval:>10.4f} [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}] {sig:>6s}")

    print(f"\n{'='*80}")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
