"""Bootstrap Confidence Intervals and Significance Tests for RAG Experiments.

Day 6: Computes 95% CIs for all experiments and paired bootstrap significance
tests for key comparisons.

Usage:
    # Phase 1: CIs for Days 1-5 existing results
    python -u scripts/day6/bootstrap_analysis.py --phase existing \
        --output_dir /projects/prjs1800/analysis/day6

    # Phase 2: CIs for Day 6 results + significance tests
    python -u scripts/day6/bootstrap_analysis.py --phase day6 \
        --output_dir /projects/prjs1800/analysis/day6
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np


# ── Evaluation functions (same as FlashRAG) ─────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(pred, gold_list):
    norm_pred = normalize_answer(pred)
    return max(float(norm_pred == normalize_answer(g)) for g in gold_list)


def compute_f1(pred, gold_list):
    best = 0.0
    for gold in gold_list:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_tokens)
        rec = num_same / len(gold_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


# ── Experiment registry ─────────────────────────────────────────────────────

EXPERIMENTS_EXISTING = {
    # Day 1
    "day1_standard_rag_hotpotqa": {
        "path": "/projects/prjs1800/results/day1/hotpotqa_2026_02_06_13_47_standard_rag_qwen25_hotpotqa",
        "dataset": "hotpotqa",
        "method": "Standard RAG",
        "day": 1,
    },
    "day1_standard_rag_musique": {
        "path": "/projects/prjs1800/results/day1/musique_2026_02_06_14_08_standard_rag_qwen25_musique",
        "dataset": "musique",
        "method": "Standard RAG",
        "day": 1,
    },
    # Day 2
    "day2_reranker_hotpotqa": {
        "path": "/projects/prjs1800/results/day2/hotpotqa_2026_02_06_15_37_reranker_rag_qwen25_hotpotqa",
        "dataset": "hotpotqa",
        "method": "Reranker",
        "day": 2,
    },
    "day2_reranker_musique": {
        "path": "/projects/prjs1800/results/day2/musique_2026_02_06_15_37_reranker_rag_qwen25_musique",
        "dataset": "musique",
        "method": "Reranker",
        "day": 2,
    },
    # Day 5 - Reranker+CoT (short, max_tokens=32)
    "day5_reranker_cot_short_hotpotqa": {
        "path": "/projects/prjs1800/results/day5/hotpotqa_2026_02_07_12_43_reranker_cot_qwen25_hotpotqa",
        "dataset": "hotpotqa",
        "method": "Reranker+CoT",
        "day": 5,
    },
}

# Day 4 IRCoT - path needs to be found
EXPERIMENTS_DAY4_PATTERN = {
    "day4_ircot_hotpotqa": {
        "path_pattern": "/projects/prjs1800/results/day4/*ircot*hotpotqa*",
        "dataset": "hotpotqa",
        "method": "IRCoT",
        "day": 4,
    },
    "day4_ircot_musique": {
        "path_pattern": "/projects/prjs1800/results/day4/*ircot*musique*",
        "dataset": "musique",
        "method": "IRCoT",
        "day": 4,
    },
}


def find_day4_paths():
    """Find Day 4 IRCoT result directories using glob."""
    import glob
    found = {}
    for key, info in EXPERIMENTS_DAY4_PATTERN.items():
        matches = sorted(glob.glob(info["path_pattern"]))
        if matches:
            found[key] = {**info, "path": matches[-1]}  # Use latest
            del found[key]["path_pattern"]
            print(f"  Found {key}: {found[key]['path']}")
        else:
            print(f"  WARNING: No match for {key} pattern: {info['path_pattern']}")
    return found


def get_day6_experiments():
    """Build Day 6 experiment registry from result directories."""
    import glob
    base = "/projects/prjs1800/results/day6"
    experiments = {}

    patterns = {
        "day6_naive_hotpotqa": ("*naive*hotpotqa*", "hotpotqa", "Naive Gen"),
        "day6_naive_musique": ("*naive*musique*", "musique", "Naive Gen"),
        "day6_naive_2wiki": ("*naive*2wiki*", "2wikimultihopqa", "Naive Gen"),
        "day6_gold_hotpotqa": ("*gold*hotpotqa*", "hotpotqa", "Gold Context"),
        "day6_gold_musique": ("*gold*musique*", "musique", "Gold Context"),
        "day6_gold_2wiki": ("*gold*2wiki*", "2wikimultihopqa", "Gold Context"),
        "day6_standard_rag_2wiki": ("*standard*2wiki*", "2wikimultihopqa", "Standard RAG"),
        "day6_reranker_2wiki": ("*reranker*2wiki*", "2wikimultihopqa", "Reranker"),
    }

    for key, (pattern, dataset, method) in patterns.items():
        matches = sorted(glob.glob(os.path.join(base, pattern)))
        if matches:
            experiments[key] = {
                "path": matches[-1],
                "dataset": dataset,
                "method": method,
                "day": 6,
            }
            print(f"  Found {key}: {experiments[key]['path']}")
        else:
            print(f"  WARNING: No match for {key}")

    return experiments


# ── Score computation ───────────────────────────────────────────────────────

def load_scores(result_dir):
    """Load intermediate_data.json and compute per-item EM and F1 scores."""
    data_path = os.path.join(result_dir, "intermediate_data.json")
    if not os.path.exists(data_path):
        print(f"  WARNING: {data_path} not found")
        return None

    with open(data_path) as f:
        data = json.load(f)

    scores = []
    for item in data:
        gold = item.get("golden_answers", [])
        pred = item.get("output", {}).get("pred", "")
        if not gold:
            continue

        em = compute_em(pred, gold)
        f1 = compute_f1(pred, gold)

        score_item = {
            "id": item.get("id", ""),
            "em": em,
            "f1": f1,
            "question": item.get("question", ""),
        }

        # Add metadata for subgroup analysis
        meta = item.get("metadata", {})
        if meta:
            # HotpotQA: question type (bridge/comparison)
            if "type" in meta:
                score_item["question_type"] = meta["type"]
            # MuSiQue: number of hops
            if "question_decomposition" in meta:
                score_item["n_hops"] = len(meta["question_decomposition"])

        scores.append(score_item)

    return scores


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    n = len(values)
    values = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means.append(values[idx].mean())
    means = np.array(means)
    alpha = (1 - ci) / 2
    lower = np.percentile(means, 100 * alpha)
    upper = np.percentile(means, 100 * (1 - alpha))
    return {
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_width": float(upper - lower),
        "std": float(means.std()),
        "n": n,
    }


def bootstrap_subgroup(scores, group_key, n_bootstrap=1000):
    """Compute bootstrap CIs for subgroups."""
    groups = {}
    for s in scores:
        key = s.get(group_key)
        if key is None:
            continue
        groups.setdefault(key, []).append(s)

    results = {}
    for key, items in sorted(groups.items(), key=lambda x: str(x[0])):
        em_vals = [s["em"] for s in items]
        f1_vals = [s["f1"] for s in items]
        results[str(key)] = {
            "n": len(items),
            "em": bootstrap_ci(em_vals, n_bootstrap),
            "f1": bootstrap_ci(f1_vals, n_bootstrap),
        }
    return results


# ── Paired Bootstrap Significance Test ──────────────────────────────────────

def paired_bootstrap_test(scores_a, scores_b, metric="f1", n_bootstrap=1000, seed=42):
    """Paired bootstrap significance test.

    For each bootstrap sample, compute the metric for both methods on the
    same sample indices. p-value = fraction of times the difference crosses
    zero × 2 (two-sided test).

    Requires that scores_a and scores_b are aligned (same questions in same order).
    """
    rng = np.random.RandomState(seed)
    vals_a = np.array([s[metric] for s in scores_a])
    vals_b = np.array([s[metric] for s in scores_b])
    n = len(vals_a)
    assert len(vals_b) == n, f"Score arrays must be same length: {len(vals_a)} vs {len(vals_b)}"

    observed_diff = vals_a.mean() - vals_b.mean()
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diffs.append(vals_a[idx].mean() - vals_b[idx].mean())

    diffs = np.array(diffs)

    # Two-sided p-value: fraction of bootstrap diffs on the other side of zero
    if observed_diff >= 0:
        p_value = 2 * np.mean(diffs < 0)
    else:
        p_value = 2 * np.mean(diffs > 0)

    # Clamp to [0, 1]
    p_value = min(p_value, 1.0)

    return {
        "method_a_mean": float(vals_a.mean()),
        "method_b_mean": float(vals_b.mean()),
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
        "bootstrap_diff_mean": float(diffs.mean()),
        "bootstrap_diff_std": float(diffs.std()),
        "n": n,
    }


def align_scores(scores_a, scores_b):
    """Align two score lists by question ID, returning matched pairs."""
    id_to_a = {s["id"]: s for s in scores_a}
    id_to_b = {s["id"]: s for s in scores_b}
    common_ids = sorted(set(id_to_a.keys()) & set(id_to_b.keys()))
    aligned_a = [id_to_a[qid] for qid in common_ids]
    aligned_b = [id_to_b[qid] for qid in common_ids]
    return aligned_a, aligned_b, common_ids


# ── Significance test definitions ───────────────────────────────────────────

SIGNIFICANCE_TESTS = {
    "hotpotqa": [
        {
            "name": "Value of Retrieval (Naive vs Standard RAG)",
            "method_a": "day1_standard_rag_hotpotqa",
            "method_b": "day6_naive_hotpotqa",
        },
        {
            "name": "Value of Reranking (Standard RAG vs Reranker)",
            "method_a": "day2_reranker_hotpotqa",
            "method_b": "day1_standard_rag_hotpotqa",
        },
        {
            "name": "Iterative vs Single-shot (Reranker vs IRCoT)",
            "method_a": "day4_ircot_hotpotqa",
            "method_b": "day2_reranker_hotpotqa",
        },
        {
            "name": "Does CoT help? (Reranker vs Reranker+CoT)",
            "method_a": "day5_reranker_cot_short_hotpotqa",
            "method_b": "day2_reranker_hotpotqa",
        },
        {
            "name": "Room for Improvement (Reranker vs Gold Context)",
            "method_a": "day6_gold_hotpotqa",
            "method_b": "day2_reranker_hotpotqa",
        },
    ],
    "musique": [
        {
            "name": "Value of Retrieval (Naive vs Standard RAG)",
            "method_a": "day1_standard_rag_musique",
            "method_b": "day6_naive_musique",
        },
        {
            "name": "Value of Reranking (Standard RAG vs Reranker)",
            "method_a": "day2_reranker_musique",
            "method_b": "day1_standard_rag_musique",
        },
        {
            "name": "Room for Improvement (Reranker vs Gold Context)",
            "method_a": "day6_gold_musique",
            "method_b": "day2_reranker_musique",
        },
    ],
    "2wikimultihopqa": [
        {
            "name": "Value of Retrieval (Naive vs Standard RAG)",
            "method_a": "day6_standard_rag_2wiki",
            "method_b": "day6_naive_2wiki",
        },
        {
            "name": "Value of Reranking (Standard RAG vs Reranker)",
            "method_a": "day6_reranker_2wiki",
            "method_b": "day6_standard_rag_2wiki",
        },
        {
            "name": "Room for Improvement (Reranker vs Gold Context)",
            "method_a": "day6_gold_2wiki",
            "method_b": "day6_reranker_2wiki",
        },
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["existing", "day6"], required=True,
                        help="'existing' for Days 1-5, 'day6' for Day 6 + significance tests")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_bootstrap = args.n_bootstrap

    # Build experiment registry
    print("=== Building experiment registry ===")
    all_experiments = {}

    if args.phase == "existing":
        all_experiments.update(EXPERIMENTS_EXISTING)
        all_experiments.update(find_day4_paths())
    else:
        # Day 6: include everything for significance tests
        all_experiments.update(EXPERIMENTS_EXISTING)
        all_experiments.update(find_day4_paths())
        all_experiments.update(get_day6_experiments())

    # Load all scores
    print(f"\n=== Loading scores for {len(all_experiments)} experiments ===")
    all_scores = {}
    for key, info in all_experiments.items():
        print(f"\nLoading {key} ({info['method']} on {info['dataset']})...")
        scores = load_scores(info["path"])
        if scores:
            all_scores[key] = scores
            print(f"  Loaded {len(scores)} items")
        else:
            print(f"  SKIPPED (no data)")

    # Compute bootstrap CIs
    print(f"\n=== Computing Bootstrap CIs (n_bootstrap={n_bootstrap}) ===")
    t_start = time.time()

    bootstrap_results = {}
    for key, scores in all_scores.items():
        info = all_experiments[key]
        em_vals = [s["em"] for s in scores]
        f1_vals = [s["f1"] for s in scores]

        result = {
            "method": info["method"],
            "dataset": info["dataset"],
            "day": info["day"],
            "path": info["path"],
            "em": bootstrap_ci(em_vals, n_bootstrap),
            "f1": bootstrap_ci(f1_vals, n_bootstrap),
        }

        # Subgroup analysis
        if any("question_type" in s for s in scores):
            result["by_question_type"] = bootstrap_subgroup(scores, "question_type", n_bootstrap)
        if any("n_hops" in s for s in scores):
            result["by_n_hops"] = bootstrap_subgroup(scores, "n_hops", n_bootstrap)

        bootstrap_results[key] = result

        # Print summary
        em = result["em"]
        f1 = result["f1"]
        print(f"  {key}:")
        print(f"    EM: {100*em['mean']:.2f}% [{100*em['ci_lower']:.2f}, {100*em['ci_upper']:.2f}]")
        print(f"    F1: {100*f1['mean']:.2f}% [{100*f1['ci_lower']:.2f}, {100*f1['ci_upper']:.2f}]")

        # Print subgroup results
        if "by_question_type" in result:
            print(f"    By question type:")
            for qtype, sub in result["by_question_type"].items():
                print(f"      {qtype} (n={sub['n']}): F1={100*sub['f1']['mean']:.2f}% "
                      f"[{100*sub['f1']['ci_lower']:.2f}, {100*sub['f1']['ci_upper']:.2f}]")
        if "by_n_hops" in result:
            print(f"    By number of hops:")
            for nhops, sub in result["by_n_hops"].items():
                print(f"      {nhops}-hop (n={sub['n']}): F1={100*sub['f1']['mean']:.2f}% "
                      f"[{100*sub['f1']['ci_lower']:.2f}, {100*sub['f1']['ci_upper']:.2f}]")

    t_ci = time.time() - t_start
    print(f"\nBootstrap CIs computed in {t_ci:.1f}s")

    # Save bootstrap results
    if args.phase == "existing":
        out_path = os.path.join(args.output_dir, "bootstrap_results_days1to5.json")
    else:
        out_path = os.path.join(args.output_dir, "bootstrap_results_all.json")

    with open(out_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    print(f"Saved bootstrap results to {out_path}")

    # Significance tests (only in day6 phase)
    if args.phase == "day6":
        print(f"\n=== Paired Bootstrap Significance Tests ===")
        sig_results = {}

        for dataset, tests in SIGNIFICANCE_TESTS.items():
            print(f"\n--- {dataset} ---")
            sig_results[dataset] = []

            for test in tests:
                name = test["name"]
                key_a = test["method_a"]
                key_b = test["method_b"]

                if key_a not in all_scores or key_b not in all_scores:
                    print(f"  SKIP: {name} (missing {key_a} or {key_b})")
                    sig_results[dataset].append({
                        "name": name,
                        "method_a": key_a,
                        "method_b": key_b,
                        "status": "skipped",
                    })
                    continue

                # Align scores by question ID
                aligned_a, aligned_b, common_ids = align_scores(
                    all_scores[key_a], all_scores[key_b]
                )

                if len(common_ids) == 0:
                    print(f"  SKIP: {name} (no common questions)")
                    continue

                # Run test for both EM and F1
                em_test = paired_bootstrap_test(aligned_a, aligned_b, "em", n_bootstrap)
                f1_test = paired_bootstrap_test(aligned_a, aligned_b, "f1", n_bootstrap)

                result = {
                    "name": name,
                    "method_a": key_a,
                    "method_b": key_b,
                    "n_common": len(common_ids),
                    "em": em_test,
                    "f1": f1_test,
                }
                sig_results[dataset].append(result)

                # Print
                stars_em = "***" if em_test["significant_001"] else ("*" if em_test["significant_005"] else "ns")
                stars_f1 = "***" if f1_test["significant_001"] else ("*" if f1_test["significant_005"] else "ns")
                print(f"  {name} (n={len(common_ids)}):")
                print(f"    EM: {100*em_test['method_a_mean']:.2f} vs {100*em_test['method_b_mean']:.2f}, "
                      f"diff={100*em_test['observed_diff']:+.2f}, p={em_test['p_value']:.4f} {stars_em}")
                print(f"    F1: {100*f1_test['method_a_mean']:.2f} vs {100*f1_test['method_b_mean']:.2f}, "
                      f"diff={100*f1_test['observed_diff']:+.2f}, p={f1_test['p_value']:.4f} {stars_f1}")

        sig_path = os.path.join(args.output_dir, "significance_tests.json")
        with open(sig_path, "w") as f:
            json.dump(sig_results, f, indent=2)
        print(f"\nSaved significance tests to {sig_path}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
