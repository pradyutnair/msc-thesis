"""
CPU-only analysis script: 
1. Statistical significance tests (LLaDA + Dream)
2. Per-hop analysis (MuSiQue)
3. Answer length distributions
4. Error categorization
"""

import json
import os
import random
import sys
from collections import defaultdict

RESULTS_BASE = "/projects/prjs1800/msc-thesis/07-daes/results"
JUDGE_DIR = f"{RESULTS_BASE}/llm_judge"
MIXED_DIR = f"{RESULTS_BASE}/mixed"
MUSIQUE_ORIG = "/projects/prjs1800/datasets/musique/musique_ans_v1.0_dev.jsonl"
OUTPUT_DIR = f"{RESULTS_BASE}/cpu_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Paired Bootstrap ────────────────────────────────────────────────────────

def paired_bootstrap(scores_a, scores_b, n_resamples=10000, seed=42):
    """Returns p-value for H0: mean(a) <= mean(b), i.e. p for a > b."""
    rng = random.Random(seed)
    n = len(scores_a)
    observed_diff = sum(scores_a) / n - sum(scores_b) / n
    count = 0
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        diff = sum(scores_a[i] for i in idx) / n - sum(scores_b[i] for i in idx) / n
        if diff <= 0:
            count += 1
    p_value = count / n_resamples
    return observed_diff, p_value


def load_judge(dataset, model="llada"):
    """Load judge file, return list of {id, methods: {method: {judge, extracted_f1, ...}}}"""
    if model == "llada":
        path = f"{JUDGE_DIR}/llm_judge_{dataset}.json"
    else:
        path = f"{JUDGE_DIR}/llm_judge_dream_{dataset}.json"
    return json.load(open(path))


# ─── Task 1: Significance Tests ──────────────────────────────────────────────

def run_significance_tests():
    print("\n" + "=" * 70)
    print("TASK 1: STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
    comparisons = [
        ("idnmr", "ispread"),
        ("idnmr", "iaram"),
        ("pool",  "ispread"),
        ("pool",  "iaram"),
    ]

    results = {}

    for model in ["llada", "dream"]:
        results[model] = {}
        print(f"\n{'─'*50}")
        print(f"Model: {model.upper()}")
        print(f"{'─'*50}")

        for dataset in datasets:
            try:
                data = load_judge(dataset, model)
            except FileNotFoundError:
                print(f"  [{dataset}] MISSING judge file")
                continue

            results[model][dataset] = {}
            print(f"\n  {dataset} (N={len(data)})")

            for method_a, method_b in comparisons:
                # Use judge correct (binary) as score
                scores_a, scores_b = [], []
                f1_a, f1_b = [], []
                for item in data:
                    m = item["methods"]
                    if method_a not in m or method_b not in m:
                        continue
                    scores_a.append(1 if m[method_a].get("judge") == "correct" else 0)
                    scores_b.append(1 if m[method_b].get("judge") == "correct" else 0)
                    f1_a.append(m[method_a].get("extracted_f1", 0) or 0)
                    f1_b.append(m[method_b].get("extracted_f1", 0) or 0)

                if not scores_a:
                    continue

                diff_judge, p_judge = paired_bootstrap(scores_a, scores_b)
                diff_f1, p_f1 = paired_bootstrap(f1_a, f1_b)

                sig_judge = "***" if p_judge < 0.001 else ("**" if p_judge < 0.01 else ("*" if p_judge < 0.05 else "ns"))
                sig_f1 = "***" if p_f1 < 0.001 else ("**" if p_f1 < 0.01 else ("*" if p_f1 < 0.05 else "ns"))

                mean_a_j = sum(scores_a) / len(scores_a) * 100
                mean_b_j = sum(scores_b) / len(scores_b) * 100

                print(f"    {method_a:8s} vs {method_b:8s} | "
                      f"Judge: {mean_a_j:.1f}% vs {mean_b_j:.1f}% (Δ={diff_judge*100:+.1f}pp, p={p_judge:.4f} {sig_judge}) | "
                      f"ExtF1: Δ={diff_f1*100:+.2f}pp, p={p_f1:.4f} {sig_f1}")

                results[model][dataset][f"{method_a}_vs_{method_b}"] = {
                    "n": len(scores_a),
                    "judge_a": mean_a_j,
                    "judge_b": mean_b_j,
                    "diff_judge_pp": diff_judge * 100,
                    "p_judge": p_judge,
                    "sig_judge": sig_judge,
                    "diff_f1_pp": diff_f1 * 100,
                    "p_f1": p_f1,
                    "sig_f1": sig_f1,
                }

    out_path = f"{OUTPUT_DIR}/significance_tests.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n  Saved → {out_path}")
    return results


# ─── Task 2: Per-Hop Analysis ────────────────────────────────────────────────

def run_per_hop_analysis():
    print("\n" + "=" * 70)
    print("TASK 2: PER-HOP ANALYSIS (MuSiQue)")
    print("=" * 70)

    # Build hop count map from original MuSiQue
    hop_map = {}
    for line in open(MUSIQUE_ORIG):
        item = json.loads(line)
        qid = item["id"]
        # id looks like "2hop__460946_294723"
        n_hops = int(qid.split("hop__")[0]) if "hop__" in qid else None
        # Our question file uses dev_0, dev_1 etc. Need to match by question text
        hop_map[item["question"].strip()] = n_hops

    # Also build by answer for robustness
    hop_map_by_answer = {}
    for line in open(MUSIQUE_ORIG):
        item = json.loads(line)
        qid = item["id"]
        n_hops = int(qid.split("hop__")[0]) if "hop__" in qid else None
        hop_map_by_answer[(item["question"].strip(), item["answer"].strip())] = n_hops

    results = {}

    for model in ["llada", "dream"]:
        try:
            data = load_judge("musique", model)
        except FileNotFoundError:
            print(f"  {model} MISSING judge file")
            continue

        # Assign hop counts
        hop_counts = []
        for item in data:
            q = item["question"].strip()
            gold = item["gold"].strip()
            nh = hop_map.get(q) or hop_map_by_answer.get((q, gold))
            hop_counts.append(nh)

        assigned = sum(1 for h in hop_counts if h is not None)
        print(f"\n  {model.upper()} — hop count assigned: {assigned}/{len(data)}")

        hop_dist = defaultdict(int)
        for h in hop_counts:
            if h:
                hop_dist[h] += 1
        print(f"  Distribution: {dict(sorted(hop_dist.items()))}")

        methods_of_interest = ["baseline", "aram", "pool", "ispread", "iaram", "idnmr"]
        results[model] = defaultdict(lambda: defaultdict(list))

        for item, n_hops in zip(data, hop_counts):
            if n_hops is None:
                continue
            for method in methods_of_interest:
                m = item["methods"].get(method)
                if not m:
                    continue
                judge_correct = 1 if m.get("judge") == "correct" else 0
                ext_f1 = m.get("extracted_f1", 0) or 0
                results[model][n_hops][method].append((judge_correct, ext_f1))

        print(f"\n  {'Hop':<6} {'Method':<12} {'N':>5} {'Judge%':>8} {'ExtF1':>8}")
        print(f"  {'---':<6} {'------':<12} {'---':>5} {'------':>8} {'-----':>8}")
        for n_hops in sorted(set(h for h in hop_counts if h)):
            for method in methods_of_interest:
                vals = results[model][n_hops][method]
                if not vals:
                    continue
                j = sum(v[0] for v in vals) / len(vals) * 100
                f = sum(v[1] for v in vals) / len(vals)
                print(f"  {n_hops:<6} {method:<12} {len(vals):>5} {j:>7.1f}% {f:>8.4f}")

    # Convert defaultdicts for JSON serialization
    serializable = {}
    for model in results:
        serializable[model] = {}
        for hops in results[model]:
            serializable[model][str(hops)] = {}
            for method in results[model][hops]:
                vals = results[model][hops][method]
                serializable[model][str(hops)][method] = {
                    "n": len(vals),
                    "judge_pct": sum(v[0] for v in vals) / len(vals) * 100,
                    "ext_f1": sum(v[1] for v in vals) / len(vals),
                }

    out_path = f"{OUTPUT_DIR}/per_hop_analysis.json"
    json.dump(serializable, open(out_path, "w"), indent=2)
    print(f"\n  Saved → {out_path}")
    return serializable


# ─── Task 3: Answer Length Distributions ─────────────────────────────────────

def run_answer_length_analysis():
    print("\n" + "=" * 70)
    print("TASK 3: ANSWER LENGTH DISTRIBUTIONS")
    print("=" * 70)

    datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
    methods_of_interest = ["baseline", "aram", "spread", "pool", "iaram", "ispread", "idnmr"]

    results = {}

    for model in ["llada", "dream"]:
        results[model] = {}
        print(f"\n  {model.upper()}")

        for dataset in datasets:
            try:
                data = load_judge(dataset, model)
            except FileNotFoundError:
                continue

            results[model][dataset] = {}

            for method in methods_of_interest:
                lengths = []
                for item in data:
                    m = item["methods"].get(method)
                    if m and m.get("answer"):
                        lengths.append(len(m["answer"].split()))
                if not lengths:
                    continue

                avg = sum(lengths) / len(lengths)
                sorted_l = sorted(lengths)
                p50 = sorted_l[len(sorted_l) // 2]
                p90 = sorted_l[int(len(sorted_l) * 0.9)]

                results[model][dataset][method] = {
                    "n": len(lengths),
                    "mean_words": round(avg, 2),
                    "median_words": p50,
                    "p90_words": p90,
                    "pct_over_10_words": sum(1 for l in lengths if l > 10) / len(lengths) * 100,
                }

            print(f"\n    {dataset}")
            print(f"    {'Method':<14} {'Mean':>7} {'Median':>8} {'P90':>6} {'>10w%':>7}")
            print(f"    {'------':<14} {'----':>7} {'------':>8} {'---':>6} {'----':>7}")
            for method in methods_of_interest:
                d = results[model][dataset].get(method)
                if d:
                    print(f"    {method:<14} {d['mean_words']:>7.1f} {d['median_words']:>8} {d['p90_words']:>6} {d['pct_over_10_words']:>6.1f}%")

    out_path = f"{OUTPUT_DIR}/answer_length_distributions.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n  Saved → {out_path}")
    return results


# ─── Task 4: Error Categorization ────────────────────────────────────────────

def run_error_categorization():
    print("\n" + "=" * 70)
    print("TASK 4: ERROR CATEGORIZATION")
    print("=" * 70)

    datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
    method_pairs = [
        ("pool", "aram"),      # DNMR vs guidance baseline
        ("idnmr", "iaram"),    # iterative DNMR vs iterative guidance
    ]

    results = {}

    for model in ["llada", "dream"]:
        results[model] = {}
        print(f"\n  {model.upper()}")

        for dataset in datasets:
            try:
                data = load_judge(dataset, model)
            except FileNotFoundError:
                continue

            results[model][dataset] = {}

            for method_a, method_b in method_pairs:
                cats = {
                    "both_correct": 0,
                    "a_only_correct": 0,
                    "b_only_correct": 0,
                    "both_wrong": 0,
                    # For both_wrong: subcategorize by contain
                    "both_wrong_a_retrieval_hit": 0,   # pool retrieved gold but answered wrong
                    "both_wrong_b_retrieval_hit": 0,
                    "both_wrong_neither_retrieval": 0,
                }

                n = 0
                for item in data:
                    m = item["methods"]
                    if method_a not in m or method_b not in m:
                        continue
                    n += 1
                    j_a = m[method_a].get("judge") == "correct"
                    j_b = m[method_b].get("judge") == "correct"
                    cont_a = m[method_a].get("extracted_contain", 0) or 0

                    if j_a and j_b:
                        cats["both_correct"] += 1
                    elif j_a and not j_b:
                        cats["a_only_correct"] += 1
                    elif not j_a and j_b:
                        cats["b_only_correct"] += 1
                    else:
                        cats["both_wrong"] += 1
                        if cont_a > 0:
                            cats["both_wrong_a_retrieval_hit"] += 1
                        elif m[method_b].get("extracted_contain", 0) or 0 > 0:
                            cats["both_wrong_b_retrieval_hit"] += 1
                        else:
                            cats["both_wrong_neither_retrieval"] += 1

                pct = {k: v / n * 100 for k, v in cats.items()}
                results[model][dataset][f"{method_a}_vs_{method_b}"] = {
                    "n": n,
                    "counts": cats,
                    "pcts": pct,
                }

                print(f"\n    {dataset} | {method_a} vs {method_b} (N={n})")
                print(f"      Both correct:       {cats['both_correct']:>5} ({pct['both_correct']:.1f}%)")
                print(f"      {method_a} only correct:  {cats['a_only_correct']:>5} ({pct['a_only_correct']:.1f}%)")
                print(f"      {method_b} only correct:  {cats['b_only_correct']:>5} ({pct['b_only_correct']:.1f}%)")
                print(f"      Both wrong:         {cats['both_wrong']:>5} ({pct['both_wrong']:.1f}%)")
                print(f"        └ {method_a} retrieved gold but wrong:  {cats['both_wrong_a_retrieval_hit']:>5} ({pct['both_wrong_a_retrieval_hit']:.1f}%)")
                print(f"        └ Neither retrieved gold:              {cats['both_wrong_neither_retrieval']:>5} ({pct['both_wrong_neither_retrieval']:.1f}%)")

    out_path = f"{OUTPUT_DIR}/error_categorization.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n  Saved → {out_path}")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sig_results = run_significance_tests()
    hop_results = run_per_hop_analysis()
    len_results = run_answer_length_analysis()
    err_results = run_error_categorization()
    print("\n\n✓ All CPU analyses complete. Results in:", OUTPUT_DIR)
