"""Day 4 Analysis: IRCoT per-round retrieval quality and error categorization.

Computes:
1. Per-round retrieval recall against GT supporting docs
2. Accumulated retrieval recall progression (how recall grows with iterations)
3. For MuSiQue: per-hop retrieval improvement across rounds
4. Query reformulation analysis (what queries does IRCoT generate?)
5. Error categorization: what does IRCoT fix vs what remains broken?
6. Comparison with Day 1 (Standard RAG) and Day 2 (+ Reranker) baselines
7. Summary tables for the master comparison

Usage:
    python analyze_ircot_retrieval.py \
        --ircot_hotpotqa /path/to/ircot_hotpotqa/intermediate_data.json \
        --ircot_musique /path/to/ircot_musique/intermediate_data.json \
        --baseline_hotpotqa /path/to/day1_hotpotqa/intermediate_data.json \
        --baseline_musique /path/to/day1_musique/intermediate_data.json \
        --output_dir /projects/prjs1800/msc-thesis/analysis/outputs
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict


def extract_title(doc_content):
    """Extract title from FlashRAG doc format."""
    first_line = doc_content.split("\n")[0].strip()
    m = re.match(r'^"(.+)"$', first_line)
    if m:
        return m.group(1)
    return first_line


def normalize_title(title):
    return title.lower().strip().replace("_", " ")


def get_retrieved_titles(docs):
    """Get normalized titles from a list of FlashRAG doc objects."""
    titles = set()
    for doc in docs:
        title = extract_title(doc["contents"])
        titles.add(normalize_title(title))
    return titles


def get_gt_titles_hotpotqa(item):
    """Get GT supporting doc titles for HotpotQA."""
    meta = item["metadata"]
    return set(normalize_title(t) for t in meta["supporting_facts"]["title"])


def get_gt_titles_musique(item):
    """Get GT supporting doc titles for MuSiQue, per hop."""
    meta = item["metadata"]
    decomp = meta["question_decomposition"]
    hop_titles = []
    all_titles = set()
    for hop in decomp:
        sp = hop.get("support_paragraph", {})
        title = sp.get("title", "")
        if title:
            nt = normalize_title(title)
            hop_titles.append(nt)
            all_titles.add(nt)
        else:
            hop_titles.append(None)
    return all_titles, hop_titles


def compute_recall(gt_titles, retrieved_titles):
    """Compute retrieval recall."""
    if not gt_titles:
        return 1.0
    found = gt_titles & retrieved_titles
    return len(found) / len(gt_titles)


def analyze_ircot_per_round(data, dataset_type="hotpotqa"):
    """Analyze IRCoT retrieval quality per round."""
    print(f"\n{'='*70}")
    print(f"IRCOT PER-ROUND RETRIEVAL ANALYSIS ({dataset_type.upper()})")
    print(f"{'='*70}")
    print(f"Total examples: {len(data)}")

    # Collect per-round recall scores
    max_rounds = 0
    per_round_recalls = defaultdict(list)  # round -> list of recall values
    accumulated_recalls = defaultdict(list)  # round -> accumulated recall
    iteration_counts = Counter()
    total_docs_accumulated = []

    for item in data:
        output = item["output"]
        n_iter = output.get("n_iterations", 0)
        iteration_counts[n_iter] += 1
        total_docs_accumulated.append(output.get("total_accumulated_docs", 0))

        # GT titles
        if dataset_type == "hotpotqa":
            gt_titles = get_gt_titles_hotpotqa(item)
        else:
            gt_titles, _ = get_gt_titles_musique(item)

        # Per-round retrieval recall
        per_round_doc_ids = output.get("per_round_doc_ids", {})
        all_retrieval_result = output.get("retrieval_result", [])
        retrieved_titles_all = get_retrieved_titles(all_retrieval_result)

        # Build id-to-doc map from final retrieval result
        id2doc_map = {}
        for doc in all_retrieval_result:
            id2doc_map[doc["id"]] = doc

        # Track accumulated doc IDs across rounds
        accumulated_ids = set()
        for round_num in sorted(int(k) for k in per_round_doc_ids.keys()):
            round_ids = per_round_doc_ids[str(round_num)]

            # Per-round: only docs from this specific round
            round_titles = set()
            for did in round_ids:
                if did in id2doc_map:
                    title = extract_title(id2doc_map[did]["contents"])
                    round_titles.add(normalize_title(title))
            per_round_recalls[round_num].append(compute_recall(gt_titles, round_titles))

            # Accumulated: all docs up to and including this round
            accumulated_ids.update(round_ids)
            acc_titles = set()
            for did in accumulated_ids:
                if did in id2doc_map:
                    title = extract_title(id2doc_map[did]["contents"])
                    acc_titles.add(normalize_title(title))
            accumulated_recalls[round_num].append(compute_recall(gt_titles, acc_titles))
            max_rounds = max(max_rounds, round_num)

        # Also store final recall from all accumulated docs
        final_recall = compute_recall(gt_titles, retrieved_titles_all)
        accumulated_recalls["final"].append(final_recall)

    # Iteration distribution
    print(f"\nIteration distribution:")
    for n_iter in sorted(iteration_counts.keys()):
        count = iteration_counts[n_iter]
        print(f"  {n_iter} iterations: {count} ({count/len(data)*100:.1f}%)")

    avg_docs = sum(total_docs_accumulated) / len(total_docs_accumulated)
    print(f"\nAvg accumulated docs per item: {avg_docs:.1f}")

    # Per-round recall progression
    print(f"\nPer-round retrieval recall (only docs from that specific round):")
    per_round_avg = {}
    for round_num in sorted(k for k in per_round_recalls.keys()):
        recalls = per_round_recalls[round_num]
        avg = sum(recalls) / len(recalls)
        per_round_avg[round_num] = avg
        print(f"  Round {round_num} (n={len(recalls)}): avg_recall={avg:.4f} ({avg*100:.1f}%)")

    print(f"\nAccumulated retrieval recall (all docs up to and including round):")
    accumulated_avg = {}
    for round_num in sorted(k for k in accumulated_recalls.keys() if k != "final"):
        recalls = accumulated_recalls[round_num]
        avg = sum(recalls) / len(recalls)
        accumulated_avg[round_num] = avg
        full = sum(1 for r in recalls if r == 1.0)
        print(f"  Through round {round_num} (n={len(recalls)}): avg_recall={avg:.4f} ({avg*100:.1f}%), full_recall={full}/{len(recalls)} ({full/len(recalls)*100:.1f}%)")

    # Final recall
    recalls = accumulated_recalls["final"]
    avg_recall = sum(recalls) / len(recalls)
    full_recall = sum(1 for r in recalls if r == 1.0)
    zero_recall = sum(1 for r in recalls if r == 0.0)
    print(f"\nFinal accumulated retrieval recall:")
    print(f"  Average: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"  Full recall (all GT found): {full_recall}/{len(data)} ({full_recall/len(data)*100:.1f}%)")
    print(f"  Zero recall (none found): {zero_recall}/{len(data)} ({zero_recall/len(data)*100:.1f}%)")

    return {
        "avg_recall": avg_recall,
        "full_recall_pct": full_recall / len(data),
        "zero_recall_pct": zero_recall / len(data),
        "iteration_dist": dict(iteration_counts),
        "avg_accumulated_docs": avg_docs,
        "per_round_recall": per_round_avg,
        "accumulated_recall": accumulated_avg,
    }


def analyze_ircot_hotpotqa(data):
    """Full HotpotQA analysis for IRCoT."""
    print(f"\n{'='*70}")
    print(f"HOTPOTQA IRCOT ANALYSIS")
    print(f"{'='*70}")

    total = len(data)
    categories = Counter()
    type_f1 = defaultdict(list)
    recall_scores = []

    for item in data:
        output = item["output"]
        meta = item["metadata"]
        gt_titles = get_gt_titles_hotpotqa(item)
        q_type = meta.get("type", "unknown")

        retrieved_titles = get_retrieved_titles(output.get("retrieval_result", []))
        recall = compute_recall(gt_titles, retrieved_titles)
        recall_scores.append(recall)

        em = output["metric_score"]["em"]
        f1 = output["metric_score"]["f1"]
        type_f1[q_type].append(f1)

        # Categorize
        if em == 1.0:
            categories["correct"] += 1
        elif recall == 1.0:
            categories["reasoning_failure"] += 1
        elif recall > 0:
            categories["partial_retrieval"] += 1
        else:
            categories["total_retrieval_miss"] += 1

    avg_recall = sum(recall_scores) / len(recall_scores)
    full_recall = sum(1 for r in recall_scores if r == 1.0)
    zero_recall = sum(1 for r in recall_scores if r == 0.0)

    print(f"\nTotal examples: {total}")
    print(f"\nRetrieval Recall (accumulated):")
    print(f"  Average: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"  Full recall: {full_recall}/{total} ({full_recall/total*100:.1f}%)")
    print(f"  Zero recall: {zero_recall}/{total} ({zero_recall/total*100:.1f}%)")

    print(f"\nF1 by question type:")
    for qtype in sorted(type_f1.keys()):
        scores = type_f1[qtype]
        avg = sum(scores) / len(scores)
        print(f"  {qtype} (n={len(scores)}): avg_F1={avg:.4f}")

    print(f"\nError categorization:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")

    # Answer quality by recall bucket
    print(f"\nAnswer F1 by retrieval recall bucket:")
    buckets = {"recall=0": [], "0<recall<1": [], "recall=1": []}
    for item, recall in zip(data, recall_scores):
        f1 = item["output"]["metric_score"]["f1"]
        if recall == 0:
            buckets["recall=0"].append(f1)
        elif recall < 1.0:
            buckets["0<recall<1"].append(f1)
        else:
            buckets["recall=1"].append(f1)
    for bucket, f1s in buckets.items():
        if f1s:
            print(f"  {bucket}: n={len(f1s)}, avg_F1={sum(f1s)/len(f1s):.4f}")

    return {
        "avg_recall": avg_recall,
        "full_recall_pct": full_recall / total,
        "zero_recall_pct": zero_recall / total,
        "categories": dict(categories),
        "type_f1": {k: sum(v)/len(v) for k, v in type_f1.items()},
    }


def analyze_ircot_musique(data):
    """Full MuSiQue analysis with per-hop retrieval tracking."""
    print(f"\n{'='*70}")
    print(f"MUSIQUE IRCOT ANALYSIS")
    print(f"{'='*70}")

    total = len(data)
    categories = Counter()
    recall_scores = []
    hop_counts = Counter()

    # Per-hop tracking
    hop_found = defaultdict(int)
    hop_total = defaultdict(int)

    for item in data:
        output = item["output"]
        meta = item["metadata"]
        all_gt, hop_titles = get_gt_titles_musique(item)
        n_hops = len(hop_titles)
        hop_counts[n_hops] += 1

        retrieved_titles = get_retrieved_titles(output.get("retrieval_result", []))
        recall = compute_recall(all_gt, retrieved_titles)
        recall_scores.append(recall)

        # Per-hop tracking
        for hop_idx, ht in enumerate(hop_titles):
            if ht is None:
                continue
            hop_num = hop_idx + 1
            hop_total[hop_num] += 1
            if ht in retrieved_titles:
                hop_found[hop_num] += 1

        # Categorize
        em = output["metric_score"]["em"]
        if em == 1.0:
            categories["correct"] += 1
        elif recall == 1.0:
            categories["reasoning_failure"] += 1
        elif recall > 0:
            categories["partial_retrieval"] += 1
        else:
            categories["total_retrieval_miss"] += 1

    avg_recall = sum(recall_scores) / len(recall_scores)
    full_recall = sum(1 for r in recall_scores if r == 1.0)
    zero_recall = sum(1 for r in recall_scores if r == 0.0)

    print(f"\nTotal examples: {total}")
    print(f"\nHop distribution:")
    for h in sorted(hop_counts.keys()):
        print(f"  {h}-hop: {hop_counts[h]} ({hop_counts[h]/total*100:.1f}%)")

    print(f"\nRetrieval Recall (accumulated):")
    print(f"  Average: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"  Full recall: {full_recall}/{total} ({full_recall/total*100:.1f}%)")
    print(f"  Zero recall: {zero_recall}/{total} ({zero_recall/total*100:.1f}%)")

    print(f"\nPer-hop retrieval recall:")
    hop_recall_dict = {}
    for hop_num in sorted(hop_total.keys()):
        r = hop_found[hop_num] / hop_total[hop_num]
        hop_recall_dict[hop_num] = r
        print(f"  Hop {hop_num}: {hop_found[hop_num]}/{hop_total[hop_num]} ({r*100:.1f}%)")

    print(f"\nError categorization:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")

    # Answer quality by recall bucket
    print(f"\nAnswer F1 by retrieval recall bucket:")
    buckets = {"recall=0": [], "0<recall<0.5": [], "0.5<=recall<1": [], "recall=1": []}
    for item, recall in zip(data, recall_scores):
        f1 = item["output"]["metric_score"]["f1"]
        if recall == 0:
            buckets["recall=0"].append(f1)
        elif recall < 0.5:
            buckets["0<recall<0.5"].append(f1)
        elif recall < 1.0:
            buckets["0.5<=recall<1"].append(f1)
        else:
            buckets["recall=1"].append(f1)
    for bucket, f1s in buckets.items():
        if f1s:
            print(f"  {bucket}: n={len(f1s)}, avg_F1={sum(f1s)/len(f1s):.4f}")

    return {
        "avg_recall": avg_recall,
        "full_recall_pct": full_recall / total,
        "zero_recall_pct": zero_recall / total,
        "categories": dict(categories),
        "hop_recall": hop_recall_dict,
    }


def analyze_query_reformulations(data, dataset_type="hotpotqa"):
    """Analyze the quality of IRCoT's query reformulations."""
    print(f"\n{'='*70}")
    print(f"QUERY REFORMULATION ANALYSIS ({dataset_type.upper()})")
    print(f"{'='*70}")

    # Collect per-round thoughts
    round_thoughts = defaultdict(list)
    answer_found_at = Counter()

    for item in data:
        output = item["output"]
        thoughts = output.get("per_round_thoughts", {})
        n_iter = output.get("n_iterations", 0)

        for round_str, thought in thoughts.items():
            round_num = int(round_str)
            round_thoughts[round_num].append(thought)

            if "So the answer is:" in thought:
                answer_found_at[round_num] += 1

    print(f"\nThoughts per round:")
    for round_num in sorted(round_thoughts.keys()):
        thoughts = round_thoughts[round_num]
        avg_len = sum(len(t) for t in thoughts) / len(thoughts) if thoughts else 0
        n_with_answer = sum(1 for t in thoughts if "So the answer is:" in t)
        print(f"  Round {round_num}: {len(thoughts)} thoughts, avg_len={avg_len:.0f} chars, "
              f"{n_with_answer} answers ({n_with_answer/len(thoughts)*100:.1f}%)")

    # Show example thoughts per round
    print(f"\n--- Example thoughts per round (first 3) ---")
    for round_num in sorted(round_thoughts.keys())[:4]:
        print(f"\n  Round {round_num}:")
        for thought in round_thoughts[round_num][:3]:
            print(f"    '{thought[:120]}...'" if len(thought) > 120 else f"    '{thought}'")


def compare_with_baselines(ircot_data, baseline_data, dataset_type="hotpotqa"):
    """Compare IRCoT results with baseline (Day 1/Day 2)."""
    print(f"\n{'='*70}")
    print(f"IRCOT vs BASELINE COMPARISON ({dataset_type.upper()})")
    print(f"{'='*70}")

    # Match items by question (they should be in same order, but verify)
    baseline_by_q = {item["question"]: item for item in baseline_data}

    improved = 0
    degraded = 0
    unchanged = 0
    improvements = []
    degradations = []

    for item in ircot_data:
        q = item["question"]
        if q not in baseline_by_q:
            continue
        baseline_item = baseline_by_q[q]

        ircot_f1 = item["output"]["metric_score"]["f1"]
        baseline_f1 = baseline_item["output"]["metric_score"]["f1"]

        delta = ircot_f1 - baseline_f1
        if delta > 0.01:
            improved += 1
            improvements.append((q, baseline_f1, ircot_f1, delta))
        elif delta < -0.01:
            degraded += 1
            degradations.append((q, baseline_f1, ircot_f1, delta))
        else:
            unchanged += 1

    total = improved + degraded + unchanged
    print(f"\nItem-level comparison (n={total}):")
    print(f"  Improved: {improved} ({improved/total*100:.1f}%)")
    print(f"  Degraded: {degraded} ({degraded/total*100:.1f}%)")
    print(f"  Unchanged: {unchanged} ({unchanged/total*100:.1f}%)")
    print(f"  Net: +{improved - degraded} items")

    if improvements:
        avg_improvement = sum(d for _, _, _, d in improvements) / len(improvements)
        print(f"\n  Avg improvement magnitude: +{avg_improvement:.4f} F1")
    if degradations:
        avg_degradation = sum(abs(d) for _, _, _, d in degradations) / len(degradations)
        print(f"  Avg degradation magnitude: -{avg_degradation:.4f} F1")

    # Show top improvements and degradations
    improvements.sort(key=lambda x: x[3], reverse=True)
    degradations.sort(key=lambda x: x[3])

    print(f"\n--- Top 5 improvements ---")
    for q, bf1, if1, d in improvements[:5]:
        print(f"  F1: {bf1:.3f} -> {if1:.3f} (+{d:.3f}): {q[:80]}")

    print(f"\n--- Top 5 degradations ---")
    for q, bf1, if1, d in degradations[:5]:
        print(f"  F1: {bf1:.3f} -> {if1:.3f} ({d:.3f}): {q[:80]}")


def generate_summary(results, output_dir):
    """Generate the Day 4 summary markdown."""
    os.makedirs(output_dir, exist_ok=True)

    summary = "# Day 4: IRCoT + FLARE — Iterative Retrieval Analysis\n\n"
    summary += "## Summary\n\n"
    summary += json.dumps(results, indent=2, default=str)
    summary += "\n"

    with open(os.path.join(output_dir, "day4_summary.md"), "w") as f:
        f.write(summary)
    print(f"\nSummary written to {output_dir}/day4_summary.md")


def main():
    parser = argparse.ArgumentParser(description="Day 4 IRCoT analysis")
    parser.add_argument("--ircot_hotpotqa", type=str, help="IRCoT HotpotQA results")
    parser.add_argument("--ircot_musique", type=str, help="IRCoT MuSiQue results")
    parser.add_argument("--flare_hotpotqa", type=str, help="FLARE HotpotQA results (optional)")
    parser.add_argument("--flare_musique", type=str, help="FLARE MuSiQue results (optional)")
    parser.add_argument("--baseline_hotpotqa", type=str, help="Day 1 baseline HotpotQA")
    parser.add_argument("--baseline_musique", type=str, help="Day 1 baseline MuSiQue")
    parser.add_argument("--output_dir", type=str,
                        default="/projects/prjs1800/msc-thesis/analysis/outputs")
    args = parser.parse_args()

    results = {}

    # ── IRCoT HotpotQA ──────────────────────────────────────────────────
    if args.ircot_hotpotqa:
        print(f"\nLoading IRCoT HotpotQA from {args.ircot_hotpotqa}")
        with open(args.ircot_hotpotqa) as f:
            ircot_hqa = json.load(f)

        results["ircot_hotpotqa"] = {}
        results["ircot_hotpotqa"]["per_round"] = analyze_ircot_per_round(ircot_hqa, "hotpotqa")
        results["ircot_hotpotqa"]["full"] = analyze_ircot_hotpotqa(ircot_hqa)
        analyze_query_reformulations(ircot_hqa, "hotpotqa")

        if args.baseline_hotpotqa:
            with open(args.baseline_hotpotqa) as f:
                baseline_hqa = json.load(f)
            compare_with_baselines(ircot_hqa, baseline_hqa, "hotpotqa")

    # ── IRCoT MuSiQue ──────────────────────────────────────────────────
    if args.ircot_musique:
        print(f"\nLoading IRCoT MuSiQue from {args.ircot_musique}")
        with open(args.ircot_musique) as f:
            ircot_msq = json.load(f)

        results["ircot_musique"] = {}
        results["ircot_musique"]["per_round"] = analyze_ircot_per_round(ircot_msq, "musique")
        results["ircot_musique"]["full"] = analyze_ircot_musique(ircot_msq)
        analyze_query_reformulations(ircot_msq, "musique")

        if args.baseline_musique:
            with open(args.baseline_musique) as f:
                baseline_msq = json.load(f)
            compare_with_baselines(ircot_msq, baseline_msq, "musique")

    # ── FLARE analysis (if available) ──────────────────────────────────
    if args.flare_hotpotqa:
        print(f"\nLoading FLARE HotpotQA from {args.flare_hotpotqa}")
        with open(args.flare_hotpotqa) as f:
            flare_hqa = json.load(f)
        results["flare_hotpotqa"] = analyze_ircot_hotpotqa(flare_hqa)

    if args.flare_musique:
        print(f"\nLoading FLARE MuSiQue from {args.flare_musique}")
        with open(args.flare_musique) as f:
            flare_msq = json.load(f)
        results["flare_musique"] = analyze_ircot_musique(flare_msq)

    # ── Generate summary ────────────────────────────────────────────────
    generate_summary(results, args.output_dir)

    # Save full results as JSON
    results_path = os.path.join(args.output_dir, "day4_analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    main()
