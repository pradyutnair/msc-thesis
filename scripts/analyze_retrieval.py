"""Analyze retrieval quality for Day 1 Standard RAG results.

Computes:
- Retrieval Recall@5: fraction of GT supporting docs found in top-5
- Per-item retrieval success/failure
- For MuSiQue: per-hop retrieval analysis
- Error categorization: retrieval miss vs reasoning failure
"""

import json
import re
import sys
from collections import Counter


def extract_title(doc_content):
    """Extract title from FlashRAG doc format: first line in quotes."""
    first_line = doc_content.split("\n")[0].strip()
    # Try quoted title
    m = re.match(r'^"(.+)"$', first_line)
    if m:
        return m.group(1)
    # Fallback: return first line as-is
    return first_line


def normalize_title(title):
    """Normalize title for matching."""
    return title.lower().strip().replace("_", " ")


def analyze_hotpotqa(path):
    print("=" * 60)
    print("HOTPOTQA RETRIEVAL ANALYSIS")
    print("=" * 60)

    with open(path) as f:
        data = json.load(f)

    total = len(data)
    recall_scores = []
    categories = Counter()
    type_counts = Counter()
    type_recall = {}

    for item in data:
        meta = item["metadata"]
        gt_titles = set(normalize_title(t) for t in meta["supporting_facts"]["title"])
        q_type = meta.get("type", "unknown")
        type_counts[q_type] += 1

        retrieved_docs = item["output"]["retrieval_result"]
        retrieved_titles = set()
        for doc in retrieved_docs:
            title = extract_title(doc["contents"])
            retrieved_titles.add(normalize_title(title))

        # Recall: fraction of GT titles found in retrieved
        found = gt_titles & retrieved_titles
        recall = len(found) / len(gt_titles) if gt_titles else 1.0
        recall_scores.append(recall)

        # Track per-type
        if q_type not in type_recall:
            type_recall[q_type] = []
        type_recall[q_type].append(recall)

        # Categorize
        pred = item["output"].get("pred", "")
        em = item["output"]["metric_score"]["em"]
        f1 = item["output"]["metric_score"]["f1"]

        if em == 1.0:
            categories["correct"] += 1
        elif recall == 1.0:
            categories["reasoning_failure (all GT docs retrieved, wrong answer)"] += 1
        elif recall > 0:
            categories["partial_retrieval (some GT docs found)"] += 1
        else:
            categories["retrieval_miss (no GT docs found)"] += 1

    avg_recall = sum(recall_scores) / len(recall_scores)
    full_recall = sum(1 for r in recall_scores if r == 1.0)
    zero_recall = sum(1 for r in recall_scores if r == 0.0)

    print(f"\nTotal examples: {total}")
    print(f"\nRetrieval Recall@5:")
    print(f"  Average: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"  Full recall (all GT docs in top-5): {full_recall}/{total} ({full_recall/total*100:.1f}%)")
    print(f"  Zero recall (no GT docs in top-5): {zero_recall}/{total} ({zero_recall/total*100:.1f}%)")
    print(f"  Partial recall: {total - full_recall - zero_recall}/{total} ({(total-full_recall-zero_recall)/total*100:.1f}%)")

    print(f"\nRetrieval Recall by question type:")
    for qtype in sorted(type_recall.keys()):
        scores = type_recall[qtype]
        avg = sum(scores) / len(scores)
        print(f"  {qtype} (n={type_counts[qtype]}): avg recall={avg:.4f}")

    print(f"\nError categorization:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")

    # Distribution of recall scores
    print(f"\nRecall@5 distribution:")
    for threshold in [0.0, 0.5, 1.0]:
        count = sum(1 for r in recall_scores if r == threshold)
        print(f"  recall={threshold}: {count} ({count/total*100:.1f}%)")

    return {
        "avg_recall": avg_recall,
        "full_recall_pct": full_recall / total,
        "zero_recall_pct": zero_recall / total,
        "categories": dict(categories),
    }


def analyze_musique(path):
    print("\n" + "=" * 60)
    print("MUSIQUE RETRIEVAL ANALYSIS")
    print("=" * 60)

    with open(path) as f:
        data = json.load(f)

    total = len(data)
    recall_scores = []
    categories = Counter()
    hop_counts = Counter()

    # Per-hop analysis
    hop1_found = 0
    hop2_found = 0
    hop1_total = 0
    hop2_total = 0
    hop3_found = 0
    hop3_total = 0
    hop4_found = 0
    hop4_total = 0

    for item in data:
        meta = item["metadata"]
        decomp = meta["question_decomposition"]
        n_hops = len(decomp)
        hop_counts[n_hops] += 1

        # Get GT titles from all hops
        gt_titles = set()
        hop_titles = []  # ordered by hop
        for hop in decomp:
            sp = hop.get("support_paragraph", {})
            title = sp.get("title", "")
            if title:
                gt_titles.add(normalize_title(title))
                hop_titles.append(normalize_title(title))
            else:
                hop_titles.append(None)

        # Get retrieved titles
        retrieved_docs = item["output"]["retrieval_result"]
        retrieved_titles = set()
        for doc in retrieved_docs:
            title = extract_title(doc["contents"])
            retrieved_titles.add(normalize_title(title))

        # Overall recall
        found = gt_titles & retrieved_titles
        recall = len(found) / len(gt_titles) if gt_titles else 1.0
        recall_scores.append(recall)

        # Per-hop tracking
        for hop_idx, ht in enumerate(hop_titles):
            if ht is None:
                continue
            hop_num = hop_idx + 1
            is_found = ht in retrieved_titles
            if hop_num == 1:
                hop1_total += 1
                hop1_found += int(is_found)
            elif hop_num == 2:
                hop2_total += 1
                hop2_found += int(is_found)
            elif hop_num == 3:
                hop3_total += 1
                hop3_found += int(is_found)
            elif hop_num == 4:
                hop4_total += 1
                hop4_found += int(is_found)

        # Categorize
        em = item["output"]["metric_score"]["em"]

        if em == 1.0:
            categories["correct"] += 1
        elif recall == 1.0:
            categories["reasoning_failure (all GT docs retrieved)"] += 1
        elif recall > 0:
            # Check which hops are missing
            missing_hops = []
            for hop_idx, ht in enumerate(hop_titles):
                if ht and ht not in retrieved_titles:
                    missing_hops.append(hop_idx + 1)
            categories[f"partial_retrieval (missing hops: typical={missing_hops})"] += 1
            # Simplified
        else:
            categories["retrieval_miss (no GT docs found)"] += 1

    avg_recall = sum(recall_scores) / len(recall_scores)
    full_recall = sum(1 for r in recall_scores if r == 1.0)
    zero_recall = sum(1 for r in recall_scores if r == 0.0)

    print(f"\nTotal examples: {total}")
    print(f"\nHop distribution:")
    for h in sorted(hop_counts.keys()):
        print(f"  {h}-hop: {hop_counts[h]} ({hop_counts[h]/total*100:.1f}%)")

    print(f"\nRetrieval Recall@5:")
    print(f"  Average: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"  Full recall (all GT docs in top-5): {full_recall}/{total} ({full_recall/total*100:.1f}%)")
    print(f"  Zero recall (no GT docs in top-5): {zero_recall}/{total} ({zero_recall/total*100:.1f}%)")

    print(f"\nPer-hop retrieval recall (key insight):")
    if hop1_total:
        print(f"  Hop 1: {hop1_found}/{hop1_total} ({hop1_found/hop1_total*100:.1f}%)")
    if hop2_total:
        print(f"  Hop 2: {hop2_found}/{hop2_total} ({hop2_found/hop2_total*100:.1f}%)")
    if hop3_total:
        print(f"  Hop 3: {hop3_found}/{hop3_total} ({hop3_found/hop3_total*100:.1f}%)")
    if hop4_total:
        print(f"  Hop 4: {hop4_found}/{hop4_total} ({hop4_found/hop4_total*100:.1f}%)")

    # Simplified error categorization
    cats_simple = Counter()
    for item in data:
        meta = item["metadata"]
        decomp = meta["question_decomposition"]
        gt_titles = set()
        hop_titles = []
        for hop in decomp:
            sp = hop.get("support_paragraph", {})
            title = sp.get("title", "")
            if title:
                gt_titles.add(normalize_title(title))
                hop_titles.append(normalize_title(title))
            else:
                hop_titles.append(None)

        retrieved_titles = set()
        for doc in item["output"]["retrieval_result"]:
            retrieved_titles.add(normalize_title(extract_title(doc["contents"])))

        found = gt_titles & retrieved_titles
        recall = len(found) / len(gt_titles) if gt_titles else 1.0
        em = item["output"]["metric_score"]["em"]

        if em == 1.0:
            cats_simple["correct"] += 1
        elif recall == 1.0:
            cats_simple["reasoning_failure"] += 1
        elif 0 < recall < 1.0:
            cats_simple["partial_retrieval"] += 1
        else:
            cats_simple["total_retrieval_miss"] += 1

    print(f"\nError categorization:")
    for cat, count in cats_simple.most_common():
        print(f"  {cat}: {count} ({count/total*100:.1f}%)")

    # Cross-tabulate: F1 by retrieval recall bucket
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
            avg_f1 = sum(f1s) / len(f1s)
            print(f"  {bucket}: n={len(f1s)}, avg_F1={avg_f1:.4f}")

    # Show some example failures
    print(f"\n--- 5 Example Failures: Total Retrieval Miss ---")
    count = 0
    for item, recall in zip(data, recall_scores):
        if recall == 0 and count < 5:
            decomp = item["metadata"]["question_decomposition"]
            gt_titles_list = [h.get("support_paragraph", {}).get("title", "?") for h in decomp]
            ret_titles = [extract_title(d["contents"]) for d in item["output"]["retrieval_result"]]
            print(f"\n  Q: {item['question']}")
            print(f"  Gold: {item['golden_answers']}")
            print(f"  Pred: {item['output']['pred']}")
            print(f"  GT docs needed: {gt_titles_list}")
            print(f"  Retrieved: {ret_titles}")
            count += 1

    print(f"\n--- 5 Example Failures: Reasoning Failure (all docs found, wrong answer) ---")
    count = 0
    for item, recall in zip(data, recall_scores):
        if recall == 1.0 and item["output"]["metric_score"]["em"] == 0 and count < 5:
            print(f"\n  Q: {item['question']}")
            print(f"  Gold: {item['golden_answers']}")
            print(f"  Pred: {item['output']['pred']}")
            f1 = item["output"]["metric_score"]["f1"]
            print(f"  F1: {f1:.3f}")
            count += 1


if __name__ == "__main__":
    hqa_results = analyze_hotpotqa(
        "/projects/prjs1800/results/day1/hotpotqa_2026_02_06_13_47_standard_rag_qwen25_hotpotqa/intermediate_data.json"
    )
    msq_results = analyze_musique(
        "/projects/prjs1800/results/day1/musique_2026_02_06_14_08_standard_rag_qwen25_musique/intermediate_data.json"
    )
