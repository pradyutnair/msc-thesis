#!/usr/bin/env python3
"""
Post-hoc analysis of RAG baselines (Days 1-6).
Computes retrieval metrics, stratified analysis, correlation, and significance tests
from existing intermediate_data.json files without rerunning experiments.

Usage: python3 posthoc_analysis.py [--output-dir /path/to/output]
"""

import json
import os
import re
import string
import sys
import math
import random
import time
from collections import defaultdict, Counter
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_BASE = "/projects/prjs1800/results"
DATASET_BASE = "/projects/prjs1800/datasets/flashrag"
OUTPUT_DIR = "/projects/prjs1800/analysis"

# (label, method, dataset, result_subdir, has_retrieval)
EXPERIMENTS = [
    # Day 1: Standard RAG
    ("Day1_StdRAG", "Standard RAG", "hotpotqa",
     "day1/hotpotqa_2026_02_06_13_47_standard_rag_qwen25_hotpotqa", True),
    ("Day1_StdRAG", "Standard RAG", "musique",
     "day1/musique_2026_02_06_14_08_standard_rag_qwen25_musique", True),
    # Day 2: Reranker
    ("Day2_Reranker", "Reranker", "hotpotqa",
     "day2/hotpotqa_2026_02_06_15_37_reranker_rag_qwen25_hotpotqa", True),
    ("Day2_Reranker", "Reranker", "musique",
     "day2/musique_2026_02_06_15_37_reranker_rag_qwen25_musique", True),
    # Day 3: RECOMP
    ("Day3_RECOMP", "RECOMP", "hotpotqa",
     "day3/hotpotqa_2026_02_06_17_18_recomp_rag_qwen25_hotpotqa", True),
    ("Day3_RECOMP", "RECOMP", "musique",
     "day3/musique_2026_02_06_17_18_recomp_rag_qwen25_musique", True),
    # Day 3: Selective-Context
    ("Day3_SC", "Selective-Context", "hotpotqa",
     "day3/hotpotqa_2026_02_06_17_18_sc_rag_qwen25_hotpotqa", True),
    ("Day3_SC", "Selective-Context", "musique",
     "day3/musique_2026_02_06_17_18_sc_rag_qwen25_musique", True),
    # Day 4: IRCoT (note: accumulates ~18 docs across rounds, not just 5)
    ("Day4_IRCoT", "IRCoT", "hotpotqa",
     "day4/hotpotqa_2026_02_07_00_20_ircot_qwen25_hotpotqa", True),
    ("Day4_IRCoT", "IRCoT", "musique",
     "day4/musique_2026_02_07_00_20_ircot_qwen25_musique", True),
    # Day 4: FLARE (no retrieval_result stored — only judge booleans)
    ("Day4_FLARE", "FLARE", "hotpotqa",
     "day4/hotpotqa_2026_02_06_22_00_flare_qwen25_hotpotqa", False),
    ("Day4_FLARE", "FLARE", "musique",
     "day4/musique_2026_02_07_00_10_flare_qwen25_musique", False),
    # Day 5: Reranker + CoT
    ("Day5_RnkCoT", "Reranker+CoT", "hotpotqa",
     "day5/hotpotqa_2026_02_07_12_43_reranker_cot_qwen25_hotpotqa", True),
    ("Day5_RnkCoT", "Reranker+CoT", "musique",
     "day5/musique_2026_02_07_13_23_reranker_cot_qwen25_musique", True),
    # Day 6: Naive Generation (no retrieval)
    ("Day6_Naive", "Naive Gen", "hotpotqa",
     "day6/hotpotqa_2026_02_10_10_52_naive_gen_qwen25_hotpotqa", False),
    ("Day6_Naive", "Naive Gen", "musique",
     "day6/musique_2026_02_10_10_56_naive_gen_qwen25_musique", False),
    ("Day6_Naive", "Naive Gen", "2wikimultihopqa",
     "day6/2wikimultihopqa_2026_02_10_10_57_naive_gen_qwen25_2wiki", False),
    # Day 6: Gold Context (gold docs injected, not retrieved)
    ("Day6_Gold", "Gold Context", "hotpotqa",
     "day6/hotpotqa_2026_02_10_10_58_gold_context_qwen25_hotpotqa", False),
    ("Day6_Gold", "Gold Context", "musique",
     "day6/musique_2026_02_10_11_00_gold_context_qwen25_musique", False),
    ("Day6_Gold", "Gold Context", "2wikimultihopqa",
     "day6/2wikimultihopqa_2026_02_10_11_08_gold_context_qwen25_2wiki", False),
    # Day 6: Standard RAG on 2Wiki
    ("Day6_StdRAG", "Standard RAG", "2wikimultihopqa",
     "day6/2wikimultihopqa_2026_02_10_11_16_standard_rag_qwen25_2wiki", True),
    # Day 6: Reranker on 2Wiki
    ("Day6_Reranker", "Reranker", "2wikimultihopqa",
     "day6/2wikimultihopqa_2026_02_10_11_52_reranker_qwen25_2wiki", True),
]

# Significance test pairs: (label_a, label_b, dataset)
# We test key comparisons within each dataset
SIG_PAIRS_HQA = [
    ("Day1_StdRAG", "Day2_Reranker"),      # Reranker value
    ("Day1_StdRAG", "Day4_IRCoT"),          # IRCoT value
    ("Day2_Reranker", "Day4_IRCoT"),        # Reranker vs IRCoT
    ("Day2_Reranker", "Day5_RnkCoT"),       # CoT effect
    ("Day1_StdRAG", "Day3_RECOMP"),         # RECOMP effect
    ("Day1_StdRAG", "Day3_SC"),             # SC effect
    ("Day1_StdRAG", "Day4_FLARE"),          # FLARE effect
    ("Day6_Naive", "Day1_StdRAG"),          # Retrieval value
    ("Day2_Reranker", "Day6_Gold"),         # Remaining gap
]

SIG_PAIRS_MSQ = SIG_PAIRS_HQA  # Same pairs for MuSiQue

SIG_PAIRS_2WIKI = [
    ("Day6_Naive", "Day6_StdRAG"),          # Retrieval value
    ("Day6_StdRAG", "Day6_Reranker"),       # Reranker value
    ("Day6_Reranker", "Day6_Gold"),         # Remaining gap
]


# ═══════════════════════════════════════════════════════════════════════════════
# QA EVALUATION (for experiments missing per-question metric_score)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def compute_em(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_qa_metrics(prediction, golden_answers):
    """Compute EM and F1 against best-matching gold answer."""
    if not prediction or not golden_answers:
        return 0.0, 0.0
    best_em = max(compute_em(prediction, ga) for ga in golden_answers)
    best_f1 = max(compute_f1(prediction, ga) for ga in golden_answers)
    return best_em, best_f1


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_title(contents):
    """Extract title from contents field (format: '"Title"\\ncontent...')."""
    if not contents:
        return ""
    if contents.startswith('"'):
        end = contents.find('"', 1)
        if end > 0:
            return contents[1:end].lower().strip()
    return contents.split("\n")[0].lower().strip()


def get_gold_titles(item, dataset):
    """Get set of gold supporting document titles for a question."""
    metadata = item.get("metadata", {})

    if dataset == "musique":
        titles = set()
        for step in metadata.get("question_decomposition", []):
            sp = step.get("support_paragraph", {})
            t = sp.get("title", "")
            if t:
                titles.add(t.lower().strip())
        return titles
    else:  # hotpotqa, 2wikimultihopqa
        sf = metadata.get("supporting_facts", {})
        raw = sf.get("title", [])
        return set(t.lower().strip() for t in raw if t)


def get_retrieved_titles_ordered(item):
    """Get ordered list of retrieved document titles."""
    rr = item.get("output", {}).get("retrieval_result", [])
    if not rr:
        return []
    titles = []
    for doc in rr:
        if isinstance(doc, dict) and "contents" in doc:
            titles.append(extract_title(doc["contents"]))
    return titles


def compute_retrieval_metrics(gold_titles, retrieved_titles):
    """Compute retrieval metrics for a single question.

    Returns dict with contain, recall, precision, mrr, n_gold, n_found, n_retrieved.
    Returns None if no gold titles available.
    """
    if not gold_titles:
        return None

    k = len(retrieved_titles)
    retrieved_set = set(retrieved_titles)

    found = gold_titles & retrieved_set
    n_found = len(found)
    n_gold = len(gold_titles)

    contain = 1.0 if n_found > 0 else 0.0
    recall = n_found / n_gold
    precision = n_found / k if k > 0 else 0.0

    # MRR: reciprocal rank of first gold doc
    mrr = 0.0
    for i, title in enumerate(retrieved_titles):
        if title in gold_titles:
            mrr = 1.0 / (i + 1)
            break

    return {
        "contain": contain,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "n_gold": n_gold,
        "n_found": n_found,
        "n_retrieved": k,
    }


def get_strata(item, dataset):
    """Get stratification keys for a question."""
    metadata = item.get("metadata", {})
    strata = {}

    if dataset == "hotpotqa":
        strata["type"] = metadata.get("type", "unknown")
        strata["level"] = metadata.get("level", "unknown")
    elif dataset == "musique":
        decomp = metadata.get("question_decomposition", [])
        strata["n_hops"] = str(len(decomp))
    elif dataset == "2wikimultihopqa":
        strata["type"] = metadata.get("type", "unknown")

    return strata


def spearman_rank(x, y):
    """Compute Spearman rank correlation and approximate p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    def rankdata(arr):
        indexed = sorted(range(len(arr)), key=lambda i: arr[i])
        ranks = [0.0] * len(arr)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and arr[indexed[j]] == arr[indexed[i]]:
                j += 1
            avg_rank = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[indexed[k]] = avg_rank
            i = j
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)

    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n

    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    std_x = math.sqrt(max(0, sum((rx[i] - mean_rx) ** 2 for i in range(n))))
    std_y = math.sqrt(max(0, sum((ry[i] - mean_ry) ** 2 for i in range(n))))

    if std_x == 0 or std_y == 0:
        return 0.0, 1.0

    r = cov / (std_x * std_y)
    r = max(-1.0, min(1.0, r))

    if abs(r) >= 0.9999:
        return r, 0.0

    t_stat = r * math.sqrt((n - 2) / (1 - r * r))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    return r, p


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=None, seed=42):
    """Paired bootstrap significance test (two-sided).

    Returns (observed_mean_diff, p_value).
    """
    rng = random.Random(seed)
    n = len(scores_a)
    assert n == len(scores_b), f"Length mismatch: {n} vs {len(scores_b)}"

    if n_bootstrap is None:
        n_bootstrap = 1000 if n > 10000 else 10000

    diffs = [scores_a[i] - scores_b[i] for i in range(n)]
    observed = sum(diffs) / n

    # Center diffs under H0 (mean diff = 0)
    centered = [d - observed for d in diffs]

    count = 0
    for _ in range(n_bootstrap):
        boot = [centered[rng.randint(0, n - 1)] for _ in range(n)]
        boot_mean = sum(boot) / n
        if abs(boot_mean) >= abs(observed):
            count += 1

    p_value = count / n_bootstrap
    return observed, p_value


def sig_stars(p):
    """Return significance stars for p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def process_experiment(label, method, dataset, result_subdir, has_retrieval):
    """Process one experiment's intermediate_data.json.

    Returns a list of per-question result dicts.
    """
    data_path = os.path.join(RESULTS_BASE, result_subdir, "intermediate_data.json")

    if not os.path.exists(data_path):
        print(f"  WARNING: {data_path} not found, skipping")
        return []

    t0 = time.time()
    with open(data_path, "r") as f:
        data = json.load(f)
    t1 = time.time()
    print(f"  Loaded {len(data)} items in {t1 - t0:.1f}s")

    results = []
    for item in data:
        qid = item.get("id", "")
        output = item.get("output", {})

        # Answer metrics (use stored metric_score if available, else recompute)
        ms = output.get("metric_score", {})
        em = ms.get("em", None)
        f1 = ms.get("f1", None)

        if em is None or f1 is None:
            pred = output.get("pred", "")
            golden = item.get("golden_answers", [])
            em, f1 = compute_qa_metrics(pred, golden)

        # Stratification
        strata = get_strata(item, dataset)

        row = {
            "qid": qid,
            "em": em,
            "f1": f1,
            "strata": strata,
        }

        # Retrieval metrics (only if experiment has retrieval data)
        if has_retrieval:
            gold_titles = get_gold_titles(item, dataset)
            retrieved_titles = get_retrieved_titles_ordered(item)
            ret_metrics = compute_retrieval_metrics(gold_titles, retrieved_titles)
            if ret_metrics:
                row.update(ret_metrics)
            else:
                row["contain"] = None

        results.append(row)

    return results


def aggregate_metrics(per_question):
    """Compute aggregate metrics from per-question results."""
    n = len(per_question)
    if n == 0:
        return {}

    agg = {
        "n": n,
        "em": sum(r["em"] for r in per_question) / n,
        "f1": sum(r["f1"] for r in per_question) / n,
    }

    # Retrieval metrics (only for items that have them)
    ret_items = [r for r in per_question if r.get("contain") is not None]
    if ret_items:
        nr = len(ret_items)
        agg["retrieval_n"] = nr
        agg["contain"] = sum(r["contain"] for r in ret_items) / nr
        agg["recall"] = sum(r["recall"] for r in ret_items) / nr
        agg["precision"] = sum(r["precision"] for r in ret_items) / nr
        agg["mrr"] = sum(r["mrr"] for r in ret_items) / nr
        agg["avg_n_retrieved"] = sum(r["n_retrieved"] for r in ret_items) / nr
        agg["avg_n_gold"] = sum(r["n_gold"] for r in ret_items) / nr

        # Coverage distribution
        coverage = Counter()
        for r in ret_items:
            if r["n_found"] == 0:
                coverage["0_found"] += 1
            elif r["n_found"] < r["n_gold"]:
                coverage["partial"] += 1
            else:
                coverage["all_found"] += 1
        agg["coverage_0_found"] = coverage["0_found"] / nr
        agg["coverage_partial"] = coverage["partial"] / nr
        agg["coverage_all_found"] = coverage["all_found"] / nr

    return agg


def stratified_metrics(per_question, stratum_key):
    """Compute metrics stratified by a given key."""
    groups = defaultdict(list)
    for r in per_question:
        key = r["strata"].get(stratum_key, "unknown")
        groups[key].append(r)

    result = {}
    for key, items in sorted(groups.items()):
        agg = aggregate_metrics(items)
        result[key] = agg

    return result


def compute_correlation(per_question):
    """Compute Spearman correlation between retrieval recall and answer F1."""
    pairs = [(r["recall"], r["f1"]) for r in per_question
             if r.get("recall") is not None and r.get("f1") is not None]
    if len(pairs) < 10:
        return None

    recalls = [p[0] for p in pairs]
    f1s = [p[1] for p in pairs]

    r, p = spearman_rank(recalls, f1s)

    # Also compute stratified F1 by recall bins
    bins = {"recall=0": [], "0<recall<1": [], "recall=1": []}
    for rec, f1 in pairs:
        if rec == 0:
            bins["recall=0"].append(f1)
        elif rec < 1.0:
            bins["0<recall<1"].append(f1)
        else:
            bins["recall=1"].append(f1)

    binned = {}
    for bname, vals in bins.items():
        if vals:
            binned[bname] = {
                "n": len(vals),
                "mean_f1": sum(vals) / len(vals),
                "mean_em": None,  # not tracked here
            }

    return {
        "spearman_r": r,
        "spearman_p": p,
        "n_pairs": len(pairs),
        "binned_f1": binned,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(all_results, all_agg, all_strat, all_corr, all_sig):
    """Generate a markdown summary report."""
    lines = []
    lines.append("# Post-Hoc Analysis Report: Single-Agent RAG Baselines (Days 1-6)")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── Table 1: Retrieval Metrics ──────────────────────────────────────────
    lines.append("## 1. Retrieval Metrics")
    lines.append("")
    lines.append("| Method | Dataset | k | Contain@k | Recall@k | Precision@k | MRR | Coverage: 0 found | Partial | All found |")
    lines.append("|--------|---------|---|-----------|----------|-------------|-----|-------------------|---------|-----------|")

    for (label, dataset), agg in sorted(all_agg.items()):
        if "contain" not in agg:
            continue
        method_name = label.split("_", 1)[1] if "_" in label else label
        for exp in EXPERIMENTS:
            if exp[0] == label and exp[2] == dataset:
                method_name = exp[1]
                break
        k = f"{agg.get('avg_n_retrieved', 0):.0f}"
        lines.append(
            f"| {method_name} | {dataset} | {k} "
            f"| {agg['contain']*100:.1f}% "
            f"| {agg['recall']*100:.1f}% "
            f"| {agg['precision']*100:.1f}% "
            f"| {agg['mrr']:.3f} "
            f"| {agg['coverage_0_found']*100:.1f}% "
            f"| {agg['coverage_partial']*100:.1f}% "
            f"| {agg['coverage_all_found']*100:.1f}% |"
        )
    lines.append("")

    # ── Table 2: Stratified Analysis ────────────────────────────────────────
    lines.append("## 2. Stratified Analysis")
    lines.append("")

    # HotpotQA by type
    lines.append("### HotpotQA by Question Type")
    lines.append("")
    lines.append("| Method | Bridge EM | Bridge F1 | Comparison EM | Comparison F1 | Bridge Contain | Comp Contain |")
    lines.append("|--------|-----------|-----------|---------------|---------------|----------------|--------------|")

    for (label, dataset), strat in sorted(all_strat.items()):
        if dataset != "hotpotqa" or "type" not in strat:
            continue
        method_name = label
        for exp in EXPERIMENTS:
            if exp[0] == label and exp[2] == dataset:
                method_name = exp[1]
                break
        st = strat["type"]
        bridge = st.get("bridge", {})
        comp = st.get("comparison", {})

        b_contain = f"{bridge.get('contain', 0)*100:.1f}%" if "contain" in bridge else "—"
        c_contain = f"{comp.get('contain', 0)*100:.1f}%" if "contain" in comp else "—"

        lines.append(
            f"| {method_name} "
            f"| {bridge.get('em', 0)*100:.1f}% "
            f"| {bridge.get('f1', 0)*100:.1f}% "
            f"| {comp.get('em', 0)*100:.1f}% "
            f"| {comp.get('f1', 0)*100:.1f}% "
            f"| {b_contain} "
            f"| {c_contain} |"
        )
    lines.append("")

    # HotpotQA by level
    lines.append("### HotpotQA by Difficulty Level")
    lines.append("")
    lines.append("| Method | Easy F1 | Medium F1 | Hard F1 |")
    lines.append("|--------|---------|-----------|---------|")

    for (label, dataset), strat in sorted(all_strat.items()):
        if dataset != "hotpotqa" or "level" not in strat:
            continue
        method_name = label
        for exp in EXPERIMENTS:
            if exp[0] == label and exp[2] == dataset:
                method_name = exp[1]
                break
        st = strat["level"]
        lines.append(
            f"| {method_name} "
            f"| {st.get('easy', {}).get('f1', 0)*100:.1f}% "
            f"| {st.get('medium', {}).get('f1', 0)*100:.1f}% "
            f"| {st.get('hard', {}).get('f1', 0)*100:.1f}% |"
        )
    lines.append("")

    # MuSiQue by number of hops
    lines.append("### MuSiQue by Number of Hops")
    lines.append("")
    lines.append("| Method | 2-hop F1 (n) | 3-hop F1 (n) | 4-hop F1 (n) | 2-hop Contain | 3-hop Contain | 4-hop Contain |")
    lines.append("|--------|-------------|-------------|-------------|---------------|---------------|---------------|")

    for (label, dataset), strat in sorted(all_strat.items()):
        if dataset != "musique" or "n_hops" not in strat:
            continue
        method_name = label
        for exp in EXPERIMENTS:
            if exp[0] == label and exp[2] == dataset:
                method_name = exp[1]
                break
        st = strat["n_hops"]

        cols = []
        for h in ["2", "3", "4"]:
            hd = st.get(h, {})
            f1_str = f"{hd.get('f1', 0)*100:.1f}% ({hd.get('n', 0)})"
            cols.append(f1_str)
        for h in ["2", "3", "4"]:
            hd = st.get(h, {})
            c_str = f"{hd.get('contain', 0)*100:.1f}%" if "contain" in hd else "—"
            cols.append(c_str)

        lines.append(f"| {method_name} | " + " | ".join(cols) + " |")
    lines.append("")

    # 2Wiki by type
    lines.append("### 2WikiMultihopQA by Question Type")
    lines.append("")
    type_names = ["compositional", "comparison", "inference", "bridge_comparison"]
    header = "| Method | " + " | ".join(f"{t} F1" for t in type_names) + " |"
    sep = "|--------|" + "|".join("----------" for _ in type_names) + "|"
    lines.append(header)
    lines.append(sep)

    for (label, dataset), strat in sorted(all_strat.items()):
        if dataset != "2wikimultihopqa" or "type" not in strat:
            continue
        method_name = label
        for exp in EXPERIMENTS:
            if exp[0] == label and exp[2] == dataset:
                method_name = exp[1]
                break
        st = strat["type"]
        cols = []
        for t in type_names:
            td = st.get(t, {})
            cols.append(f"{td.get('f1', 0)*100:.1f}% ({td.get('n', 0)})")
        lines.append(f"| {method_name} | " + " | ".join(cols) + " |")
    lines.append("")

    # ── Table 3: Retrieval-Answer Correlation ───────────────────────────────
    lines.append("## 3. Retrieval-Answer Correlation")
    lines.append("")
    lines.append("| Method | Dataset | Spearman r | p-value | n | F1 (recall=0) | F1 (0<recall<1) | F1 (recall=1) |")
    lines.append("|--------|---------|------------|---------|---|---------------|-----------------|---------------|")

    for (label, dataset), corr in sorted(all_corr.items()):
        if corr is None:
            continue
        method_name = label
        for exp in EXPERIMENTS:
            if exp[0] == label and exp[2] == dataset:
                method_name = exp[1]
                break
        binned = corr["binned_f1"]
        f1_0 = f"{binned['recall=0']['mean_f1']*100:.1f}% ({binned['recall=0']['n']})" if "recall=0" in binned else "—"
        f1_p = f"{binned['0<recall<1']['mean_f1']*100:.1f}% ({binned['0<recall<1']['n']})" if "0<recall<1" in binned else "—"
        f1_1 = f"{binned['recall=1']['mean_f1']*100:.1f}% ({binned['recall=1']['n']})" if "recall=1" in binned else "—"

        lines.append(
            f"| {method_name} | {dataset} "
            f"| {corr['spearman_r']:.3f} "
            f"| {corr['spearman_p']:.2e} "
            f"| {corr['n_pairs']} "
            f"| {f1_0} "
            f"| {f1_p} "
            f"| {f1_1} |"
        )
    lines.append("")

    # ── Table 4: Statistical Significance ───────────────────────────────────
    lines.append("## 4. Statistical Significance (Paired Bootstrap, n=10000)")
    lines.append("")
    lines.append("| Method A | Method B | Dataset | Mean F1 A | Mean F1 B | Delta F1 | p-value | Sig |")
    lines.append("|----------|----------|---------|-----------|-----------|----------|---------|-----|")

    for entry in all_sig:
        lines.append(
            f"| {entry['method_a']} | {entry['method_b']} | {entry['dataset']} "
            f"| {entry['mean_a']*100:.2f}% "
            f"| {entry['mean_b']*100:.2f}% "
            f"| {entry['delta']*100:+.2f} "
            f"| {entry['p_value']:.4f} "
            f"| {entry['sig']} |"
        )
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    output_dir = OUTPUT_DIR
    if len(sys.argv) > 2 and sys.argv[1] == "--output-dir":
        output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    # Storage for all results
    all_per_question = {}   # (label, dataset) -> list of per-question dicts
    all_agg = {}            # (label, dataset) -> aggregate metrics dict
    all_strat = {}          # (label, dataset) -> {stratum_key: {value: agg}}
    all_corr = {}           # (label, dataset) -> correlation dict
    all_sig = []            # list of significance test results

    # ── Phase 1: Process each experiment ────────────────────────────────────
    print("=" * 70)
    print("PHASE 1: Processing experiments")
    print("=" * 70)

    for label, method, dataset, result_subdir, has_retrieval in EXPERIMENTS:
        key = (label, dataset)
        print(f"\n[{label}] {method} on {dataset} (retrieval={has_retrieval})")

        per_q = process_experiment(label, method, dataset, result_subdir, has_retrieval)
        if not per_q:
            print(f"  SKIPPED (no data)")
            continue

        all_per_question[key] = per_q
        all_agg[key] = aggregate_metrics(per_q)

        # Stratification
        strat = {}
        if dataset == "hotpotqa":
            strat["type"] = stratified_metrics(per_q, "type")
            strat["level"] = stratified_metrics(per_q, "level")
        elif dataset == "musique":
            strat["n_hops"] = stratified_metrics(per_q, "n_hops")
        elif dataset == "2wikimultihopqa":
            strat["type"] = stratified_metrics(per_q, "type")
        all_strat[key] = strat

        # Correlation (only for experiments with retrieval)
        if has_retrieval:
            corr = compute_correlation(per_q)
            all_corr[key] = corr
            if corr:
                print(f"  Correlation: r={corr['spearman_r']:.3f}, p={corr['spearman_p']:.2e}")

        # Print summary
        agg = all_agg[key]
        print(f"  EM={agg['em']*100:.2f}%, F1={agg['f1']*100:.2f}%", end="")
        if "contain" in agg:
            print(f", Contain={agg['contain']*100:.1f}%, Recall={agg['recall']*100:.1f}%, "
                  f"Precision={agg['precision']*100:.1f}%, MRR={agg['mrr']:.3f}", end="")
        print()

    # ── Phase 2: Significance tests ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Statistical significance tests")
    print("=" * 70)

    sig_configs = [
        ("hotpotqa", SIG_PAIRS_HQA),
        ("musique", SIG_PAIRS_MSQ),
        ("2wikimultihopqa", SIG_PAIRS_2WIKI),
    ]

    for dataset, pairs in sig_configs:
        print(f"\n--- {dataset} ---")
        for label_a, label_b in pairs:
            key_a = (label_a, dataset)
            key_b = (label_b, dataset)

            if key_a not in all_per_question or key_b not in all_per_question:
                print(f"  {label_a} vs {label_b}: SKIPPED (missing data)")
                continue

            pq_a = all_per_question[key_a]
            pq_b = all_per_question[key_b]

            # Align by question ID
            f1_map_a = {r["qid"]: r["f1"] for r in pq_a}
            f1_map_b = {r["qid"]: r["f1"] for r in pq_b}
            common_ids = sorted(set(f1_map_a.keys()) & set(f1_map_b.keys()))

            if len(common_ids) < 10:
                print(f"  {label_a} vs {label_b}: SKIPPED (only {len(common_ids)} common)")
                continue

            scores_a = [f1_map_a[qid] for qid in common_ids]
            scores_b = [f1_map_b[qid] for qid in common_ids]

            delta, p_val = paired_bootstrap_test(scores_a, scores_b)
            mean_a = sum(scores_a) / len(scores_a)
            mean_b = sum(scores_b) / len(scores_b)

            method_a = label_a
            method_b = label_b
            for exp in EXPERIMENTS:
                if exp[0] == label_a and exp[2] == dataset:
                    method_a = exp[1]
                if exp[0] == label_b and exp[2] == dataset:
                    method_b = exp[1]

            entry = {
                "label_a": label_a, "label_b": label_b,
                "method_a": method_a, "method_b": method_b,
                "dataset": dataset,
                "n_common": len(common_ids),
                "mean_a": mean_a, "mean_b": mean_b,
                "delta": delta, "p_value": p_val,
                "sig": sig_stars(p_val),
            }
            all_sig.append(entry)

            print(f"  {method_a} vs {method_b}: "
                  f"delta={delta*100:+.2f} F1, p={p_val:.4f} {sig_stars(p_val)}")

    # ── Phase 3: Generate reports ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Generating reports")
    print("=" * 70)

    # Save raw results as JSON
    json_output = {
        "aggregates": {f"{k[0]}_{k[1]}": v for k, v in all_agg.items()},
        "stratified": {f"{k[0]}_{k[1]}": v for k, v in all_strat.items()},
        "correlations": {f"{k[0]}_{k[1]}": v for k, v in all_corr.items()},
        "significance": all_sig,
    }

    json_path = os.path.join(output_dir, "posthoc_analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")

    # Generate markdown report
    md_report = generate_markdown_report(all_per_question, all_agg, all_strat, all_corr, all_sig)
    md_path = os.path.join(output_dir, "posthoc_analysis.md")
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"  Saved Markdown: {md_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
