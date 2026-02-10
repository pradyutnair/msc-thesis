"""Error Taxonomy, Cross-Method Venn Analysis, and Per-Hop Analysis.

Day 6: Systematic failure categorization for multi-agent justification.

Error categories:
1. Correct: F1 >= 0.5
2. Retrieval Miss (Total): retrieval recall = 0 (no gold docs in top-k)
3. Retrieval Miss (Partial): 0 < recall < 1.0 (some gold docs missing)
4. Reasoning Failure: recall = 1.0 but F1 < 0.5 (had everything, still wrong)
5. Answer Extraction Failure: gold answer substring in raw output but pred doesn't match

Also computes:
- Cross-method Venn (complementarity analysis)
- Per-hop retrieval analysis for MuSiQue

Usage:
    python -u scripts/day6/error_taxonomy.py --output_dir /projects/prjs1800/analysis/day6
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ── Evaluation functions ────────────────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


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


# ── Title extraction from retrieved docs ────────────────────────────────────

def extract_title_from_contents(contents):
    """Extract title from retrieved doc contents field.
    Format: '"Title"\nContent...' — first line is quoted title.
    """
    if not contents:
        return ""
    first_line = contents.split("\n")[0].strip()
    # Strip surrounding quotes
    return first_line.strip('"').strip("'").strip()


def get_gold_titles(metadata, dataset_name):
    """Get set of gold supporting paragraph titles from metadata."""
    titles = set()

    if dataset_name in ("hotpotqa", "2wikimultihopqa"):
        sf = metadata.get("supporting_facts", {})
        sf_titles = sf.get("title", [])
        titles = set(t.lower().strip() for t in sf_titles)

    elif dataset_name == "musique":
        decomp = metadata.get("question_decomposition", [])
        for step in decomp:
            sp = step.get("support_paragraph", {})
            title = sp.get("title", "")
            if title:
                titles.add(title.lower().strip())

    return titles


def get_retrieved_titles(retrieval_result):
    """Get list of titles from retrieval result docs."""
    titles = []
    if not retrieval_result:
        return titles
    for doc in retrieval_result:
        if isinstance(doc, dict):
            contents = doc.get("contents", "")
        else:
            contents = str(doc)
        title = extract_title_from_contents(contents)
        titles.append(title.lower().strip())
    return titles


def compute_retrieval_recall(gold_titles, retrieved_titles):
    """Compute what fraction of gold titles appear in retrieved titles."""
    if not gold_titles:
        return 1.0  # No gold docs needed = perfect recall
    found = sum(1 for gt in gold_titles if gt in set(retrieved_titles))
    return found / len(gold_titles)


# ── Error categorization ───────────────────────────────────────────────────

def categorize_error(item, dataset_name, f1_threshold=0.5):
    """Categorize a single item's error type.

    Returns: category string and details dict.
    """
    gold = item.get("golden_answers", [])
    pred = item.get("output", {}).get("pred", "")
    metadata = item.get("metadata", {})
    retrieval_result = item.get("output", {}).get("retrieval_result", [])

    f1 = compute_f1(pred, gold)

    # Category 1: Correct
    if f1 >= f1_threshold:
        return "correct", {"f1": f1}

    # Get retrieval recall
    gold_titles = get_gold_titles(metadata, dataset_name)
    retrieved_titles = get_retrieved_titles(retrieval_result)
    recall = compute_retrieval_recall(gold_titles, retrieved_titles)

    # Category 2: Total retrieval miss
    if recall == 0 and gold_titles:
        return "retrieval_miss_total", {"f1": f1, "recall": recall,
                                         "gold_titles": list(gold_titles),
                                         "retrieved_titles": retrieved_titles[:5]}

    # Category 3: Partial retrieval miss
    if recall < 1.0 and gold_titles:
        return "retrieval_miss_partial", {"f1": f1, "recall": recall,
                                           "gold_titles": list(gold_titles),
                                           "retrieved_titles": retrieved_titles[:5]}

    # Category 5: Answer extraction failure
    # Check if any gold answer appears as substring in the raw output
    raw_output = item.get("output", {}).get("raw_cot_output", "")
    if not raw_output:
        raw_output = pred  # For non-CoT methods, raw output IS the pred

    extraction_failure = False
    for g in gold:
        if g.lower() in raw_output.lower() and normalize_answer(g) != normalize_answer(pred):
            extraction_failure = True
            break

    if extraction_failure:
        return "answer_extraction_failure", {"f1": f1, "recall": recall}

    # Category 4: Reasoning failure (had everything, still wrong)
    return "reasoning_failure", {"f1": f1, "recall": recall}


# ── Experiment registry ─────────────────────────────────────────────────────

EXPERIMENTS = {
    # Day 1
    "day1_standard_rag_hotpotqa": {
        "path": "/projects/prjs1800/results/day1/hotpotqa_2026_02_06_13_47_standard_rag_qwen25_hotpotqa",
        "dataset": "hotpotqa",
        "method": "Standard RAG",
    },
    "day1_standard_rag_musique": {
        "path": "/projects/prjs1800/results/day1/musique_2026_02_06_14_08_standard_rag_qwen25_musique",
        "dataset": "musique",
        "method": "Standard RAG",
    },
    # Day 2
    "day2_reranker_hotpotqa": {
        "path": "/projects/prjs1800/results/day2/hotpotqa_2026_02_06_15_37_reranker_rag_qwen25_hotpotqa",
        "dataset": "hotpotqa",
        "method": "Reranker",
    },
    "day2_reranker_musique": {
        "path": "/projects/prjs1800/results/day2/musique_2026_02_06_15_37_reranker_rag_qwen25_musique",
        "dataset": "musique",
        "method": "Reranker",
    },
}


def find_experiments():
    """Find all experiment result directories including Day 4 and Day 6."""
    import glob

    all_exp = dict(EXPERIMENTS)

    # Day 4 IRCoT
    for dataset, ds_short in [("hotpotqa", "hotpotqa"), ("musique", "musique")]:
        matches = sorted(glob.glob(f"/projects/prjs1800/results/day4/*ircot*{ds_short}*"))
        if matches:
            key = f"day4_ircot_{dataset}"
            all_exp[key] = {"path": matches[-1], "dataset": dataset, "method": "IRCoT"}

    # Day 6 experiments
    base = "/projects/prjs1800/results/day6"
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
            all_exp[key] = {"path": matches[-1], "dataset": dataset, "method": method}

    return all_exp


# ── Per-hop analysis (MuSiQue) ──────────────────────────────────────────────

def per_hop_analysis(data, dataset_name):
    """Analyze per-hop retrieval success for MuSiQue."""
    if dataset_name != "musique":
        return None

    hop_results = defaultdict(lambda: {"total": 0, "found": 0})

    for item in data:
        metadata = item.get("metadata", {})
        decomp = metadata.get("question_decomposition", [])
        retrieval_result = item.get("output", {}).get("retrieval_result", [])

        if not decomp or not retrieval_result:
            continue

        retrieved_titles = set(get_retrieved_titles(retrieval_result))

        for hop_idx, step in enumerate(decomp):
            sp = step.get("support_paragraph", {})
            title = sp.get("title", "")
            if not title:
                continue

            hop_num = hop_idx + 1
            hop_results[hop_num]["total"] += 1
            if title.lower().strip() in retrieved_titles:
                hop_results[hop_num]["found"] += 1

    # Compute recall per hop
    results = {}
    for hop_num in sorted(hop_results.keys()):
        h = hop_results[hop_num]
        recall = h["found"] / h["total"] if h["total"] > 0 else 0
        results[f"hop_{hop_num}"] = {
            "total": h["total"],
            "found": h["found"],
            "recall": recall,
        }

    return results


# ── Cross-method Venn (complementarity analysis) ────────────────────────────

def cross_method_venn(all_data, methods, dataset, f1_threshold=0.5):
    """Compute which questions each method answers correctly, for complementarity analysis.

    Returns per-method correct sets, unique-to-method counts, union (ensemble ceiling).
    """
    # Build correct sets per method
    method_correct = {}
    all_ids = set()

    for method_key in methods:
        data = all_data.get(method_key)
        if data is None:
            continue
        correct_ids = set()
        for item in data:
            qid = item.get("id", "")
            all_ids.add(qid)
            gold = item.get("golden_answers", [])
            pred = item.get("output", {}).get("pred", "")
            f1 = compute_f1(pred, gold)
            if f1 >= f1_threshold:
                correct_ids.add(qid)
        method_correct[method_key] = correct_ids

    if not method_correct:
        return None

    # Compute set operations
    result = {
        "dataset": dataset,
        "total_questions": len(all_ids),
        "methods": {},
        "union": {},
        "intersections": {},
    }

    # Per-method stats
    all_correct = set()
    for mk, correct in method_correct.items():
        all_correct |= correct
        result["methods"][mk] = {
            "n_correct": len(correct),
            "pct_correct": len(correct) / len(all_ids) if all_ids else 0,
        }

    # Unique-to-method: correct by this method but NOT by any other
    for mk, correct in method_correct.items():
        others = set()
        for other_mk, other_correct in method_correct.items():
            if other_mk != mk:
                others |= other_correct
        unique = correct - others
        result["methods"][mk]["n_unique"] = len(unique)
        result["methods"][mk]["pct_unique"] = len(unique) / len(all_ids) if all_ids else 0

    # Union (ensemble ceiling)
    result["union"] = {
        "n_correct": len(all_correct),
        "pct_correct": len(all_correct) / len(all_ids) if all_ids else 0,
    }

    # Pairwise intersections
    method_keys = list(method_correct.keys())
    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            mk_i, mk_j = method_keys[i], method_keys[j]
            overlap = method_correct[mk_i] & method_correct[mk_j]
            result["intersections"][f"{mk_i} & {mk_j}"] = {
                "n_overlap": len(overlap),
            }

    # All-method intersection
    if len(method_correct) > 1:
        all_intersection = set.intersection(*method_correct.values())
        result["intersections"]["all_methods"] = {
            "n_overlap": len(all_intersection),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all experiments
    print("=== Finding experiments ===")
    experiments = find_experiments()
    for key, info in sorted(experiments.items()):
        print(f"  {key}: {info['method']} on {info['dataset']} @ {info['path']}")

    # Load all intermediate data
    print(f"\n=== Loading intermediate data ===")
    all_data = {}
    for key, info in experiments.items():
        data_path = os.path.join(info["path"], "intermediate_data.json")
        if os.path.exists(data_path):
            with open(data_path) as f:
                all_data[key] = json.load(f)
            print(f"  {key}: loaded {len(all_data[key])} items")
        else:
            print(f"  {key}: MISSING {data_path}")

    # ── Error Taxonomy ──────────────────────────────────────────────────────
    print(f"\n=== Error Taxonomy ===")

    taxonomy_results = {}
    for key, data in all_data.items():
        info = experiments[key]
        dataset_name = info["dataset"]
        method = info["method"]

        # Skip naive gen (no retrieval to analyze)
        if "naive" in key:
            continue

        print(f"\n--- {key} ({method} on {dataset_name}) ---")

        category_counts = defaultdict(int)
        category_items = defaultdict(list)

        for item in data:
            cat, details = categorize_error(item, dataset_name)
            category_counts[cat] += 1
            if len(category_items[cat]) < 5:  # Keep up to 5 examples per category
                category_items[cat].append({
                    "id": item.get("id", ""),
                    "question": item.get("question", "")[:100],
                    "pred": item.get("output", {}).get("pred", "")[:100],
                    "gold": item.get("golden_answers", []),
                    **details,
                })

        n_total = len(data)
        result = {
            "method": method,
            "dataset": dataset_name,
            "n_total": n_total,
            "categories": {},
        }

        for cat in ["correct", "retrieval_miss_total", "retrieval_miss_partial",
                     "reasoning_failure", "answer_extraction_failure"]:
            count = category_counts[cat]
            pct = count / n_total if n_total else 0
            result["categories"][cat] = {
                "count": count,
                "pct": pct,
                "examples": category_items[cat],
            }
            print(f"  {cat}: {count} ({100*pct:.1f}%)")

        # Verify sum
        total_categorized = sum(category_counts.values())
        assert total_categorized == n_total, \
            f"Category sum {total_categorized} != total {n_total}"

        taxonomy_results[key] = result

    # Save taxonomy
    tax_path = os.path.join(args.output_dir, "error_taxonomy.json")
    with open(tax_path, "w") as f:
        json.dump(taxonomy_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved error taxonomy to {tax_path}")

    # Per-dataset summaries
    for dataset in ["hotpotqa", "musique", "2wikimultihopqa"]:
        ds_results = {k: v for k, v in taxonomy_results.items()
                      if v["dataset"] == dataset}
        if ds_results:
            ds_path = os.path.join(args.output_dir, f"error_taxonomy_{dataset}.json")
            with open(ds_path, "w") as f:
                json.dump(ds_results, f, indent=2, ensure_ascii=False)

    # ── Per-Hop Analysis (MuSiQue) ──────────────────────────────────────────
    print(f"\n=== Per-Hop Analysis (MuSiQue) ===")

    hop_results = {}
    for key, data in all_data.items():
        info = experiments[key]
        if info["dataset"] != "musique":
            continue
        if "naive" in key or "gold" in key:
            continue

        print(f"\n--- {key} ({info['method']}) ---")
        result = per_hop_analysis(data, "musique")
        if result:
            hop_results[key] = {
                "method": info["method"],
                **result,
            }
            for hop_name, hop_data in sorted(result.items()):
                print(f"  {hop_name}: recall={hop_data['recall']:.3f} "
                      f"({hop_data['found']}/{hop_data['total']})")

    hop_path = os.path.join(args.output_dir, "hop_analysis_musique.json")
    with open(hop_path, "w") as f:
        json.dump(hop_results, f, indent=2)
    print(f"\nSaved hop analysis to {hop_path}")

    # ── Cross-Method Venn Analysis ──────────────────────────────────────────
    print(f"\n=== Cross-Method Venn (Complementarity Analysis) ===")

    venn_results = {}

    # Methods to compare per dataset
    venn_methods = {
        "hotpotqa": [
            "day1_standard_rag_hotpotqa",
            "day2_reranker_hotpotqa",
            "day4_ircot_hotpotqa",
        ],
        "musique": [
            "day1_standard_rag_musique",
            "day2_reranker_musique",
            "day4_ircot_musique",
        ],
        "2wikimultihopqa": [
            "day6_standard_rag_2wiki",
            "day6_reranker_2wiki",
        ],
    }

    for dataset, methods in venn_methods.items():
        available = [m for m in methods if m in all_data]
        if len(available) < 2:
            print(f"\n--- {dataset}: SKIP (need >= 2 methods, have {len(available)}) ---")
            continue

        print(f"\n--- {dataset} ---")
        result = cross_method_venn(all_data, available, dataset)
        if result:
            venn_results[dataset] = result
            print(f"  Total questions: {result['total_questions']}")
            print(f"  Ensemble ceiling: {result['union']['n_correct']} "
                  f"({100*result['union']['pct_correct']:.1f}%)")
            for mk, minfo in result["methods"].items():
                print(f"  {mk}: {minfo['n_correct']} correct, "
                      f"{minfo['n_unique']} unique ({100*minfo['pct_unique']:.1f}%)")

    venn_path = os.path.join(args.output_dir, "cross_method_venn.json")
    with open(venn_path, "w") as f:
        json.dump(venn_results, f, indent=2)
    print(f"\nSaved cross-method Venn to {venn_path}")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()
