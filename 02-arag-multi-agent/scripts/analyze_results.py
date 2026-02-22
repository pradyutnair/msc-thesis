#!/usr/bin/env python3
"""Analysis script: tables, figures, and failure taxonomy for MA²RAG results.

Usage:
    # Summarize a single experiment
    python scripts/analyze_results.py --results results/m1/hotpotqa/full/

    # Compare experiments across datasets
    python scripts/analyze_results.py --compare m1 m2 m3 m4 --datasets all

    # Failure taxonomy for MuSiQue
    python scripts/analyze_results.py --failure-taxonomy results/m1/musique/full/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# --------------------------------------------------------------------------
# Metric helpers
# --------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles/punctuation, collapse whitespace."""
    import re
    import string

    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_predictions(results_dir: Path) -> list[dict[str, Any]]:
    pred_file = results_dir / "predictions.jsonl"
    if not pred_file.exists():
        logger.error("No predictions.jsonl found in %s", results_dir)
        return []

    preds = []
    with open(pred_file, "r") as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summarize(results_dir: Path) -> dict[str, Any]:
    """Compute summary metrics for a results directory."""
    preds = load_predictions(results_dir)
    if not preds:
        return {}

    ems = [exact_match(p["pred_answer"], p["gold_answer"]) for p in preds]
    f1s = [token_f1(p["pred_answer"], p["gold_answer"]) for p in preds]
    latencies = [p.get("wall_clock_seconds", 0) for p in preds]
    tokens = [p.get("total_tokens", 0) for p in preds]
    errors = [p for p in preds if p.get("error")]

    # Question type distribution
    q_types = Counter(p.get("question_type", "unknown") for p in preds)

    # Per-type accuracy
    type_em: dict[str, list[bool]] = {}
    for p, em in zip(preds, ems):
        qt = p.get("question_type", "unknown")
        type_em.setdefault(qt, []).append(em)

    summary = {
        "n": len(preds),
        "em": np.mean(ems),
        "f1": np.mean(f1s),
        "em_ci_95": 1.96 * np.std(ems) / np.sqrt(len(ems)),
        "latency_p50": np.percentile(latencies, 50),
        "latency_p90": np.percentile(latencies, 90),
        "tokens_mean": np.mean(tokens),
        "tokens_per_correct": (
            np.sum(tokens) / max(sum(ems), 1)
        ),
        "error_rate": len(errors) / len(preds),
        "question_types": dict(q_types),
        "per_type_em": {
            qt: np.mean(vals) for qt, vals in type_em.items()
        },
    }

    return summary


def print_summary(results_dir: Path) -> None:
    s = summarize(results_dir)
    if not s:
        return

    print(f"\n{'='*60}")
    print(f"Results: {results_dir}")
    print(f"{'='*60}")
    print(f"  N = {s['n']}")
    print(f"  EM = {s['em']:.3f} (±{s['em_ci_95']:.3f})")
    print(f"  F1 = {s['f1']:.3f}")
    print(f"  Latency (p50/p90) = {s['latency_p50']:.1f}s / {s['latency_p90']:.1f}s")
    print(f"  Tokens/Q = {s['tokens_mean']:.0f}")
    print(f"  Tokens/correct = {s['tokens_per_correct']:.0f}")
    print(f"  Error rate = {s['error_rate']:.1%}")
    print(f"  Question types: {s['question_types']}")
    print(f"  Per-type EM:")
    for qt, em in sorted(s["per_type_em"].items()):
        print(f"    {qt}: {em:.3f}")


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------


def compare_experiments(
    experiments: list[str],
    datasets: list[str],
    tag: str = "full",
) -> None:
    """Print a comparison table across experiments and datasets."""
    header = ["Experiment"] + datasets + ["Mean"]
    rows = []

    for exp in experiments:
        row = [exp]
        ems = []
        for ds in datasets:
            rd = RESULTS_DIR / exp / ds / tag
            s = summarize(rd)
            if s:
                row.append(f"{s['em']:.3f}")
                ems.append(s["em"])
            else:
                row.append("—")
        row.append(f"{np.mean(ems):.3f}" if ems else "—")
        rows.append(row)

    # Print table
    col_widths = [
        max(len(str(row[i])) for row in [header] + rows)
        for i in range(len(header))
    ]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)

    print(f"\n{'='*60}")
    print("Experiment Comparison (EM)")
    print(f"{'='*60}")
    print(fmt.format(*header))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


# --------------------------------------------------------------------------
# Failure taxonomy
# --------------------------------------------------------------------------


def failure_taxonomy(results_dir: Path) -> None:
    """Classify failures into taxonomy categories."""
    preds = load_predictions(results_dir)
    if not preds:
        return

    categories = Counter()
    for p in preds:
        if exact_match(p["pred_answer"], p["gold_answer"]):
            categories["correct"] += 1
            continue

        error = p.get("error")
        if error:
            categories["pipeline_error"] += 1
            continue

        # Analyze agent results
        agent_results = p.get("agent_results", {})
        q_type = p.get("question_type", "unknown")

        if not agent_results:
            categories["no_agent_results"] += 1
            continue

        # Check for decomposition issues
        if q_type == "single_hop" and p.get("num_sub_questions", 0) > 1:
            categories["DE_unnecessary_decomposition"] += 1
            continue

        # Check retrieval misses per hop
        has_retrieval_miss = False
        for idx, ar in agent_results.items():
            if not ar.get("evidence_doc_ids") and ar.get("loops", 0) > 0:
                categories[f"RM_{idx}"] += 1
                has_retrieval_miss = True

        if has_retrieval_miss:
            continue

        # If agents found evidence but final answer is wrong →
        # aggregation synthesis failure
        has_evidence = any(
            ar.get("evidence_doc_ids") for ar in agent_results.values()
        )
        if has_evidence:
            categories["ASF_synthesis_failure"] += 1
        else:
            categories["AH_hallucination"] += 1

    print(f"\n{'='*60}")
    print(f"Failure Taxonomy: {results_dir}")
    print(f"{'='*60}")
    total = sum(categories.values())
    for cat, count in categories.most_common():
        print(f"  {cat}: {count} ({count/total:.1%})")


# --------------------------------------------------------------------------
# Bootstrap test
# --------------------------------------------------------------------------


def paired_bootstrap(
    preds_a: list[dict],
    preds_b: list[dict],
    n_resamples: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test on EM between two prediction sets."""
    rng = np.random.RandomState(seed)

    # Align by qid
    qids_a = {p.get("qid") or p.get("id"): p for p in preds_a}
    qids_b = {p.get("qid") or p.get("id"): p for p in preds_b}
    common_qids = sorted(set(qids_a.keys()) & set(qids_b.keys()))

    if not common_qids:
        return {"error": "No common questions"}

    em_a = np.array(
        [exact_match(qids_a[q]["pred_answer"], qids_a[q]["gold_answer"]) for q in common_qids],
        dtype=float,
    )
    em_b = np.array(
        [exact_match(qids_b[q]["pred_answer"], qids_b[q]["gold_answer"]) for q in common_qids],
        dtype=float,
    )

    observed_delta = em_b.mean() - em_a.mean()

    deltas = []
    n = len(common_qids)
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        deltas.append(em_b[idx].mean() - em_a[idx].mean())

    deltas = np.array(deltas)
    p_value = np.mean(deltas <= 0) if observed_delta > 0 else np.mean(deltas >= 0)

    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)

    return {
        "n_common": len(common_qids),
        "em_a": em_a.mean(),
        "em_b": em_b.mean(),
        "delta": observed_delta,
        "ci_95": (ci_lo, ci_hi),
        "p_value": p_value,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze MA²RAG results")
    parser.add_argument("--results", type=Path, help="Single results directory")
    parser.add_argument(
        "--compare", nargs="+", help="Experiment IDs to compare"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "musique", "2wiki"],
        help="Datasets for comparison",
    )
    parser.add_argument("--failure-taxonomy", type=Path, help="Run failure taxonomy")
    parser.add_argument("--tag", default="full", help="Result tag (pilot/full)")
    args = parser.parse_args()

    if args.results:
        print_summary(args.results)

    if args.compare:
        datasets = (
            ["hotpotqa", "musique", "2wiki"]
            if "all" in args.datasets
            else args.datasets
        )
        compare_experiments(args.compare, datasets, tag=args.tag)

    if args.failure_taxonomy:
        failure_taxonomy(args.failure_taxonomy)


if __name__ == "__main__":
    main()
