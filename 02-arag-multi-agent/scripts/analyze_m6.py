#!/usr/bin/env python3
"""M6-specific analysis: backtracking rate, per-hop accuracy, tick distribution, token efficiency.

Usage:
    python scripts/analyze_m6.py --results results/m6/hotpotqa/full
    python scripts/analyze_m6.py --compare results/m6/hotpotqa/full results/m5/hotpotqa/full
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return predictions


def analyze_results(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute M6-specific metrics."""
    n = len(predictions)
    if n == 0:
        return {"error": "No predictions"}

    # Basic metrics
    em_scores = []
    f1_scores = []
    for p in predictions:
        pred = p.get("pred_answer", "")
        gold = p.get("gold_answer", "")
        em_scores.append(1.0 if exact_match(pred, gold) else 0.0)
        f1_scores.append(token_f1(pred, gold))

    em = sum(em_scores) / n * 100
    f1 = sum(f1_scores) / n * 100

    # M6-specific metrics
    ticks = [p.get("total_ticks", 0) for p in predictions]
    tokens = [p.get("total_tokens", 0) for p in predictions]
    times = [p.get("wall_clock_seconds", 0) for p in predictions]
    backtracks = [p.get("backtrack_count", 0) for p in predictions]
    num_sqs = [p.get("num_sub_questions", 0) for p in predictions]
    verified = [p.get("verified_count", 0) for p in predictions]
    failed = [p.get("failed_count", 0) for p in predictions]
    errors = [p for p in predictions if p.get("error")]

    # Termination reason distribution
    term_reasons = Counter()
    for p in predictions:
        reason = p.get("termination_reason", "unknown")
        if "SYNTHESIZED" in reason:
            term_reasons["SYNTHESIZED"] += 1
        elif "BUDGET" in reason:
            term_reasons["BUDGET_EXHAUSTED"] += 1
        elif "TIMEOUT" in reason:
            term_reasons["TIMEOUT"] += 1
        elif "MAX_TICKS" in reason:
            term_reasons["MAX_TICKS"] += 1
        else:
            term_reasons["other"] += 1

    # Backtracking analysis
    questions_with_backtrack = sum(1 for b in backtracks if b > 0)
    backtrack_rate = questions_with_backtrack / n * 100 if n > 0 else 0

    # EM by backtrack status
    em_with_bt = []
    em_without_bt = []
    for i, p in enumerate(predictions):
        if backtracks[i] > 0:
            em_with_bt.append(em_scores[i])
        else:
            em_without_bt.append(em_scores[i])

    # Per-hop accuracy (how many sub-Qs verified vs total)
    total_sqs_sum = sum(num_sqs)
    total_verified = sum(verified)
    total_failed = sum(failed)
    per_hop_accuracy = total_verified / total_sqs_sum * 100 if total_sqs_sum > 0 else 0

    def percentile(data: list, p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = min(f + 1, len(sorted_data) - 1)
        d = k - f
        return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])

    results = {
        "n": n,
        "em": em,
        "f1": f1,
        # Tick distribution
        "ticks_mean": sum(ticks) / n,
        "ticks_p50": percentile(ticks, 50),
        "ticks_p90": percentile(ticks, 90),
        # Token efficiency
        "tokens_mean": sum(tokens) / n,
        "tokens_p50": percentile(tokens, 50),
        "tokens_per_correct": (
            sum(tokens) / sum(em_scores) if sum(em_scores) > 0 else float("inf")
        ),
        # Latency
        "latency_mean": sum(times) / n,
        "latency_p50": percentile(times, 50),
        "latency_p90": percentile(times, 90),
        # Backtracking
        "backtrack_rate_pct": backtrack_rate,
        "backtrack_total": sum(backtracks),
        "em_with_backtrack": (
            sum(em_with_bt) / len(em_with_bt) * 100 if em_with_bt else 0
        ),
        "em_without_backtrack": (
            sum(em_without_bt) / len(em_without_bt) * 100 if em_without_bt else 0
        ),
        # Per-hop
        "per_hop_accuracy_pct": per_hop_accuracy,
        "total_sub_questions": total_sqs_sum,
        "total_verified": total_verified,
        "total_failed": total_failed,
        "mean_sub_questions": sum(num_sqs) / n,
        # Termination
        "termination_reasons": dict(term_reasons),
        # Errors
        "error_count": len(errors),
        "error_rate_pct": len(errors) / n * 100,
    }

    return results


def print_report(results: dict[str, Any], label: str = "M6") -> None:
    """Print formatted analysis report."""
    print(f"\n{'=' * 60}")
    print(f"  {label} Analysis Report (n={results['n']})")
    print(f"{'=' * 60}")

    print(f"\n  Accuracy:")
    print(f"    EM:  {results['em']:.1f}%")
    print(f"    F1:  {results['f1']:.1f}%")

    print(f"\n  Tick Distribution:")
    print(f"    Mean: {results['ticks_mean']:.1f}")
    print(f"    P50:  {results['ticks_p50']:.1f}")
    print(f"    P90:  {results['ticks_p90']:.1f}")

    print(f"\n  Token Efficiency:")
    print(f"    Mean tokens/Q:       {results['tokens_mean']:.0f}")
    print(f"    Tokens/correct:      {results['tokens_per_correct']:.0f}")

    print(f"\n  Latency:")
    print(f"    Mean: {results['latency_mean']:.1f}s")
    print(f"    P50:  {results['latency_p50']:.1f}s")
    print(f"    P90:  {results['latency_p90']:.1f}s")

    print(f"\n  Backtracking:")
    print(f"    Rate:           {results['backtrack_rate_pct']:.1f}% of questions")
    print(f"    Total events:   {results['backtrack_total']}")
    print(f"    EM w/ backtrack:  {results['em_with_backtrack']:.1f}%")
    print(f"    EM w/o backtrack: {results['em_without_backtrack']:.1f}%")

    print(f"\n  Per-Hop Accuracy:")
    print(f"    Verified/Total: {results['total_verified']}/{results['total_sub_questions']} ({results['per_hop_accuracy_pct']:.1f}%)")
    print(f"    Failed:         {results['total_failed']}")
    print(f"    Mean SQs/Q:     {results['mean_sub_questions']:.1f}")

    print(f"\n  Termination Reasons:")
    for reason, count in sorted(results["termination_reasons"].items()):
        print(f"    {reason}: {count}")

    if results["error_count"] > 0:
        print(f"\n  Errors: {results['error_count']} ({results['error_rate_pct']:.1f}%)")

    print(f"\n{'=' * 60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="M6 Analysis")
    parser.add_argument("--results", "-r", help="Results directory with predictions.jsonl")
    parser.add_argument("--compare", nargs="+", help="Multiple result dirs to compare")
    args = parser.parse_args()

    if args.results:
        pred_file = Path(args.results) / "predictions.jsonl"
        predictions = load_predictions(pred_file)
        results = analyze_results(predictions)
        print_report(results, label=Path(args.results).name)

        # Save analysis
        analysis_file = Path(args.results) / "m6_analysis.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Analysis saved to {analysis_file}")

    elif args.compare:
        print(f"\n{'Experiment':<30} {'N':>5} {'EM':>7} {'F1':>7} {'Ticks':>7} {'Tokens':>8} {'BT%':>6} {'Latency':>8}")
        print("-" * 85)
        for result_dir in args.compare:
            pred_file = Path(result_dir) / "predictions.jsonl"
            if not pred_file.exists():
                print(f"  {result_dir}: predictions.jsonl not found")
                continue
            predictions = load_predictions(pred_file)
            r = analyze_results(predictions)
            label = Path(result_dir).name
            print(
                f"  {label:<28} {r['n']:>5} {r['em']:>6.1f}% {r['f1']:>6.1f}% "
                f"{r['ticks_mean']:>6.1f} {r['tokens_mean']:>8.0f} "
                f"{r['backtrack_rate_pct']:>5.1f}% {r['latency_mean']:>7.1f}s"
            )
        print()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
