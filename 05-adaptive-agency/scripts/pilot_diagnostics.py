#!/usr/bin/env python3
"""Analyze pilot results and print diagnostic verdict."""

import json
import sys


def main():
    print("=== COUNTERFACTUAL SIGNAL ===")
    try:
        with open("trajectories/pilot_hotpotqa/escalation_trajectories.jsonl") as f:
            trajs = [json.loads(l) for l in f if l.strip()]

        total_sqs = 0
        accept = escalate = both_failed = 0
        agentic_only_correct = 0
        structured_only_correct = 0
        both_have_answer = 0

        for t in trajs:
            for sq in t["sub_question_results"]:
                if sq.get("skipped"):
                    continue
                total_sqs += 1
                esc = sq.get("escalation", {})
                label = esc.get("label", "")
                if label == "ACCEPT":
                    accept += 1
                elif label == "ESCALATE":
                    escalate += 1
                elif label == "BOTH_FAILED":
                    both_failed += 1

                s = sq.get("structured", {}).get("answer", "")
                a = sq.get("agentic", {}).get("answer", "")
                s_ok = bool(s) and s.lower() not in ("unknown", "error", "", "none", "n/a")
                a_ok = bool(a) and a.lower() not in ("unknown", "error", "", "none", "n/a")
                if s_ok and a_ok:
                    both_have_answer += 1
                elif a_ok and not s_ok:
                    agentic_only_correct += 1
                elif s_ok and not a_ok:
                    structured_only_correct += 1

        print(f"  Questions: {len(trajs)}")
        print(f"  Sub-questions: {total_sqs}")
        print(f"  ACCEPT (structured suffices): {accept} ({accept/max(total_sqs,1)*100:.0f}%)")
        print(f"  ESCALATE (agentic needed):    {escalate} ({escalate/max(total_sqs,1)*100:.0f}%)")
        print(f"  BOTH_FAILED:                  {both_failed} ({both_failed/max(total_sqs,1)*100:.0f}%)")
        print(f"  Both have answers:            {both_have_answer}")
        print(f"  Agentic-only correct:         {agentic_only_correct}")
        print(f"  Structured-only correct:      {structured_only_correct}")
        print()

        s_tokens = sum(t["total_structured_tokens"] for t in trajs)
        a_tokens = sum(t["total_agentic_tokens"] for t in trajs)
        print(f"  Avg structured tokens/q: {s_tokens/max(len(trajs),1):.0f}")
        print(f"  Avg agentic tokens/q:    {a_tokens/max(len(trajs),1):.0f}")
        if a_tokens > 0:
            print(f"  Potential savings:       {(a_tokens-s_tokens)/a_tokens*100:.0f}%")
    except Exception as e:
        print(f"  Error reading counterfactual data: {e}")
        total_sqs = 0
        accept = escalate = both_failed = 0

    print()
    print("=== HEURISTIC BASELINE ===")
    summary = None
    try:
        with open("results/pilot_heuristic/hotpotqa/offline_eval_summary.json") as f:
            summary = json.load(f)
        print(f"  EM:              {summary['norm_em']*100:.1f}%")
        print(f"  F1:              {summary['token_f1']*100:.1f}%")
        print(f"  Avg tokens/q:    {summary['avg_tokens']}")
        print(f"  Escalation rate: {summary['escalation_rate']*100:.1f}%")
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=== VERDICT ===")
    issues = []

    if total_sqs > 0:
        esc_rate = escalate / total_sqs
        fail_rate = both_failed / total_sqs
        if esc_rate < 0.1:
            issues.append(f"Escalation rate {esc_rate*100:.0f}% < 10% — structured almost always suffices, escalation story is weak")
        elif esc_rate > 0.8:
            issues.append(f"Escalation rate {esc_rate*100:.0f}% > 80% — structured rarely works, pipeline is expensive")
        else:
            print(f"  OK: Escalation rate {esc_rate*100:.0f}% — healthy mix, escalation training viable")

        if fail_rate > 0.4:
            issues.append(f"Both-failed rate {fail_rate*100:.0f}% > 40% — retrieval may be broken")
        else:
            print(f"  OK: Both-failed rate {fail_rate*100:.0f}%")

    if summary:
        em = summary["norm_em"]
        if em < 0.15:
            issues.append(f"EM {em*100:.0f}% is very low — something may be broken")
        elif em < 0.25:
            issues.append(f"EM {em*100:.0f}% is below expected ~35% for HotPotQA — check retrieval quality")
        else:
            print(f"  OK: EM {em*100:.0f}% — in expected range for HotPotQA")

    if issues:
        print()
        for issue in issues:
            print(f"  WARNING: {issue}")
        print()
        print("  >>> FIX ISSUES BEFORE PROCEEDING TO FULL EXPERIMENTS <<<")
    else:
        print()
        print("  >>> ALL CHECKS PASSED — READY FOR FULL EXPERIMENTS <<<")


if __name__ == "__main__":
    main()
