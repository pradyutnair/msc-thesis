#!/usr/bin/env python3
"""
Lightweight EM/F1 evaluator for predictions.jsonl.

Why: external/arag/scripts/eval.py is LLM-judge oriented; this gives your thesis EM/F1.
"""
import argparse
import json
import re
import string
from pathlib import Path

def norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def em(pred: str, gold: str) -> float:
    return float(norm(pred) == norm(gold))

def f1(pred: str, gold: str) -> float:
    pt = norm(pred).split()
    gt = norm(gold).split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = {}
    for t in pt:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gt:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    p = overlap / len(pt)
    r = overlap / len(gt)
    return 2 * p * r / (p + r)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    rows = []
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    ems, f1s = [], []
    for r in rows:
        pred = r.get("pred_answer", "")
        gold = r.get("gold_answer", "") or r.get("answer", "")
        ems.append(em(pred, gold))
        f1s.append(f1(pred, gold))

    summary = {
        "count": len(rows),
        "em": sum(ems) / len(ems) if ems else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()