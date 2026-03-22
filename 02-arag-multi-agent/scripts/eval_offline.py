#!/usr/bin/env python3
"""Offline eval: EM + contain-match. No LLM judge needed."""
import json, re, string, sys

def normalize_answer(s):
    if s is None: return ""
    if not isinstance(s, str): s = str(s)
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())

path = sys.argv[1]
preds = [json.loads(l) for l in open(path) if l.strip()]
total = len(preds)
em = cm = empty = 0
for p in preds:
    pred = normalize_answer(p.get("pred_answer", ""))
    gold = normalize_answer(p.get("gold_answer", ""))
    if not pred: empty += 1; continue
    if pred == gold: em += 1; cm += 1
    elif gold in pred or pred in gold: cm += 1

print(f"Total: {total}")
print(f"EM:      {em}/{total} = {em/total*100:.1f}%")
print(f"Contain: {cm}/{total} = {cm/total*100:.1f}%")
print(f"Empty:   {empty}/{total} = {empty/total*100:.1f}%")

# Write summary
import os
out_dir = os.path.dirname(path)
summary = {"total": total, "em": em, "em_pct": round(em/total*100,1), "contain": cm, "contain_pct": round(cm/total*100,1), "empty": empty}
with open(os.path.join(out_dir, "offline_eval_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {out_dir}/offline_eval_summary.json")
