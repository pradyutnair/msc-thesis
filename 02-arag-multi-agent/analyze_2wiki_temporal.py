"""Deep analysis of 2Wiki temporal/comparison failures.

Questions: How many are temporal comparisons? What patterns?
Why does M6 get them wrong? Is it decomposition, retrieval, or synthesis?
"""
import json, re
from collections import Counter

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the|is|was|were|are)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())

with open("results/m6v13k_pilot100/2wiki/predictions.jsonl") as f:
    preds = [json.loads(l) for l in f if l.strip()]

total = len(preds)
correct = sum(1 for p in preds if norm(p["pred_answer"]) == norm(p["gold_answer"]))
wrong = [p for p in preds if norm(p["pred_answer"]) != norm(p["gold_answer"])]

# Classify all 100 questions by type
temporal_patterns = [
    "born first", "born earlier", "born later", "died first", "died earlier", "died later",
    "lived longer", "is older", "is younger", "who is older", "who is younger",
    "which film has the director born", "which film has the director died",
    "which film has the director who was born", "which film has the director who died",
    "which film whose director was born", "which film whose director is",
    "director who is older", "director who is younger",
    "established first", "founded first", "formed first", "formed earlier",
]

comparison_patterns = [
    "same country", "same genre", "same profession",
    "are both", "do both", "have both",
]

q_types = Counter()
temporal_wrong = []
temporal_correct = []

for p in preds:
    q = p["question"].lower()
    is_correct = norm(p["pred_answer"]) == norm(p["gold_answer"])

    is_temporal = any(pat in q for pat in temporal_patterns)
    is_comparison = any(pat in q for pat in comparison_patterns)

    if is_temporal:
        q_types["temporal"] += 1
        if not is_correct:
            temporal_wrong.append(p)
        else:
            temporal_correct.append(p)
    elif is_comparison:
        q_types["comparison_yesno"] += 1
    else:
        q_types["other"] += 1

print(f"2Wiki v13k: {correct}/{total} correct ({correct*100/total:.0f}%)")
print(f"\nQuestion types:")
for t, c in q_types.most_common():
    print(f"  {t}: {c}")

print(f"\nTemporal questions: {q_types['temporal']} total")
print(f"  Correct: {len(temporal_correct)}")
print(f"  Wrong: {len(temporal_wrong)}")
print(f"  Accuracy: {len(temporal_correct)*100/(len(temporal_correct)+len(temporal_wrong)):.0f}%")

# Categorize temporal failures
print(f"\n=== Temporal failures ({len(temporal_wrong)}) ===")
for w in temporal_wrong:
    er = w.get("entity_registry", {})
    q = w["question"]
    gold = w["gold_answer"]
    pred = w["pred_answer"]

    # Check: did the worker find the right dates?
    er_answers = {k: v for k, v in er.items() if k.startswith("answer_")}

    # Check if prediction is a date (temporal confusion)
    pred_is_date = bool(re.search(r"\b\d{4}\b", pred)) or bool(re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b", pred))
    # Check if gold is in entity registry
    gold_in_er = any(gold.lower() in v.lower() for v in er.values())

    failure_type = "unknown"
    if pred_is_date:
        failure_type = "returned_date_not_entity"
    elif gold_in_er:
        failure_type = "synthesis_picked_wrong"
    else:
        failure_type = "wrong_retrieval"

    print(f"\n  [{failure_type}] Q: {q[:95]}")
    print(f"  Gold: {gold}")
    print(f"  Pred: {pred[:60]}")
    for k, v in er_answers.items():
        print(f"    {k}: {v[:70]}")

# Summary
date_returns = sum(1 for w in temporal_wrong
    if bool(re.search(r"\b\d{4}\b", w["pred_answer"])) or
       bool(re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b", w["pred_answer"])))

print(f"\n=== Summary ===")
print(f"Temporal wrong: {len(temporal_wrong)}")
print(f"  Returned date instead of entity: {date_returns}")
print(f"  Other: {len(temporal_wrong) - date_returns}")
print(f"\nThis means {date_returns} questions could be fixed by better entity extraction")
print(f"from _compare_by_date (currently returns date string, should return entity name)")
