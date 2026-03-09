"""Compare v13k vs v13l predictions to find regressions."""
import json, re

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the|is|was|were|are)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())

for dataset in ["2wiki", "musique"]:
    print(f"\n{'='*60}")
    print(f"=== {dataset} regressions (v13k correct → v13l wrong) ===")
    print(f"{'='*60}")

    with open(f"results/m6v13k_pilot100/{dataset}/predictions.jsonl") as f:
        v13k = {json.loads(l)["question"]: json.loads(l) for l in f if l.strip()}
    with open(f"results/m6v13l_pilot100/{dataset}/predictions.jsonl") as f:
        v13l = {json.loads(l)["question"]: json.loads(l) for l in f if l.strip()}

    regressions = []
    improvements = []
    for q in v13k:
        if q not in v13l:
            continue
        k = v13k[q]
        l = v13l[q]
        k_correct = norm(k["pred_answer"]) == norm(k["gold_answer"])
        l_correct = norm(l["pred_answer"]) == norm(l["gold_answer"])

        if k_correct and not l_correct:
            regressions.append((q, k, l))
        elif not k_correct and l_correct:
            improvements.append((q, k, l))

    print(f"\nRegressions: {len(regressions)}, Improvements: {len(improvements)}")

    for q, k, l in regressions:
        print(f"\n  Q: {q[:90]}")
        print(f"  Gold: {k['gold_answer']}")
        print(f"  v13k pred: {k['pred_answer'][:60]} (CORRECT)")
        print(f"  v13l pred: {l['pred_answer'][:60]} (WRONG)")
        er_l = l.get("entity_registry", {})
        for key, val in er_l.items():
            if key.startswith("answer_"):
                print(f"    {key}: {val[:80]}")

    print(f"\n  Improvements:")
    for q, k, l in improvements:
        print(f"  Q: {q[:90]}")
        print(f"  Gold: {k['gold_answer']}, v13k: {k['pred_answer'][:40]}, v13l: {l['pred_answer'][:40]}")
