"""Compare v13l vs v13m to find what regressed on HotpotQA and what improved on 2Wiki."""
import json, re

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the|is|was|were|are)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())

for dataset in ["hotpotqa", "2wiki"]:
    print(f"\n{'='*60}")
    print(f"=== {dataset}: v13l → v13m changes ===")

    with open(f"results/m6v13l_pilot100/{dataset}/predictions.jsonl") as f:
        v13l = {json.loads(l)["question"]: json.loads(l) for l in f if l.strip()}
    with open(f"results/m6v13m_pilot100/{dataset}/predictions.jsonl") as f:
        v13m = {json.loads(l)["question"]: json.loads(l) for l in f if l.strip()}

    regressions = []
    improvements = []
    for q in v13l:
        if q not in v13m:
            continue
        l = v13l[q]
        m = v13m[q]
        l_correct = norm(l["pred_answer"]) == norm(l["gold_answer"])
        m_correct = norm(m["pred_answer"]) == norm(m["gold_answer"])
        if l_correct and not m_correct:
            regressions.append((q, l, m))
        elif not l_correct and m_correct:
            improvements.append((q, l, m))

    print(f"Regressions: {len(regressions)}, Improvements: {len(improvements)}")
    print(f"\nRegressions:")
    for q, l, m in regressions[:10]:
        print(f"  Q: {q[:90]}")
        print(f"  Gold: {l['gold_answer']}")
        print(f"  v13l: {l['pred_answer'][:50]} (CORRECT), v13m: {m['pred_answer'][:50]} (WRONG)")
        er = m.get("entity_registry", {})
        for k, v in er.items():
            if k.startswith("answer_"):
                print(f"    {k}: {v[:60]}")
        print()

    print(f"Improvements:")
    for q, l, m in improvements[:10]:
        print(f"  Q: {q[:90]}")
        print(f"  Gold: {l['gold_answer']}, v13l: {l['pred_answer'][:40]}, v13m: {m['pred_answer'][:40]}")
