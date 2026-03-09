import json, re

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the|is|was|were|are)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())

with open("results/m6v13k_pilot100/hotpotqa/predictions.jsonl") as f:
    preds = [json.loads(l) for l in f if l.strip()]

wrong = [p for p in preds if norm(p["pred_answer"]) != norm(p["gold_answer"])]

# Filter to genuinely wrong (exclude format issues)
genuine = []
for w in wrong:
    pred_n = norm(w["pred_answer"])
    gold_n = norm(w["gold_answer"])
    if gold_n in ("yes", "no") or pred_n in ("yes", "no"):
        continue
    if pred_n in gold_n or gold_n in pred_n:
        continue
    pred_toks = set(pred_n.split())
    gold_toks = set(gold_n.split())
    if pred_toks and gold_toks and len(pred_toks & gold_toks) / max(len(pred_toks), len(gold_toks)) >= 0.5:
        continue
    genuine.append(w)

# Categorize
verbose = []
wrong_synth = []
wrong_retrieval = []

for w in genuine:
    er = w.get("entity_registry", {})
    gold = w["gold_answer"].lower()

    gold_in_er = any(gold in v.lower() for v in er.values())
    has_verbose = any(len(v) > 60 for v in er.values())
    has_refusal = any(
        "evidence" in v.lower() or "does not" in v.lower() or "not mention" in v.lower()
        for v in er.values()
    )

    if gold_in_er:
        wrong_synth.append(w)
    elif has_refusal or has_verbose:
        verbose.append(w)
    else:
        wrong_retrieval.append(w)

print(f"Total genuinely wrong: {len(genuine)}")
print(f"  wrong_synthesis: {len(wrong_synth)} (gold in ER)")
print(f"  verbose_answer: {len(verbose)} (ER has verbose/refusal)")
print(f"  wrong_retrieval: {len(wrong_retrieval)} (workers found wrong info)")

print("\n=== wrong_synthesis ===")
for w in wrong_synth:
    er = w.get("entity_registry", {})
    print(f"  Q: {w['question'][:90]}")
    print(f"  Gold: {w['gold_answer']}, Pred: {w['pred_answer'][:50]}")
    for k, v in er.items():
        if k.startswith("answer_"):
            print(f"    {k}: {v[:80]}")
    print()

print("\n=== verbose_answer ===")
for w in verbose:
    er = w.get("entity_registry", {})
    print(f"  Q: {w['question'][:90]}")
    print(f"  Gold: {w['gold_answer']}, Pred: {w['pred_answer'][:60]}")
    for k, v in er.items():
        if k.startswith("answer_"):
            print(f"    {k}: {v[:100]}")
    print()

print("\n=== wrong_retrieval (sample 10) ===")
for w in wrong_retrieval[:10]:
    er = w.get("entity_registry", {})
    print(f"  Q: {w['question'][:90]}")
    print(f"  Gold: {w['gold_answer']}, Pred: {w['pred_answer'][:50]}")
    for k, v in er.items():
        if k.startswith("answer_"):
            print(f"    {k}: {v[:80]}")
    print()
