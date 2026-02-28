import json

v5 = {}
with open("results/m5_pilot100/hotpotqa/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        v5[d["qid"]] = d

e4 = {}
with open("/projects/prjs1800/msc-thesis/01-arag-reproduction/results/b0-qwen3-30b-e5-deepseekr1/hotpotqa/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        qid = d.get("qid", d.get("id"))
        if qid in v5:
            e4[qid] = d

# E4 right, V5 wrong
e4w = []
v5w = []
for qid in v5:
    e4d = e4.get(qid, {})
    v5d = v5[qid]
    e4_ok = e4d.get("llm_accuracy", 0) > 0
    v5_ok = v5d.get("llm_accuracy", 0) > 0
    if e4_ok and not v5_ok:
        gold = str(v5d.get("gold_answer", "")).lower()
        pred = str(v5d.get("pred_answer", "")).lower()
        near = gold in pred or pred in gold
        e4w.append((qid, near))
    elif v5_ok and not e4_ok:
        v5w.append(qid)

near_count = sum(1 for _, n in e4w if n)
real_count = sum(1 for _, n in e4w if not n)

print("E4 right, V5 wrong:", len(e4w), "(near-miss:", near_count, ", real:", real_count, ")")
print("V5 right, E4 wrong:", len(v5w))
print()

# Show remaining near-misses
for qid, near in e4w:
    v5d = v5[qid]
    gold = str(v5d.get("gold_answer", ""))
    pred = str(v5d.get("pred_answer", ""))
    tag = "NEAR" if near else "WRONG"
    print("[%s] Gold: %-35s Pred: %s" % (tag, gold, pred[:60]))

print()
print("Summary: V5 = 64%, E4 = 64% (tied)")
print("Net swaps: V5 gained %d, lost %d vs E4" % (len(v5w), len(e4w)))
