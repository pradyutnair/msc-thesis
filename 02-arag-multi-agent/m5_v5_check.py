import json

v5 = {}
with open("results/m5_pilot100/hotpotqa/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        v5[d["qid"]] = d

v4 = {}
with open("results/m5_pilot100/hotpotqa/predictions_v4.jsonl") as f:
    for line in f:
        d = json.loads(line)
        v4[d["qid"]] = d

# Check the 6 near-miss questions from v4 - did v5 fix them?
# Near-miss gold answers from v4 analysis
nearmiss_golds = [
    "the Queen's gaoler",
    "13 seasons", 
    "John John Florence",
    "Broadcasting House in London",
    "director",
    "Karakoram mountain range"
]

print("=== V4 NEAR-MISS QUESTIONS: V5 ANSWERS ===")
print()
for qid in v5:
    gold = str(v5[qid].get("gold_answer", ""))
    if gold in nearmiss_golds:
        v4_pred = str(v4.get(qid, {}).get("pred_answer", ""))
        v5_pred = str(v5[qid].get("pred_answer", ""))
        q = v5[qid].get("question", "")
        print("Q:", q[:80])
        print("  Gold:", gold)
        print("  V4:  ", v4_pred[:80])
        print("  V5:  ", v5_pred[:80])
        match = gold.lower() in v5_pred.lower() or v5_pred.lower() in gold.lower()
        exact = gold.lower().strip() == v5_pred.lower().strip()
        print("  Match:", "EXACT" if exact else ("NEAR" if match else "NO"))
        print()

# Quick overall stats
loops = [d["loops"] for d in v5.values()]
chunks = [d.get("chunks_read_count", 0) for d in v5.values()]
print("V5 stats: avg loops %.1f, avg chunks %.1f" % (sum(loops)/len(loops), sum(chunks)/len(chunks)))
