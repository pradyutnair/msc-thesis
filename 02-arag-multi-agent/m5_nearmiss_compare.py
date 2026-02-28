import json

v4 = {}
with open("results/m5_pilot100/hotpotqa/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        v4[d["qid"]] = d

e4 = {}
with open("/projects/prjs1800/msc-thesis/01-arag-reproduction/results/b0-qwen3-30b-e5-deepseekr1/hotpotqa/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        qid = d.get("qid", d.get("id"))
        if qid in v4:
            e4[qid] = d

# Near-miss questions where M5's answer contains or is contained in gold
print("=== NEAR-MISS COMPARISON: E4 vs M5 ===")
print()
for qid in v4:
    e4d = e4.get(qid, {})
    v4d = v4[qid]
    if e4d.get("llm_accuracy", 0) > 0 and v4d.get("llm_accuracy", 0) == 0:
        gold = str(v4d.get("gold_answer", "")).lower()
        pred = str(v4d.get("pred_answer", "")).lower()
        if gold in pred or pred in gold:
            e4_pred = str(e4d.get("pred_answer", ""))
            m5_pred = str(v4d.get("pred_answer", ""))
            gold_orig = str(v4d.get("gold_answer", ""))
            print("Q:", v4d.get("question", "")[:90])
            print("  Gold:", gold_orig)
            print("  E4:  ", e4_pred[:80], "  [CORRECT by judge]")
            print("  M5:  ", m5_pred[:80], "  [INCORRECT by judge]")
            print()

# Also check: does E4 have similar near-misses that get marked correct?
print("=== E4 answers that are substring matches with gold ===")
e4_nearmiss = 0
for qid in v4:
    e4d = e4.get(qid, {})
    if e4d.get("llm_accuracy", 0) > 0:
        gold = str(e4d.get("gold_answer", v4[qid].get("gold_answer", ""))).lower()
        pred = str(e4d.get("pred_answer", "")).lower()
        if (gold in pred or pred in gold) and gold != pred:
            e4_nearmiss += 1
print("E4 correct answers that are substring matches:", e4_nearmiss)
