import json

m5 = {}
with open("results/m5v8_p100/hotpotqa/predictions.jsonl") as f:
    for line in f:
        d = json.loads(line)
        m5[d["qid"]] = d

e4 = {}
with open("/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-30b-e5-deepseekr1/hotpotqa/predictions.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 100: break
        d = json.loads(line)
        qid = d.get("qid", d.get("id"))
        if qid in m5:
            e4[qid] = d

e4_right_m5_wrong = []
m5_right_e4_wrong = []
for qid in m5:
    e4d = e4.get(qid, {})
    m5d = m5[qid]
    e4_ok = e4d.get("llm_accuracy", 0) > 0
    m5_ok = m5d.get("llm_accuracy", 0) > 0
    if e4_ok and not m5_ok:
        e4_right_m5_wrong.append(qid)
    elif m5_ok and not e4_ok:
        m5_right_e4_wrong.append(qid)

print("E4 right, M5v8 wrong:", len(e4_right_m5_wrong))
print("M5v8 right, E4 wrong:", len(m5_right_e4_wrong))
print()

for qid in e4_right_m5_wrong:
    m5d = m5[qid]
    e4d = e4[qid]
    gold = str(m5d.get("gold_answer", ""))
    m5_pred = str(m5d.get("pred_answer", ""))
    e4_pred = str(e4d.get("pred_answer", ""))
    loops = m5d.get("loops", 0)
    chunks = m5d.get("chunks_read_count", 0)
    near = gold.lower() in m5_pred.lower() or m5_pred.lower() in gold.lower()
    tag = "NEAR" if near else "WRONG"
    print("[%s] Q: %s" % (tag, m5d.get("question", "")[:80]))
    print("  Gold: %s" % gold)
    print("  E4:   %s" % e4_pred[:80])
    print("  M5v8: %s" % m5_pred[:80])
    print("  Loops: %d, Chunks: %d" % (loops, chunks))
    print()
