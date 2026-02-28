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

e4_right_v4_wrong = []
for qid in v4:
    e4d = e4.get(qid, {})
    v4d = v4[qid]
    if e4d.get("llm_accuracy", 0) > 0 and v4d.get("llm_accuracy", 0) == 0:
        e4_right_v4_wrong.append(qid)

near_miss_count = 0
real_wrong_count = 0
print("E4 right, V4 wrong:", len(e4_right_v4_wrong))
print()
for qid in e4_right_v4_wrong:
    v4d = v4[qid]
    gold = str(v4d.get("gold_answer", ""))
    pred = str(v4d.get("pred_answer", ""))
    q = v4d.get("question", "")
    near = gold.lower() in pred.lower() or pred.lower() in gold.lower()
    if near:
        near_miss_count += 1
        tag = "NEAR-MISS"
    else:
        real_wrong_count += 1
        tag = "WRONG"
    print("[%s] Q: %s" % (tag, q[:80]))
    print("  Gold: %s" % gold)
    print("  Pred: %s" % pred[:80])
    loops = v4d.get("loops", 0)
    chunks = v4d.get("chunks_read_count", 0)
    print("  Loops: %d, Chunks: %d" % (loops, chunks))
    print()

print("Near-misses: %d, Real wrong: %d" % (near_miss_count, real_wrong_count))
print("If near-misses counted: V4 = %d/100 = %d%%" % (61 + near_miss_count, 61 + near_miss_count))
