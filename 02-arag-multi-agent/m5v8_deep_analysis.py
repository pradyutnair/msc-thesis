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

# === PART 1: Overall tool usage comparison ===
print("=" * 70)
print("TOOL USAGE COMPARISON (100 questions)")
print("=" * 70)

# E4 tool usage
e4_tools = {}
e4_total_tools = 0
for qid, d in e4.items():
    for t in d.get("trajectory", []):
        name = t.get("tool_name", "unknown")
        e4_tools[name] = e4_tools.get(name, 0) + 1
        e4_total_tools += 1

# M5 tool usage
m5_tools = {}
m5_total_tools = 0
for qid, d in m5.items():
    for t in d.get("trajectory", []):
        name = t.get("tool_name", "unknown")
        m5_tools[name] = m5_tools.get(name, 0) + 1
        m5_total_tools += 1

print("\nE4 tools (total %d calls):" % e4_total_tools)
for k, v in sorted(e4_tools.items(), key=lambda x: -x[1]):
    print("  %-20s %d (%.0f%%)" % (k, v, 100*v/e4_total_tools))

print("\nM5v8 tools (total %d calls):" % m5_total_tools)
for k, v in sorted(m5_tools.items(), key=lambda x: -x[1]):
    print("  %-20s %d (%.0f%%)" % (k, v, 100*v/m5_total_tools))

# E4 chunks read stats
e4_chunks = [d.get("chunks_read_count", 0) for d in e4.values()]
m5_chunks = [d.get("chunks_read_count", 0) for d in m5.values()]
print("\nChunks read: E4 avg=%.1f, M5 avg=%.1f" % (sum(e4_chunks)/len(e4_chunks), sum(m5_chunks)/len(m5_chunks)))
print("E4 questions with 0 chunks: %d, M5 questions with 0 chunks: %d" % (
    sum(1 for c in e4_chunks if c == 0), sum(1 for c in m5_chunks if c == 0)))

e4_loops = [d.get("loops", 0) for d in e4.values()]
m5_loops = [d.get("loops", 0) for d in m5.values()]
print("Avg loops: E4=%.1f, M5=%.1f" % (sum(e4_loops)/len(e4_loops), sum(m5_loops)/len(m5_loops)))

# === PART 2: E4-right M5-wrong deep dive ===
print("\n" + "=" * 70)
print("E4 RIGHT, M5v8 WRONG — FAILURE CATEGORIZATION")
print("=" * 70)

categories = {"no_chunks_wrong_answer": [], "has_chunks_wrong_answer": [], "near_miss": [], "search_fail": []}

for qid in m5:
    e4d = e4.get(qid, {})
    m5d = m5[qid]
    if e4d.get("llm_accuracy", 0) > 0 and m5d.get("llm_accuracy", 0) == 0:
        gold = str(m5d.get("gold_answer", "")).lower()
        pred = str(m5d.get("pred_answer", "")).lower()
        chunks = m5d.get("chunks_read_count", 0)
        near = gold in pred or pred in gold
        
        traj = m5d.get("trajectory", [])
        searches = [t for t in traj if t.get("tool_name") in ("keyword_agent", "semantic_agent")]
        no_results = sum(1 for t in searches if "No results" in str(t.get("tool_result", ""))[:50] or "no relevant" in str(t.get("tool_result", "")).lower()[:100])
        
        if near:
            categories["near_miss"].append(qid)
        elif chunks == 0 and no_results == len(searches) and len(searches) > 0:
            categories["search_fail"].append(qid)
        elif chunks == 0:
            categories["no_chunks_wrong_answer"].append(qid)
        else:
            categories["has_chunks_wrong_answer"].append(qid)

print("\nNear-miss (answer close but judged wrong): %d" % len(categories["near_miss"]))
print("Search failure (all searches returned nothing): %d" % len(categories["search_fail"]))
print("No chunks read but got wrong answer: %d" % len(categories["no_chunks_wrong_answer"]))
print("Read chunks but still wrong answer: %d" % len(categories["has_chunks_wrong_answer"]))

# === PART 3: Detailed trace of failures ===
print("\n" + "=" * 70)
print("DETAILED TRACES — E4 RIGHT, M5v8 WRONG")
print("=" * 70)

for qid in m5:
    e4d = e4.get(qid, {})
    m5d = m5[qid]
    if e4d.get("llm_accuracy", 0) > 0 and m5d.get("llm_accuracy", 0) == 0:
        gold = str(m5d.get("gold_answer", ""))
        m5_pred = str(m5d.get("pred_answer", ""))
        e4_pred = str(e4d.get("pred_answer", ""))
        
        # E4 trajectory
        e4_traj = e4d.get("trajectory", [])
        e4_tool_seq = []
        for t in e4_traj:
            name = t.get("tool_name", "?")
            args = t.get("arguments", {})
            if name == "keyword_search":
                e4_tool_seq.append("kw(%s)" % str(args.get("keywords", []))[:40])
            elif name == "semantic_search":
                e4_tool_seq.append("sem(%s)" % str(args.get("query", ""))[:40])
            elif name == "read_chunk":
                e4_tool_seq.append("read(%s)" % str(args.get("chunk_ids", []))[:20])
            else:
                e4_tool_seq.append(name)
        
        # M5 trajectory
        m5_traj = m5d.get("trajectory", [])
        m5_tool_seq = []
        for t in m5_traj:
            name = t.get("tool_name", "?")
            args = t.get("arguments", {})
            if name == "keyword_agent":
                m5_tool_seq.append("kw_agent(%s)" % str(args.get("task", ""))[:40])
            elif name == "semantic_agent":
                m5_tool_seq.append("sem_agent(%s)" % str(args.get("task", ""))[:40])
            elif name == "chunk_reader":
                m5_tool_seq.append("reader(%s)" % str(args.get("chunk_ids", ""))[:20])
            elif name == "finish":
                m5_tool_seq.append("finish")
            else:
                m5_tool_seq.append(name)
        
        print("\nQ: %s" % m5d.get("question", "")[:90])
        print("  Gold: %s" % gold)
        print("  E4:   %s" % e4_pred[:60])
        print("  M5v8: %s" % m5_pred[:60])
        print("  E4 tools (%d): %s" % (len(e4_traj), " -> ".join(e4_tool_seq)))
        print("  M5 tools (%d): %s" % (len(m5_traj), " -> ".join(m5_tool_seq)))
        print("  M5 loops: %d, chunks: %d" % (m5d.get("loops", 0), m5d.get("chunks_read_count", 0)))
