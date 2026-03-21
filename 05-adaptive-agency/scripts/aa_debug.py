import json, re, string, unicodedata

def norm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load(path):
    preds = []
    with open(path) as fh:
        for line in fh:
            try: preds.append(json.loads(line.strip()))
            except: continue
    return preds

m6 = {p["qid"]: p for p in load("/projects/prjs1800/msc-thesis/02-arag-multi-agent/results/m6v23_1000/hotpotqa/predictions.jsonl")}
aa = {p.get("qid",""): p for p in load("/projects/prjs1800/msc-thesis/05-adaptive-agency/results_qwen3/heuristic_1000q/hotpotqa/predictions.jsonl")}

count = 0
for qid in sorted(set(m6) & set(aa)):
    g = str(m6[qid].get("gold_answer",""))
    c_m6 = norm(str(m6[qid].get("pred_answer","") or "")) == norm(g)
    c_aa = norm(str(aa[qid].get("pred_answer","") or "")) == norm(g)
    if c_m6 and not c_aa:
        count += 1
        if count <= 10:
            q = m6[qid]["question"][:90]
            pred_m6 = m6[qid].get("pred_answer","")
            pred_aa = aa[qid].get("pred_answer","")
            tok_aa = aa[qid].get("total_tokens",0)
            nsq_aa = aa[qid].get("num_sub_questions",0)
            print(f"Q: {q}")
            print(f"  Gold: {g}")
            print(f"  m6v23: {pred_m6}")
            print(f"  AA:    {pred_aa}")
            print(f"  AA tokens={tok_aa}, sub_qs={nsq_aa}")
            print()
