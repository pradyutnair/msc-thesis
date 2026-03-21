import json, re, string, unicodedata, os

def norm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_f1(pred, gold):
    pt = set(norm(pred).split()); gt = set(norm(gold).split())
    if not pt or not gt: return 0.0
    c = pt & gt
    if not c: return 0.0
    p = len(c)/len(pt); r = len(c)/len(gt)
    return 2*p*r/(p+r)

def contain_bi(pred, gold):
    np_ = norm(pred); ng = norm(gold)
    return 1 if (ng in np_ or np_ in ng) else 0

def load(path):
    preds = []
    with open(path) as fh:
        for line in fh:
            try: preds.append(json.loads(line.strip()))
            except: continue
    return preds

def ev(preds):
    n = len(preds)
    em = sum(1 for p in preds if norm(str(p.get("pred_answer","") or "")) == norm(str(p.get("gold_answer",""))))/n*100
    f1 = sum(token_f1(str(p.get("pred_answer","") or ""), str(p.get("gold_answer",""))) for p in preds)/n*100
    co = sum(contain_bi(str(p.get("pred_answer","") or ""), str(p.get("gold_answer",""))) for p in preds)/n*100
    tok = sum(p.get("total_tokens", 0) for p in preds)/n
    return em, f1, co, tok, n

systems = [
    ("m6v23 (warm)", "/projects/prjs1800/msc-thesis/02-arag-multi-agent/results/m6v23_pilot100", ["hotpotqa","2wiki","musique"]),
    ("m6v23 (no-warm)", "/projects/prjs1800/msc-thesis/02-arag-multi-agent/results/m6v23_nowarm_pilot100", ["hotpotqa","2wiki","musique"]),
    ("AA-v2 (fixed)", "/projects/prjs1800/msc-thesis/05-adaptive-agency/results_qwen3/aa_v2_pilot100", ["hotpotqa","2wikimultihop","musique"]),
]

print("+" + "-"*16 + "+" + "-"*26 + "+" + "-"*26 + "+" + "-"*26 + "+" + "-"*26 + "+")
print("| {:14s} | {:^24s} | {:^24s} | {:^24s} | {:^24s} |".format("System", "HotpotQA", "2Wiki", "MuSiQue", "MEAN"))
print("| {:14s} | {:>7s} {:>7s} {:>7s} | {:>7s} {:>7s} {:>7s} | {:>7s} {:>7s} {:>7s} | {:>7s} {:>7s} {:>7s} |".format(
    "", "EM", "F1", "Cont", "EM", "F1", "Cont", "EM", "F1", "Cont", "EM", "F1", "Cont"))
print("+" + "-"*16 + "+" + "-"*26 + "+" + "-"*26 + "+" + "-"*26 + "+" + "-"*26 + "+")

for name, base, ds_list in systems:
    results = {}
    for ds in ds_list:
        path = f"{base}/{ds}/predictions.jsonl"
        if os.path.exists(path):
            preds = load(path)
            if preds:
                ds_key = "2wiki" if "2wiki" in ds else ds
                results[ds_key] = ev(preds)
    if len(results) == 3:
        h = results["hotpotqa"]; w = results["2wiki"]; m = results["musique"]
        me = (h[0]+w[0]+m[0])/3; mf = (h[1]+w[1]+m[1])/3; mc = (h[2]+w[2]+m[2])/3
        print("| {:14s} | {:>6.1f}% {:>6.1f}% {:>6.1f}% | {:>6.1f}% {:>6.1f}% {:>6.1f}% | {:>6.1f}% {:>6.1f}% {:>6.1f}% | {:>6.1f}% {:>6.1f}% {:>6.1f}% |".format(
            name, h[0],h[1],h[2], w[0],w[1],w[2], m[0],m[1],m[2], me,mf,mc))

print("+" + "-"*16 + "+" + "-"*26 + "+" + "-"*26 + "+" + "-"*26 + "+" + "-"*26 + "+")

# Token efficiency
print("\nTokens/question:")
for name, base, ds_list in systems:
    toks = []
    for ds in ds_list:
        path = f"{base}/{ds}/predictions.jsonl"
        if os.path.exists(path):
            preds = load(path)
            toks.extend([p.get("total_tokens",0) for p in preds])
    if toks:
        print(f"  {name}: {sum(toks)/len(toks):.0f} tokens/q")
