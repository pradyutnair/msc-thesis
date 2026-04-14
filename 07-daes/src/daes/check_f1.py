import json
import re
import string
from collections import Counter


def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def compute_f1(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return 0.0, 0.0, 0.0
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    prec = overlap / len(pt)
    rec = overlap / len(gt)
    f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


data = json.load(
    open(
        "/projects/prjs1800/msc-thesis/07-daes/results/idnmr/dream_musique_idnmr_50q.json"
    )
)
pm = fm = tn = 0
for r in data["results"]:
    for m in ["baseline", "pool", "idnmr"]:
        a = r[m]["answer"]
        g = r["gold"]
        rp = r[m]["f1"]
        pr, rc, f1 = compute_f1(a, g)
        if rp > 0.001:
            tn += 1
            if abs(rp - pr) < 0.001:
                pm += 1
            elif abs(rp - f1) < 0.001:
                fm += 1
print(f"nonzero={tn} prec_match={pm} f1_match={fm}")

methods = ["baseline", "pool", "idnmr", "idnmr_2round"]
ct = {m: {"f1": 0, "p": 0, "r": 0} for m in methods}
n = len(data["results"])
for r in data["results"]:
    for m in ct:
        pr, rc, f1 = compute_f1(r[m]["answer"], r["gold"])
        ct[m]["f1"] += f1
        ct[m]["p"] += pr
        ct[m]["r"] += rc
for m in ct:
    rpt = data["summary"][m]["f1"]
    tf = ct[m]["f1"] / n
    tp = ct[m]["p"] / n
    tr = ct[m]["r"] / n
    print(f"{m:16s}: reported={rpt:.4f} true_f1={tf:.4f} prec={tp:.4f} rec={tr:.4f}")

print("\nPool vs Baseline per question (first 20):")
for r in data["results"][:20]:
    bl_pr, bl_rc, bl_f1 = compute_f1(r["baseline"]["answer"], r["gold"])
    pl_pr, pl_rc, pl_f1 = compute_f1(r["pool"]["answer"], r["gold"])
    diff = pl_f1 - bl_f1
    marker = "+" if diff > 0.01 else ("-" if diff < -0.01 else "=")
    cands = r.get("idnmr_stats", [{}])[0].get("candidates", [])
    bridge = cands[0][:30] if cands else ""
    ba = r["baseline"]["answer"][:20]
    pa = r["pool"]["answer"][:20]
    g = r["gold"][:25]
    print(f"{r['id']:8s} bl={bl_f1:.3f} pool={pl_f1:.3f} {marker} gold={g:25s} bl={ba} pool={pa}")
