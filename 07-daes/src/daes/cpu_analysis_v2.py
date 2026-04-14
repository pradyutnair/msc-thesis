"""
CPU-only analysis: significance tests, per-hop, answer lengths, error categorization.
Uses numpy for fast bootstrap.
"""
import json, os, sys
import numpy as np
from collections import defaultdict

RESULTS_BASE = "/projects/prjs1800/msc-thesis/07-daes/results"
JUDGE_DIR    = f"{RESULTS_BASE}/llm_judge"
MUSIQUE_ORIG = "/projects/prjs1800/datasets/musique/musique_ans_v1.0_dev.jsonl"
OUTPUT_DIR   = f"{RESULTS_BASE}/cpu_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────────

def load_judge(dataset, model="llada"):
    prefix = "" if model == "llada" else "dream_"
    path = f"{JUDGE_DIR}/llm_judge_{prefix}{dataset}.json"
    return json.load(open(path))

def paired_bootstrap_np(a, b, n=10000, seed=42):
    """Vectorised paired bootstrap. Returns (observed_diff, p_value) for H0: a<=b."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    obs = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    p = (diffs <= 0).mean()
    return float(obs), float(p)

def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

DATASETS  = ["musique", "hotpotqa", "2wikimultihopqa"]
METHODS_9 = ["baseline","spread","aram","ispread","iaram","pool","ipool","idnmr","idnmr_2round"]

# ── Task 1: Significance tests ───────────────────────────────────────────────

def task_significance():
    print("\n" + "="*70)
    print("TASK 1: SIGNIFICANCE TESTS (paired bootstrap, N=10000)")
    print("="*70)
    comparisons = [("idnmr","ispread"),("idnmr","iaram"),("pool","ispread"),("pool","iaram")]
    out = {}
    for model in ["llada","dream"]:
        out[model] = {}
        print(f"\n── {model.upper()} ──")
        for dataset in DATASETS:
            try: data = load_judge(dataset, model)
            except FileNotFoundError: print(f"  {dataset}: MISSING"); continue
            out[model][dataset] = {}
            print(f"\n  {dataset} (N={len(data)})")
            for ma, mb in comparisons:
                ja, jb, fa, fb = [], [], [], []
                for item in data:
                    m = item["methods"]
                    if ma not in m or mb not in m: continue
                    ja.append(1 if m[ma].get("judge")=="correct" else 0)
                    jb.append(1 if m[mb].get("judge")=="correct" else 0)
                    fa.append(m[ma].get("extracted_f1") or 0)
                    fb.append(m[mb].get("extracted_f1") or 0)
                if not ja: continue
                dj, pj = paired_bootstrap_np(ja, jb)
                df, pf = paired_bootstrap_np(fa, fb)
                print(f"    {ma:8s} vs {mb:8s} | "
                      f"Judge Δ={dj*100:+.1f}pp p={pj:.4f}{sig_stars(pj):3s} | "
                      f"ExtF1 Δ={df*100:+.2f}pp p={pf:.4f}{sig_stars(pf):3s}  "
                      f"[{sum(ja)/len(ja)*100:.1f}% vs {sum(jb)/len(jb)*100:.1f}%]")
                out[model][dataset][f"{ma}_vs_{mb}"] = {
                    "n": len(ja),
                    "judge_a_pct": round(sum(ja)/len(ja)*100,2),
                    "judge_b_pct": round(sum(jb)/len(jb)*100,2),
                    "diff_judge_pp": round(dj*100,3), "p_judge": round(pj,5), "sig_judge": sig_stars(pj),
                    "diff_f1_pp":   round(df*100,3), "p_f1":    round(pf,5), "sig_f1":   sig_stars(pf),
                }
    json.dump(out, open(f"{OUTPUT_DIR}/significance_tests.json","w"), indent=2)
    print(f"\n  → {OUTPUT_DIR}/significance_tests.json")
    return out

# ── Task 2: Per-hop analysis ─────────────────────────────────────────────────

def task_per_hop():
    print("\n" + "="*70)
    print("TASK 2: PER-HOP ANALYSIS (MuSiQue)")
    print("="*70)
    # Build question→hops map from original MuSiQue (hop count in ID: "2hop__...")
    hop_map = {}
    for line in open(MUSIQUE_ORIG):
        item = json.loads(line)
        qid = item["id"]
        nh = int(qid.split("hop__")[0]) if "hop__" in qid else None
        hop_map[item["question"].strip()] = nh

    out = {}
    for model in ["llada","dream"]:
        try: data = load_judge("musique", model)
        except FileNotFoundError: print(f"  {model}: MISSING"); continue

        assigned = 0
        records = []
        for item in data:
            nh = hop_map.get(item["question"].strip())
            if nh: assigned += 1
            records.append((nh, item["methods"]))

        print(f"\n── {model.upper()} (assigned {assigned}/{len(data)}) ──")
        dist = defaultdict(int)
        for nh,_ in records:
            if nh: dist[nh] += 1
        print(f"  Hop distribution: {dict(sorted(dist.items()))}")

        methods_show = ["baseline","aram","pool","iaram","ispread","idnmr"]
        by_hop = defaultdict(lambda: defaultdict(list))
        for nh, m in records:
            if nh is None: continue
            for method in methods_show:
                if method not in m: continue
                by_hop[nh][method].append((
                    1 if m[method].get("judge")=="correct" else 0,
                    m[method].get("extracted_f1") or 0
                ))

        out[model] = {}
        print(f"\n  {'Hop':<5} {'Method':<12} {'N':>5} {'Judge%':>8} {'ExtF1':>8}")
        for nh in sorted(dist.keys()):
            out[model][str(nh)] = {}
            for method in methods_show:
                vals = by_hop[nh][method]
                if not vals: continue
                j = sum(v[0] for v in vals)/len(vals)*100
                f = sum(v[1] for v in vals)/len(vals)
                print(f"  {nh:<5} {method:<12} {len(vals):>5} {j:>7.1f}% {f:>8.4f}")
                out[model][str(nh)][method] = {"n":len(vals),"judge_pct":round(j,2),"ext_f1":round(f,4)}

    json.dump(out, open(f"{OUTPUT_DIR}/per_hop_analysis.json","w"), indent=2)
    print(f"\n  → {OUTPUT_DIR}/per_hop_analysis.json")
    return out

# ── Task 3: Answer length distributions ──────────────────────────────────────

def task_answer_lengths():
    print("\n" + "="*70)
    print("TASK 3: ANSWER LENGTH DISTRIBUTIONS")
    print("="*70)
    out = {}
    for model in ["llada","dream"]:
        out[model] = {}
        print(f"\n── {model.upper()} ──")
        for dataset in DATASETS:
            try: data = load_judge(dataset, model)
            except FileNotFoundError: continue
            out[model][dataset] = {}
            print(f"\n  {dataset}")
            print(f"  {'Method':<14} {'Mean':>7} {'Med':>5} {'P90':>5} {'>10w%':>7} {'N':>6}")
            for method in METHODS_9:
                lengths = [len((m[method].get("answer") or "").split())
                           for item in data
                           if (m := item["methods"]) and method in m and m[method].get("answer")]
                if not lengths: continue
                arr = np.array(lengths)
                print(f"  {method:<14} {arr.mean():>7.1f} {int(np.median(arr)):>5} "
                      f"{int(np.percentile(arr,90)):>5} {(arr>10).mean()*100:>6.1f}% {len(arr):>6}")
                out[model][dataset][method] = {
                    "n": len(arr), "mean": round(float(arr.mean()),2),
                    "median": int(np.median(arr)), "p90": int(np.percentile(arr,90)),
                    "pct_over_10w": round(float((arr>10).mean()*100),1)
                }
    json.dump(out, open(f"{OUTPUT_DIR}/answer_length_distributions.json","w"), indent=2)
    print(f"\n  → {OUTPUT_DIR}/answer_length_distributions.json")
    return out

# ── Task 4: Error categorization ─────────────────────────────────────────────

def task_error_categorization():
    print("\n" + "="*70)
    print("TASK 4: ERROR CATEGORIZATION")
    print("="*70)
    pairs = [("pool","aram"), ("idnmr","iaram")]
    out = {}
    for model in ["llada","dream"]:
        out[model] = {}
        print(f"\n── {model.upper()} ──")
        for dataset in DATASETS:
            try: data = load_judge(dataset, model)
            except FileNotFoundError: continue
            out[model][dataset] = {}
            for ma, mb in pairs:
                cats = defaultdict(int)
                n = 0
                for item in data:
                    m = item["methods"]
                    if ma not in m or mb not in m: continue
                    n += 1
                    ja = m[ma].get("judge")=="correct"
                    jb = m[mb].get("judge")=="correct"
                    ca = m[ma].get("extracted_contain") or 0
                    cb = m[mb].get("extracted_contain") or 0
                    if   ja and jb:   cats["both_correct"] += 1
                    elif ja and not jb: cats["a_only"] += 1
                    elif not ja and jb: cats["b_only"] += 1
                    else:
                        cats["both_wrong"] += 1
                        if ca > 0:   cats["bw_a_retrieved_gold"] += 1
                        elif cb > 0: cats["bw_b_retrieved_gold"] += 1
                        else:        cats["bw_neither"] += 1
                if not n: continue
                pct = {k: round(v/n*100,1) for k,v in cats.items()}
                print(f"\n  {dataset} | {ma} vs {mb} (N={n})")
                print(f"    Both correct:         {cats['both_correct']:>5}  ({pct['both_correct']}%)")
                print(f"    {ma} only correct:   {cats['a_only']:>5}  ({pct['a_only']}%)")
                print(f"    {mb} only correct:   {cats['b_only']:>5}  ({pct['b_only']}%)")
                print(f"    Both wrong:           {cats['both_wrong']:>5}  ({pct['both_wrong']}%)")
                print(f"      └ {ma} retrieved gold but wrong: {cats['bw_a_retrieved_gold']:>4} ({pct['bw_a_retrieved_gold']}%)")
                print(f"      └ {mb} retrieved gold but wrong: {cats['bw_b_retrieved_gold']:>4} ({pct['bw_b_retrieved_gold']}%)")
                print(f"      └ Neither retrieved gold:        {cats['bw_neither']:>4} ({pct['bw_neither']}%)")
                out[model][dataset][f"{ma}_vs_{mb}"] = {"n":n,"counts":dict(cats),"pcts":pct}
    json.dump(out, open(f"{OUTPUT_DIR}/error_categorization.json","w"), indent=2)
    print(f"\n  → {OUTPUT_DIR}/error_categorization.json")
    return out

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sig  = task_significance()
    hop  = task_per_hop()
    lens = task_answer_lengths()
    err  = task_error_categorization()
    print("\n\n✓ All 4 CPU analyses complete.")
