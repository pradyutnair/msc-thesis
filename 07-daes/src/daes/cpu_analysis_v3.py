#!/usr/bin/env python3
"""
CPU Analysis v3 — Additional analyses for paper:
  A) IRCoT (Qwen3-8B) significance test vs Dream DNMR Pool
  B) Metric correlation analysis (Judge% vs ExtF1 vs Contain)
  C) Budget ablation significance (baseline_10 vs DNMR Pool, N=340)
  D) Pareto efficiency table (queries, passages, judge% per method)
"""

import json
import os
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path("/projects/prjs1800/msc-thesis/07-daes")
RESULTS = BASE / "results"
OUT_DIR = RESULTS / "cpu_analysis"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ["musique", "hotpotqa", "2wikimultihopqa"]


# ──────────────────────────────────────────────────────────────────────────────
# Utility: paired bootstrap (vectorized)
# ──────────────────────────────────────────────────────────────────────────────

def paired_bootstrap_np(a, b, n=10_000, seed=42):
    """Return (observed_delta, p_value). a, b are 1-D arrays of per-q scores."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    diff = a - b
    obs = diff.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    boot = diff[idx].mean(axis=1)
    p = np.mean(boot >= 2 * obs) if obs > 0 else np.mean(boot <= 2 * obs)
    return float(obs), float(p)


# ──────────────────────────────────────────────────────────────────────────────
# A) IRCoT (Qwen3-8B) significance test vs Dream DNMR Pool
# ──────────────────────────────────────────────────────────────────────────────

def analysis_a_ircot_significance():
    print("\n=== A) IRCoT (Qwen3-8B) vs Dream DNMR Pool significance ===")
    results = {}

    for dataset in DATASETS:
        # Load qwen3 shards (ar_mqr = IRCoT)
        shards = []
        for s in range(5):
            p = RESULTS / "ar_mqr" / f"qwen3_{dataset}_ar_mqr_1k_s{s}.json"
            if not p.exists():
                print(f"  MISSING: {p}")
                continue
            shards.append(json.load(open(p)))

        if not shards:
            print(f"  No shards for {dataset}, skipping")
            continue

        # Aggregate per-question F1 from ar_mqr field
        ircot_per_q = {}
        for shard in shards:
            for r in shard["results"]:
                qid = r["id"]
                ircot_per_q[qid] = r["ar_mqr"]["f1"]

        print(f"  {dataset}: {len(ircot_per_q)} IRCoT questions")

        # Load Dream judge for Pool F1
        judge_file = RESULTS / "llm_judge" / f"llm_judge_dream_{dataset}.json"
        if not judge_file.exists():
            print(f"  MISSING judge: {judge_file}")
            continue
        judge_data = json.load(open(judge_file))
        pool_per_q = {}
        for entry in judge_data:
            qid = entry["id"]
            if "pool" in entry["methods"]:
                pool_per_q[qid] = entry["methods"]["pool"]["extracted_f1"]

        # Align on common question IDs
        common = sorted(set(ircot_per_q) & set(pool_per_q))
        print(f"  {dataset}: {len(common)} common questions")

        if len(common) < 10:
            print(f"  Too few common questions, skipping bootstrap")
            results[dataset] = {"error": "insufficient overlap"}
            continue

        pool_scores = [pool_per_q[q] for q in common]
        ircot_scores = [ircot_per_q[q] for q in common]

        mean_pool = float(np.mean(pool_scores))
        mean_ircot = float(np.mean(ircot_scores))

        # Bootstrap: pool vs ircot
        delta, p_val = paired_bootstrap_np(pool_scores, ircot_scores)

        results[dataset] = {
            "n": len(common),
            "mean_dnmr_pool_f1": round(mean_pool, 4),
            "mean_ircot_qwen3_f1": round(mean_ircot, 4),
            "delta_pool_minus_ircot": round(delta, 4),
            "p_value": round(p_val, 4),
            "significant_p05": p_val < 0.05,
        }
        print(f"  {dataset}: Pool={mean_pool:.4f} IRCoT={mean_ircot:.4f} "
              f"delta={delta:+.4f} p={p_val:.4f}")

    out_path = OUT_DIR / "ircot_significance.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# B) Metric correlation analysis
# ──────────────────────────────────────────────────────────────────────────────

def analysis_b_metric_correlation():
    print("\n=== B) Metric correlation analysis ===")
    all_results = {}

    for model_tag, model_label in [("dream", "Dream"), ("llada", "LLaDA")]:
        all_results[model_label] = {}
        for dataset in DATASETS:
            if model_tag == "dream":
                judge_file = RESULTS / "llm_judge" / f"llm_judge_dream_{dataset}.json"
            else:
                judge_file = RESULTS / "llm_judge" / f"llm_judge_{dataset}.json"

            if not judge_file.exists():
                print(f"  MISSING: {judge_file}")
                continue

            judge_data = json.load(open(judge_file))

            # Collect per-question vectors across all methods
            # Metrics: judge (0/1), extracted_f1, extracted_contain
            # We compare these three metric types across all (question, method) pairs
            judge_vals = []
            ext_f1_vals = []
            contain_vals = []

            methods_in_data = set()
            for entry in judge_data:
                for method, mdata in entry["methods"].items():
                    methods_in_data.add(method)
                    judge_binary = 1.0 if mdata.get("judge") == "correct" else 0.0
                    ext_f1 = float(mdata.get("extracted_f1", 0.0))
                    contain = float(mdata.get("extracted_contain", 0.0))
                    judge_vals.append(judge_binary)
                    ext_f1_vals.append(ext_f1)
                    contain_vals.append(contain)

            n = len(judge_vals)
            print(f"  {model_label}/{dataset}: {n} (question,method) pairs, methods={sorted(methods_in_data)}")

            j = np.array(judge_vals)
            f = np.array(ext_f1_vals)
            c = np.array(contain_vals)

            # Pearson correlations
            r_jf, p_jf = stats.pearsonr(j, f)
            r_jc, p_jc = stats.pearsonr(j, c)
            r_fc, p_fc = stats.pearsonr(f, c)

            # Spearman correlations
            rho_jf, sp_jf = stats.spearmanr(j, f)
            rho_jc, sp_jc = stats.spearmanr(j, c)
            rho_fc, sp_fc = stats.spearmanr(f, c)

            # Aggregate means per method for table
            method_stats = {}
            for entry in judge_data:
                for method, mdata in entry["methods"].items():
                    if method not in method_stats:
                        method_stats[method] = {"judge": [], "ext_f1": [], "contain": []}
                    method_stats[method]["judge"].append(
                        1.0 if mdata.get("judge") == "correct" else 0.0)
                    method_stats[method]["ext_f1"].append(float(mdata.get("extracted_f1", 0.0)))
                    method_stats[method]["contain"].append(float(mdata.get("extracted_contain", 0.0)))

            method_means = {}
            for method, vals in method_stats.items():
                method_means[method] = {
                    "judge_pct": round(np.mean(vals["judge"]) * 100, 1),
                    "ext_f1": round(np.mean(vals["ext_f1"]), 4),
                    "contain_pct": round(np.mean(vals["contain"]) * 100, 1),
                }

            all_results[model_label][dataset] = {
                "n_pairs": n,
                "pearson": {
                    "judge_vs_ext_f1": {"r": round(r_jf, 4), "p": round(p_jf, 6)},
                    "judge_vs_contain": {"r": round(r_jc, 4), "p": round(p_jc, 6)},
                    "ext_f1_vs_contain": {"r": round(r_fc, 4), "p": round(p_fc, 6)},
                },
                "spearman": {
                    "judge_vs_ext_f1": {"rho": round(rho_jf, 4), "p": round(sp_jf, 6)},
                    "judge_vs_contain": {"rho": round(rho_jc, 4), "p": round(sp_jc, 6)},
                    "ext_f1_vs_contain": {"rho": round(rho_fc, 4), "p": round(sp_fc, 6)},
                },
                "method_means": method_means,
            }

            print(f"    Pearson judge~F1={r_jf:.3f}, judge~contain={r_jc:.3f}, F1~contain={r_fc:.3f}")
            print(f"    Spearman rho: judge~F1={rho_jf:.3f}, judge~contain={rho_jc:.3f}, F1~contain={rho_fc:.3f}")

    out_path = OUT_DIR / "metric_correlation.json"
    json.dump(all_results, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")
    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# C) Budget ablation significance (baseline_10 vs DNMR Pool, N=340)
# ──────────────────────────────────────────────────────────────────────────────

def analysis_c_budget_significance():
    print("\n=== C) Budget ablation significance (N=340) ===")

    ablation_file = RESULTS / "ablation_budget_dream_musique_1k.json"
    if not ablation_file.exists():
        print(f"  MISSING: {ablation_file}")
        return {}

    data = json.load(open(ablation_file))
    records = data["results"]
    print(f"  N={len(records)} records")

    b10_f1 = [r["baseline_10"]["f1"] for r in records]
    pool_f1 = [r["dnmr_pool"]["f1"] for r in records]
    b10_em = [r["baseline_10"]["em"] for r in records]
    pool_em = [r["dnmr_pool"]["em"] for r in records]
    b10_contain = [r["baseline_10"]["contain"] for r in records]
    pool_contain = [r["dnmr_pool"]["contain"] for r in records]

    # Paired bootstrap on each metric
    delta_f1, p_f1 = paired_bootstrap_np(pool_f1, b10_f1)
    delta_em, p_em = paired_bootstrap_np(pool_em, b10_em)
    delta_contain, p_contain = paired_bootstrap_np(pool_contain, b10_contain)

    results = {
        "n": len(records),
        "note": "Dream MuSiQue, N=340 (budget ablation subset)",
        "baseline_10": {
            "mean_f1": round(np.mean(b10_f1), 4),
            "mean_em": round(np.mean(b10_em), 4),
            "mean_contain": round(np.mean(b10_contain) * 100, 1),
        },
        "dnmr_pool": {
            "mean_f1": round(np.mean(pool_f1), 4),
            "mean_em": round(np.mean(pool_em), 4),
            "mean_contain": round(np.mean(pool_contain) * 100, 1),
        },
        "paired_bootstrap": {
            "f1": {"delta": round(delta_f1, 4), "p": round(p_f1, 4), "sig_p05": p_f1 < 0.05},
            "em": {"delta": round(delta_em, 4), "p": round(p_em, 4), "sig_p05": p_em < 0.05},
            "contain": {"delta": round(delta_contain, 4), "p": round(p_contain, 4), "sig_p05": p_contain < 0.05},
        },
    }

    print(f"  baseline_10: F1={results['baseline_10']['mean_f1']:.4f} "
          f"Contain={results['baseline_10']['mean_contain']:.1f}%")
    print(f"  dnmr_pool:   F1={results['dnmr_pool']['mean_f1']:.4f} "
          f"Contain={results['dnmr_pool']['mean_contain']:.1f}%")
    print(f"  Bootstrap F1: delta={delta_f1:+.4f} p={p_f1:.4f} sig={p_f1<0.05}")
    print(f"  Bootstrap Contain: delta={delta_contain:+.4f} p={p_contain:.4f} sig={p_contain<0.05}")

    out_path = OUT_DIR / "budget_ablation_significance.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# D) Pareto efficiency table
# ──────────────────────────────────────────────────────────────────────────────

def analysis_d_pareto():
    print("\n=== D) Pareto efficiency table ===")

    # Load the idnmr_pilot results for passage/query counts
    # These are stored in the llm_judge files which have all methods
    # We also need to check if there's a separate timing/passage-count file

    # Strategy: use judge files for performance, and check if we have
    # passage count info in raw result files
    pareto = {}

    for model_tag, model_label in [("dream", "Dream"), ("llada", "LLaDA")]:
        pareto[model_label] = {}
        for dataset in DATASETS:
            if model_tag == "dream":
                judge_file = RESULTS / "llm_judge" / f"llm_judge_dream_{dataset}.json"
            else:
                judge_file = RESULTS / "llm_judge" / f"llm_judge_{dataset}.json"

            if not judge_file.exists():
                print(f"  MISSING: {judge_file}")
                continue

            judge_data = json.load(open(judge_file))
            methods = set()
            for entry in judge_data:
                methods.update(entry["methods"].keys())

            method_stats = {}
            for entry in judge_data:
                for method, mdata in entry["methods"].items():
                    if method not in method_stats:
                        method_stats[method] = {"judge": [], "ext_f1": []}
                    method_stats[method]["judge"].append(
                        1.0 if mdata.get("judge") == "correct" else 0.0)
                    method_stats[method]["ext_f1"].append(float(mdata.get("extracted_f1", 0.0)))

            dataset_stats = {}
            for method, vals in sorted(method_stats.items()):
                dataset_stats[method] = {
                    "n": len(vals["judge"]),
                    "judge_pct": round(np.mean(vals["judge"]) * 100, 1),
                    "ext_f1": round(np.mean(vals["ext_f1"]), 4),
                }
            pareto[model_label][dataset] = dataset_stats

    # Now add retrieval budget info from raw result files (if available)
    # Check idnmr_pilot results for passage counts
    passage_info = {}
    for dataset in DATASETS:
        raw_file = RESULTS / f"idnmr_1k_{dataset}.json"
        if not raw_file.exists():
            raw_file = RESULTS / f"dnmr_pool_1k_{dataset}.json"
        if raw_file.exists():
            try:
                d = json.load(open(raw_file))
                records = d.get("results", d) if isinstance(d, dict) else d
                print(f"  {dataset} raw file: {len(records)} records, keys={list(records[0].keys())[:6]}")
            except Exception as e:
                print(f"  Error reading {raw_file}: {e}")

    # Add retrieval budget from DNMR run metadata
    # Expected retrieval rounds per method:
    retrieval_budget = {
        "baseline": {"queries": 1, "passages": 5, "note": "1 query, top-5"},
        "spread": {"queries": "k", "passages": 5, "note": "k queries, top-5 pooled"},
        "aram": {"queries": 2, "passages": "5+5", "note": "2 queries, top-5 each"},
        "pool": {"queries": 1, "passages": "~7-8", "note": "DNMR 1-round, ~7-8 passages"},
        "ipool": {"queries": "1-2", "passages": "~7-10", "note": "iDNMR pool variant"},
        "idnmr": {"queries": "2-3", "passages": "~10-14", "note": "iDNMR 2-3 rounds"},
        "ispread": {"queries": "2k", "passages": "5k", "note": "iterative SPREAD"},
        "iaram": {"queries": 4, "passages": "5+5+5+5", "note": "iterative ARAM 2 rounds"},
    }
    pareto["retrieval_budget"] = retrieval_budget

    out_path = OUT_DIR / "pareto_efficiency.json"
    json.dump(pareto, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")

    # Print summary table
    print("\n  --- Dream Performance Summary ---")
    for dataset in DATASETS:
        print(f"  {dataset}:")
        if "Dream" in pareto and dataset in pareto["Dream"]:
            for method, stats in sorted(pareto["Dream"][dataset].items()):
                print(f"    {method:12s}: judge={stats['judge_pct']:5.1f}%  F1={stats['ext_f1']:.4f}")

    print("\n  --- LLaDA Performance Summary ---")
    for dataset in DATASETS:
        print(f"  {dataset}:")
        if "LLaDA" in pareto and dataset in pareto["LLaDA"]:
            for method, stats in sorted(pareto["LLaDA"][dataset].items()):
                print(f"    {method:12s}: judge={stats['judge_pct']:5.1f}%  F1={stats['ext_f1']:.4f}")

    return pareto


# ──────────────────────────────────────────────────────────────────────────────
# E) Cross-model significance: LLaDA iDNMR vs LLaDA iARAM
# ──────────────────────────────────────────────────────────────────────────────

def analysis_e_llada_idnmr_significance():
    print("\n=== E) LLaDA iDNMR vs iARAM significance ===")
    results = {}

    for dataset in DATASETS:
        judge_file = RESULTS / "llm_judge" / f"llm_judge_{dataset}.json"
        if not judge_file.exists():
            print(f"  MISSING: {judge_file}")
            continue

        judge_data = json.load(open(judge_file))

        idnmr_f1 = []
        iaram_f1 = []
        idnmr_judge = []
        iaram_judge = []

        for entry in judge_data:
            methods = entry["methods"]
            if "idnmr" in methods and "iaram" in methods:
                idnmr_f1.append(float(methods["idnmr"].get("extracted_f1", 0.0)))
                iaram_f1.append(float(methods["iaram"].get("extracted_f1", 0.0)))
                idnmr_judge.append(1.0 if methods["idnmr"].get("judge") == "correct" else 0.0)
                iaram_judge.append(1.0 if methods["iaram"].get("judge") == "correct" else 0.0)

        n = len(idnmr_f1)
        print(f"  {dataset}: N={n}")

        delta_f1, p_f1 = paired_bootstrap_np(idnmr_f1, iaram_f1)
        delta_judge, p_judge = paired_bootstrap_np(idnmr_judge, iaram_judge)

        results[dataset] = {
            "n": n,
            "llada_idnmr_f1": round(np.mean(idnmr_f1), 4),
            "llada_iaram_f1": round(np.mean(iaram_f1), 4),
            "llada_idnmr_judge_pct": round(np.mean(idnmr_judge) * 100, 1),
            "llada_iaram_judge_pct": round(np.mean(iaram_judge) * 100, 1),
            "paired_bootstrap_f1": {
                "delta": round(delta_f1, 4),
                "p": round(p_f1, 4),
                "sig_p05": p_f1 < 0.05,
            },
            "paired_bootstrap_judge": {
                "delta": round(delta_judge, 4),
                "p": round(p_judge, 4),
                "sig_p05": p_judge < 0.05,
            },
        }
        print(f"    iDNMR F1={results[dataset]['llada_idnmr_f1']:.4f} "
              f"iARAM F1={results[dataset]['llada_iaram_f1']:.4f} "
              f"delta={delta_f1:+.4f} p={p_f1:.4f}")
        print(f"    iDNMR Judge={results[dataset]['llada_idnmr_judge_pct']:.1f}% "
              f"iARAM Judge={results[dataset]['llada_iaram_judge_pct']:.1f}% "
              f"delta={delta_judge:+.4f} p={p_judge:.4f}")

    out_path = OUT_DIR / "llada_idnmr_significance.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"  Saved: {out_path}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CPU Analysis v3 — Additional paper-readiness analyses")
    print("=" * 60)

    res_a = analysis_a_ircot_significance()
    res_b = analysis_b_metric_correlation()
    res_c = analysis_c_budget_significance()
    res_d = analysis_d_pareto()
    res_e = analysis_e_llada_idnmr_significance()

    print("\n" + "=" * 60)
    print("All analyses complete. Results in results/cpu_analysis/")
    print("  ircot_significance.json")
    print("  metric_correlation.json")
    print("  budget_ablation_significance.json")
    print("  pareto_efficiency.json")
    print("  llada_idnmr_significance.json")
