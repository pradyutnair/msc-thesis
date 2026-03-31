"""Retrieval-side metrics for DNMR paper.

Computes:
- Bridge hit rate: % of extracted bridges that retrieve NEW useful passages
- New passage yield per method
- Total passages per method
- Support recall (if gold supporting facts available)

Run: python src/daes/retrieval_metrics.py
"""
import json, glob, os
import numpy as np
from collections import defaultdict


def compute_retrieval_metrics(results_dir, methods, datasets):
    """Compute retrieval-side metrics from saved JSON results."""
    
    for dataset in datasets:
        print(f"\n=== {dataset} ===")
        
        # Load iDNMR results (has passage counts and round stats)
        pattern = f"{results_dir}/idnmr/dream_{dataset}_idnmr_1k_s*.json"
        files = sorted(glob.glob(pattern))
        
        all_results = []
        for f in files:
            d = json.load(open(f))
            all_results.extend(d.get("results", []))
        
        if not all_results:
            print("  No results found")
            continue
        
        print(f"  Questions: {len(all_results)}")
        
        # Compute per-method metrics
        for method in ["baseline", "pool", "ipool", "idnmr", "idnmr_2round"]:
            f1s = []
            for r in all_results:
                if method in r:
                    f1s.append(r[method]["f1"])
            
            if f1s:
                mean_f1 = np.mean(f1s)
                ci_lo = np.percentile(np.random.choice(f1s, (10000, len(f1s)), replace=True).mean(axis=1), 2.5)
                ci_hi = np.percentile(np.random.choice(f1s, (10000, len(f1s)), replace=True).mean(axis=1), 97.5)
                print(f"  {method:16s} F1={mean_f1:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
        
        # iDNMR round stats
        rounds_stats = []
        for r in all_results:
            stats = r.get("idnmr_stats", [])
            if stats:
                total_new = sum(s.get("new_passages", 0) for s in stats)
                n_rounds = len(stats)
                rounds_stats.append({"total_new": total_new, "n_rounds": n_rounds})
        
        if rounds_stats:
            mean_new = np.mean([s["total_new"] for s in rounds_stats])
            mean_rounds = np.mean([s["n_rounds"] for s in rounds_stats])
            print(f"\n  iDNMR retrieval stats:")
            print(f"    Mean new passages: {mean_new:.1f}")
            print(f"    Mean rounds executed: {mean_rounds:.1f}")


def compute_bootstrap_cis(results_dir, datasets):
    """Compute bootstrap confidence intervals for all methods."""
    
    print("\n" + "="*80)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("="*80)
    
    np.random.seed(42)
    
    for source, pattern, methods in [
        ("iDNMR", "{dir}/idnmr/dream_{ds}_idnmr_1k_s*.json", ["baseline","pool","ipool","idnmr","idnmr_2round"]),
        ("baselines", "{dir}/baselines/dream_{ds}_baselines_s*.json", ["spread","aram","ispread","iaram"]),
    ]:
        for dataset in datasets:
            files = sorted(glob.glob(pattern.format(dir=results_dir, ds=dataset)))
            per_q = {}
            for f in files:
                d = json.load(open(f))
                for r in d.get("results", []):
                    qid = r["id"]
                    for m in methods:
                        if m in r:
                            if m not in per_q:
                                per_q[m] = {}
                            per_q[m][qid] = r[m]["f1"]
            
            print(f"\n{dataset} ({source}):")
            print(f"  {'Method':<16s} {'F1':>8s} {'95% CI':>20s}")
            print(f"  {'-'*46}")
            for m in methods:
                if m in per_q:
                    scores = list(per_q[m].values())
                    n = len(scores)
                    arr = np.array(scores)
                    boot = np.random.choice(arr, (10000, n), replace=True).mean(axis=1)
                    ci_lo = np.percentile(boot, 2.5)
                    ci_hi = np.percentile(boot, 97.5)
                    print(f"  {m:<16s} {arr.mean():>8.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")


def compute_efficiency_summary(results_dir, datasets):
    """Compute efficiency metrics from baselines results (which log timing)."""
    
    print("\n" + "="*80)
    print("EFFICIENCY METRICS")
    print("="*80)
    
    for dataset in datasets:
        pattern = f"{results_dir}/baselines/dream_{dataset}_baselines_s*.json"
        files = sorted(glob.glob(pattern))
        
        method_walls = defaultdict(list)
        method_fps = defaultdict(list)
        method_rqs = defaultdict(list)
        
        for f in files:
            d = json.load(open(f))
            for r in d.get("results", []):
                stats = r.get("method_stats", {})
                for m, s in stats.items():
                    method_walls[m].append(s.get("wall_sec", 0))
                    method_fps[m].append(s.get("forward_passes", 0))
                    method_rqs[m].append(s.get("retrieval_queries", 0))
        
        if method_walls:
            print(f"\n{dataset}:")
            print(f"  {'Method':<16s} {'Wall(s)':>8s} {'FwdPass':>8s} {'RetQ':>6s}")
            print(f"  {'-'*40}")
            for m in ["spread", "aram", "ispread", "iaram"]:
                if m in method_walls:
                    print(f"  {m:<16s} {np.mean(method_walls[m]):>8.2f} {np.mean(method_fps[m]):>8.1f} {np.mean(method_rqs[m]):>6.1f}")


if __name__ == "__main__":
    results_dir = "/projects/prjs1800/msc-thesis/07-daes/results"
    datasets = ["musique", "hotpotqa", "2wikimultihopqa"]
    
    compute_retrieval_metrics(results_dir, None, datasets)
    compute_bootstrap_cis(results_dir, datasets)
    compute_efficiency_summary(results_dir, datasets)
