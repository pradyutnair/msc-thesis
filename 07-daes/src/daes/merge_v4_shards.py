"""Merge v4 shard JSON results into a single summary table."""
import json, sys, os

shard_dir = "/projects/prjs1800/msc-thesis/07-daes/results/v4"

# Find shard JSON files
shards = {}
for f in sorted(os.listdir(shard_dir)):
    if f.endswith(".json") and "shard" in f:
        data = json.load(open(os.path.join(shard_dir, f)))
        for method, metrics in data.get("summary", {}).items():
            shards[method] = metrics
        print(f"Loaded {f}: methods={list(data.get(summary, {}).keys())}")

if not shards:
    print("No shard JSON files found yet. Waiting...")
    sys.exit(0)

# Print merged table
print(f"\n{Method:<16s} {F1:>6s} {EM:>6s} {Contain:>8s}")
print("-" * 38)
for method in ["baseline", "pool", "ipool", "ispread", "iaram", "v4_trajectory"]:
    if method in shards:
        s = shards[method]
        print(f"{method:<16s} {s[f1]:>6.3f} {s[em]:>6.3f} {s[contain]:>8.3f}")
    else:
        print(f"{method:<16s}   (not available)")
