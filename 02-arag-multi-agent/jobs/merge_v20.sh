#!/bin/bash
set -euo pipefail
cd /projects/prjs1800/msc-thesis/02-arag-multi-agent

for ds in hotpotqa 2wiki musique; do
    echo "=== Merging $ds ==="
    OUT="results/m6v20_1000/$ds"
    MERGED="$OUT/predictions.jsonl"
    > "$MERGED"
    for i in $(seq 0 14); do
        SHARD="$OUT/shard_${i}/predictions.jsonl"
        if [ -f "$SHARD" ]; then
            cat "$SHARD" >> "$MERGED"
            echo "  shard_${i}: $(wc -l < "$SHARD") predictions"
        else
            echo "  WARNING: shard_${i} missing!"
        fi
    done
    TOTAL=$(wc -l < "$MERGED")
    echo "  Total: $TOTAL predictions"

    echo "=== Offline Eval: $ds ==="
    python -u scripts/eval_offline.py "$MERGED"
    echo ""
done
