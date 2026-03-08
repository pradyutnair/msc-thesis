#!/bin/bash
# Merge shard predictions and run evaluation
set -euo pipefail
cd /projects/prjs1800/msc-thesis/04-sage-autonomous
source /projects/prjs1800/venvs/arag-venv/bin/activate

for DATASET in hotpotqa 2wikimultihop musique; do
    MERGED="results/auto_1k_final/${DATASET}/predictions.jsonl"
    echo "=== Merging $DATASET ==="

    # Concatenate all shard predictions
    > "$MERGED"  # truncate
    for i in 0 1 2 3 4; do
        SHARD="results/auto_1k_final/${DATASET}/shard_${i}/predictions.jsonl"
        if [[ -f "$SHARD" ]]; then
            cat "$SHARD" >> "$MERGED"
            echo "  shard $i: $(wc -l < "$SHARD") predictions"
        else
            echo "  shard $i: MISSING"
        fi
    done
    echo "  Total: $(wc -l < "$MERGED") predictions"
    echo
done

echo "=== Running evaluation ==="
PYTHONPATH=src python -u scripts/eval_offline.py \
    results/auto_1k_final/hotpotqa/predictions.jsonl \
    results/auto_1k_final/2wikimultihop/predictions.jsonl \
    results/auto_1k_final/musique/predictions.jsonl

echo
echo "=== Comparison with v3r2 baseline ==="
PYTHONPATH=src python -u scripts/eval_offline.py \
    results/auto_1k_final/hotpotqa/predictions.jsonl \
    results/auto_1k_final/2wikimultihop/predictions.jsonl \
    results/auto_1k_final/musique/predictions.jsonl \
    results/sage_v3r2_1000/hotpotqa/predictions.jsonl \
    results/sage_v3r2_1000/2wikimultihop/predictions.jsonl \
    results/sage_v3r2_1000/musique/predictions.jsonl
