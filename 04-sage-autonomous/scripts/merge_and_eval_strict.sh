#!/bin/bash
set -euo pipefail

cd /projects/prjs1800/msc-thesis/04-sage-autonomous
source /projects/prjs1800/venvs/arag-venv/bin/activate
export PYTHONNOUSERSITE=1

for DATASET in hotpotqa 2wikimultihop musique; do
    MERGED="results/auto_1k_strict/${DATASET}/predictions.jsonl"
    echo "Merging $DATASET"

    : > "$MERGED"
    for i in 0 1 2 3 4; do
        SHARD="results/auto_1k_strict/${DATASET}/shard_${i}/predictions.jsonl"
        if [[ -f "$SHARD" ]]; then
            cat "$SHARD" >> "$MERGED"
            echo "  shard $i: $(wc -l < "$SHARD") predictions"
        else
            echo "  shard $i: MISSING"
        fi
    done

    echo "  total: $(wc -l < "$MERGED") predictions"
done

PYTHONPATH=src python -u scripts/eval_offline.py \
    results/auto_1k_strict/hotpotqa/predictions.jsonl \
    results/auto_1k_strict/2wikimultihop/predictions.jsonl \
    results/auto_1k_strict/musique/predictions.jsonl
