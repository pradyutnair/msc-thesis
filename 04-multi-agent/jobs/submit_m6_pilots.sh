#!/bin/bash
set -euo pipefail
cd /projects/prjs1800/msc-thesis/04-multi-agent
for DATASET in hotpotqa 2wikimultihop musique; do
  GEN_JOB=$(sbatch --parsable --export=ALL,DATASET="$DATASET",LIMIT=20,RUN_NAME=m6_pilot20,CONFIG_PATH=configs/m6_litcore.yaml jobs/m6_generate_dataset.job)
  sbatch --dependency=afterok:${GEN_JOB} --export=ALL,PREDICTIONS="results/m6_pilot20/$DATASET/predictions.jsonl",OUTPUT_DIR="results/m6_pilot20/$DATASET" jobs/m6_eval_deepseek.job
  echo "$DATASET generation job: $GEN_JOB"
done
