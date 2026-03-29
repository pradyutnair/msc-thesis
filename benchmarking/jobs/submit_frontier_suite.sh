#!/bin/bash
set -euo pipefail

REPO=${REPO:-/projects/prjs1800/msc-thesis-benchmark}
OUTPUT_ROOT=${OUTPUT_ROOT:-/projects/prjs1800/msc-thesis/benchmarking/results/frontier_single}
SHARD_SIZE=${SHARD_SIZE:-100}
ARRAY=${ARRAY:-0-9}

declare -A QUESTIONS=(
  [hotpotqa]="/projects/prjs1800/msc-thesis/01-arag-reproduction/data/questions_wiki18/hotpotqa.json"
  [musique]="/projects/prjs1800/msc-thesis/01-arag-reproduction/data/questions_wiki18/musique.json"
  [2wikimultihopqa]="/projects/prjs1800/msc-thesis/01-arag-reproduction/data/questions_wiki18/2wikimultihopqa.json"
)

declare -A ARAG_CONFIGS=(
  [hotpotqa]="/projects/prjs1800/msc-thesis/01-arag-reproduction/configs/arag_qwen3_vllm_e5_hotpotqa.yaml"
  [musique]="/projects/prjs1800/msc-thesis/01-arag-reproduction/configs/arag_qwen3_vllm_e5_musique.yaml"
  [2wikimultihopqa]="/projects/prjs1800/msc-thesis/01-arag-reproduction/configs/arag_qwen3_vllm_e5_2wikimultihop.yaml"
)

declare -A IRCOT_CONFIGS=(
  [hotpotqa]="${REPO}/00-baseline-preparation/configs/day4/ircot_qwen3_hotpotqa.yaml"
  [musique]="${REPO}/00-baseline-preparation/configs/day4/ircot_qwen3_musique.yaml"
  [2wikimultihopqa]="${REPO}/00-baseline-preparation/configs/day4/ircot_qwen3_2wikimultihopqa.yaml"
)

MERGE_IDS=()

for DATASET in hotpotqa musique 2wikimultihopqa; do
  B0_JOB=$(sbatch --parsable --array="$ARRAY" \
    --export=ALL,REPO="$REPO",DATASET="$DATASET",QUESTIONS="${QUESTIONS[$DATASET]}",CONFIG="${ARAG_CONFIGS[$DATASET]}",OUTPUT_ROOT="$OUTPUT_ROOT",SHARD_SIZE="$SHARD_SIZE" \
    "${REPO}/benchmarking/jobs/frontier_b0_array.job")
  E2_JOB=$(sbatch --parsable --array="$ARRAY" \
    --export=ALL,REPO="$REPO",DATASET="$DATASET",QUESTIONS="${QUESTIONS[$DATASET]}",CONFIG="${ARAG_CONFIGS[$DATASET]}",OUTPUT_ROOT="$OUTPUT_ROOT",SHARD_SIZE="$SHARD_SIZE" \
    "${REPO}/benchmarking/jobs/frontier_e2_array.job")
  IRCOT_JOB=$(sbatch --parsable --array="$ARRAY" \
    --export=ALL,REPO="$REPO",DATASET="$DATASET",QUESTIONS="${QUESTIONS[$DATASET]}",CONFIG="${IRCOT_CONFIGS[$DATASET]}",OUTPUT_ROOT="$OUTPUT_ROOT",SHARD_SIZE="$SHARD_SIZE" \
    "${REPO}/benchmarking/jobs/frontier_ircot_array.job")
  LLaDA_JOB=$(sbatch --parsable --array="$ARRAY" \
    --export=ALL,REPO="$REPO",DATASET="$DATASET",OUTPUT_ROOT="$OUTPUT_ROOT",SHARD_SIZE="$SHARD_SIZE" \
    "${REPO}/benchmarking/jobs/frontier_llada_array.job")

  MERGE_IDS+=("$(sbatch --parsable --dependency=afterok:$B0_JOB --export=ALL,REPO="$REPO",DATASET="$DATASET",METHOD="b0",OUTPUT_ROOT="$OUTPUT_ROOT" \
    "${REPO}/benchmarking/jobs/frontier_merge_dataset.job")")
  MERGE_IDS+=("$(sbatch --parsable --dependency=afterok:$E2_JOB --export=ALL,REPO="$REPO",DATASET="$DATASET",METHOD="e2_react",OUTPUT_ROOT="$OUTPUT_ROOT" \
    "${REPO}/benchmarking/jobs/frontier_merge_dataset.job")")
  MERGE_IDS+=("$(sbatch --parsable --dependency=afterok:$IRCOT_JOB --export=ALL,REPO="$REPO",DATASET="$DATASET",METHOD="ircot",OUTPUT_ROOT="$OUTPUT_ROOT" \
    "${REPO}/benchmarking/jobs/frontier_merge_dataset.job")")
  MERGE_IDS+=("$(sbatch --parsable --dependency=afterok:$LLaDA_JOB --export=ALL,REPO="$REPO",DATASET="$DATASET",METHOD="llada",OUTPUT_ROOT="$OUTPUT_ROOT" \
    "${REPO}/benchmarking/jobs/frontier_merge_dataset.job")")
done

DEP=$(IFS=:; echo "${MERGE_IDS[*]}")
sbatch --dependency=afterok:$DEP --export=ALL,REPO="$REPO",OUTPUT_ROOT="$OUTPUT_ROOT" \
  "${REPO}/benchmarking/jobs/frontier_aggregate_all.job"
