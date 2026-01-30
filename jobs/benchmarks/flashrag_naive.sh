#!/bin/bash
#SBATCH --job-name=flashrag_naive
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/flashrag_naive_%j.log
#SBATCH --error=jobs/logs/flashrag_naive_%j.err

# FlashRAG Naive Generation baseline across datasets

echo "=========================================="
echo "FlashRAG Naive Generation Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.1.1

ENV_PATH="/projects/prjs1800/conda_envs/multi_agentic_rag"
DATA_ROOT="/projects/prjs1800/datasets"
RESULT_ROOT="/projects/prjs1800/results/flashrag"

if [ -z "$DATASET" ]; then
  echo "ERROR: DATASET environment variable not set. Choose one of: 2wikimultihopqa, hotpotqa, natural_questions, triviaqa"
  exit 1
fi

source activate "$ENV_PATH"

mkdir -p "$RESULT_ROOT"

cd "$HOME/msc-thesis"

export DATASET_NAME="$DATASET"
export DATASET_PATH="$DATA_ROOT/$DATASET"
export FLASHRAG_SAVE_DIR="$RESULT_ROOT/${DATASET}_naive"
mkdir -p "$FLASHRAG_SAVE_DIR"

echo "Dataset: $DATASET_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Results: $FLASHRAG_SAVE_DIR"

python -m flashrag.runner.baseline \
  --method naive \
  --dataset "$DATASET_NAME" \
  --data_dir "$DATASET_PATH" \
  --save_dir "$FLASHRAG_SAVE_DIR" \
  --split dev \
  --gpu "0"

echo "=========================================="
echo "Completed FlashRAG Naive baseline"
echo "End time: $(date)"
echo "=========================================="
