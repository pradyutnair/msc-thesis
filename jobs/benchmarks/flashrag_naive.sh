#!/bin/bash
#SBATCH --job-name=flashrag_naive
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=jobs/output/flashrag_naive.log

# FlashRAG Naive Generation baseline across datasets

echo "=========================================="
echo "FlashRAG Naive Generation Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

ENV_PATH="/projects/prjs1800/conda_envs/multi_agentic_rag"
DATA_ROOT="/projects/prjs1800/datasets/flashrag"
RESULT_ROOT="/projects/prjs1800/results/flashrag"

# DATASET from env, first argument, or default
if [ -z "$DATASET" ] && [ -n "$1" ]; then
  export DATASET="$1"
fi
if [ -z "$DATASET" ]; then
  export DATASET="2wikimultihopqa"
fi

source activate "$ENV_PATH"

# Hugging Face cache (unified under .cache/huggingface)
export HF_HOME="/projects/prjs1800/.cache/huggingface"
export HF_HUB_CACHE="/projects/prjs1800/.cache/huggingface"
export TRANSFORMERS_CACHE="/projects/prjs1800/.cache/huggingface"
export HF_DATASETS_CACHE="/projects/prjs1800/.cache/huggingface"
# Avoid MKL vs libgomp conflict in vLLM subprocess (model inspection)
export MKL_THREADING_LAYER=GNU

mkdir -p "$RESULT_ROOT"

cd /projects/prjs1800/FlashRAG

export DATASET_NAME="$DATASET"
export DATASET_PATH="$DATA_ROOT/$DATASET"
export FLASHRAG_SAVE_DIR="$RESULT_ROOT/${DATASET}_naive"
mkdir -p "$FLASHRAG_SAVE_DIR"

echo "Dataset: $DATASET_NAME"
echo "Dataset path: $DATA_ROOT"
echo "Results: $FLASHRAG_SAVE_DIR"

# FlashRAG uses examples/methods/run_exp.py with my_config.yaml
cd /projects/prjs1800/FlashRAG/examples/methods

# Run the naive baseline using run_exp.py
python run_exp.py \
  --method_name naive \
  --dataset_name "$DATASET_NAME" \
  --split dev \
  --gpu_id "0"

echo "=========================================="
echo "Completed FlashRAG Naive baseline"
echo "End time: $(date)"
echo "=========================================="
