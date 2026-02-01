#!/bin/bash
#SBATCH --job-name=fr_naive
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/flashrag_naive.log

# FlashRAG Naive Generation (no retrieval) on NQ, TriviaQA, HotpotQA, 2Wiki, PopQA, WebQA

echo "=========================================="
echo "FlashRAG Naive Generation Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

cd /projects/prjs1800
source venvs/FlashRAG-venv/bin/activate

FLASHRAG_ROOT="/projects/prjs1800/external/FlashRAG"
FLASHRAG_METHODS="$FLASHRAG_ROOT/examples/methods"
export PYTHONPATH="$FLASHRAG_ROOT:$PYTHONPATH"
export HF_HOME="/projects/prjs1800/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
# Config file in msc-thesis for version control
FLASHRAG_CONFIG="/projects/prjs1800/msc-thesis/configs/flashrag/naive_generation.yaml"
DATASETS="nq triviaqa hotpotqa 2wikimultihopqa popqa web_questions"

cd "$FLASHRAG_METHODS"

for DATASET in $DATASETS; do
  echo "----------------------------------------"
  echo "Dataset: $DATASET (naive)"
  echo "----------------------------------------"
  python run_exp.py \
    --config "$FLASHRAG_CONFIG" \
    --method_name naive \
    --dataset_name "$DATASET" \
    --split test \
    --gpu_id "0" || exit 1
done

echo "=========================================="
echo "Completed FlashRAG Naive (all datasets)"
echo "End time: $(date)"
echo "=========================================="
