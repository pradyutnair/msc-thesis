#!/bin/bash
#SBATCH --job-name=flashrag_standard
#SBATCH --partition=gpu_a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=jobs/output/flashrag_standard.log

# FlashRAG Standard RAG baseline across datasets

echo "=========================================="
echo "FlashRAG Standard RAG Baseline"
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

if [ -z "$DATASET" ]; then
  echo "ERROR: DATASET environment variable not set. Choose one of: 2wikimultihopqa, hotpotqa, natural_questions, triviaqa"
  exit 1
fi

source activate "$ENV_PATH"

mkdir -p "$RESULT_ROOT"

export DATASET_NAME="$DATASET"

echo "Dataset: $DATASET_NAME"
echo "Data root: $DATA_ROOT"
echo "Results: $RESULT_ROOT"

# FlashRAG uses examples/methods/run_exp.py with my_config.yaml
# Note: "naive" in FlashRAG = standard sequential RAG (retrieve + generate)
# Use "zero-shot" for generation without retrieval
cd /projects/prjs1800/FlashRAG/examples/methods

python run_exp.py \
  --method_name naive \
  --dataset_name "$DATASET_NAME" \
  --split dev \
  --gpu_id "0"

echo "=========================================="
echo "Completed FlashRAG Standard RAG baseline"
echo "End time: $(date)"
echo "=========================================="
