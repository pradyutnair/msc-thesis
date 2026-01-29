#!/bin/bash
#SBATCH --job-name=naive_hotpot
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/naive_hotpot_%j.log
#SBATCH --error=jobs/logs/naive_hotpot_%j.err

# Naive Vector Search Baseline on HotpotQA
# Simple cosine similarity retrieval without multi-hop reasoning

echo "=========================================="
echo "Naive Vector Search Baseline - HotpotQA"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.1.1

# Activate conda environment
source activate /projects/prjs1800/conda_envs/multi_agentic_rag

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Navigate to project directory
cd $HOME/msc-thesis

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Dataset: HotpotQA"
echo "Model: sentence-transformers/all-MiniLM-L6-v2"
echo "Retrieval: Cosine similarity (FAISS)"
echo "Top-k: 10"
echo "=========================================="

# Run baseline
echo "Running baseline evaluation..."
python -m baselines.naive_vector_search.run_baseline \
    --config baselines/naive_vector_search/configs/hotpotqa.yaml

echo "=========================================="
echo "Baseline evaluation complete!"
echo "End time: $(date)"
echo "=========================================="
