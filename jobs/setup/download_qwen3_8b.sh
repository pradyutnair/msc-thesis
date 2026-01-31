#!/bin/bash
#SBATCH --job-name=dl_qwen3
#SBATCH --partition=cbuild
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/download_qwen3_8b.log

# One-time download of Qwen/Qwen3-8B to project cache. Run before flashrag_naive/standard with Qwen config.

echo "=========================================="
echo "Download Qwen/Qwen3-8B"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

module purge
module load 2025
module load Anaconda3/2025.06-1

cd /projects/prjs1800
source venvs/FlashRAG-venv/bin/activate

export HF_HOME="/projects/prjs1800/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"

rm -rf "$HF_HOME/models--Qwen--Qwen3-8B"
echo "Downloading Qwen/Qwen3-8B via HF CLI..."
huggingface-cli download Qwen/Qwen3-8B || exit 1
echo "=========================================="
echo "Qwen3-8B ready. End: $(date)"
echo "=========================================="
