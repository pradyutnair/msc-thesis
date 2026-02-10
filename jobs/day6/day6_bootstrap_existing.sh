#!/bin/bash
#SBATCH --job-name=d6_bootstrap_existing
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day6/bootstrap_existing_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

echo ==========================================
echo "Day 6: Bootstrap CIs for Days 1-5"
echo Job ID: $SLURM_JOB_ID
echo Node: $SLURM_NODELIST
echo "Start time: $(date)"
echo ==========================================

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate

export PYTHONPATH=/projects/prjs1800/external/FlashRAG:$PYTHONPATH
export HF_HOME=/projects/prjs1800/.cache/huggingface

cd /projects/prjs1800/msc-thesis

mkdir -p /projects/prjs1800/analysis/day6

python -u scripts/day6/bootstrap_analysis.py \
    --phase existing \
    --output_dir /projects/prjs1800/analysis/day6 \
    --n_bootstrap 1000

echo ==========================================
echo "Bootstrap Days 1-5 complete"
echo "End time: $(date)"
echo ==========================================
