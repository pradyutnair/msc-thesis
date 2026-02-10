#!/bin/bash
#SBATCH --job-name=d6_bootstrap_day6
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day6/bootstrap_day6_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 6 CPU Job 3: Bootstrap CIs for Day 6 results + significance tests
# Submit AFTER GPU Job 1 completes

echo ==========================================
echo "Day 6: Bootstrap CIs + Significance Tests"
echo Job ID: $SLURM_JOB_ID
echo Node: $SLURM_NODELIST
echo "Start time: $(date)"
echo ==========================================

module purge
module load 2025
module load Anaconda3/2025.06-1

source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate

export PYTHONPATH=/projects/prjs1800/external/FlashRAG:$PYTHONPATH
export HF_HOME=/projects/prjs1800/.cache/huggingface

cd /projects/prjs1800/msc-thesis

mkdir -p /projects/prjs1800/analysis/day6

python -u scripts/day6/bootstrap_analysis.py \
    --phase day6 \
    --output_dir /projects/prjs1800/analysis/day6 \
    --n_bootstrap 1000

echo ==========================================
echo "Bootstrap Day 6 + significance tests complete"
echo "End time: $(date)"
echo ==========================================
