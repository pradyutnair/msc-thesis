#!/bin/bash
#SBATCH --job-name=d6_error_taxonomy
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day6/error_taxonomy_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 6 CPU Job 4: Error taxonomy + cross-method Venn + per-hop analysis
# Submit AFTER GPU Job 1 completes

echo ==========================================
echo "Day 6: Error Taxonomy + Venn + Per-Hop"
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

python -u scripts/day6/error_taxonomy.py \
    --output_dir /projects/prjs1800/analysis/day6

echo ==========================================
echo "Error taxonomy complete"
echo "End time: $(date)"
echo ==========================================
