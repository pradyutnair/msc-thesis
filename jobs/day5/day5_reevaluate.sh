#!/bin/bash
#SBATCH --job-name=d5_reeval
#SBATCH --partition=rome
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day5/reevaluate_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 5: Re-evaluate CoT results with fixed regex
# No GPU needed — just reads intermediate_data.json and re-computes metrics

echo ==========================================
echo Day 5: Re-evaluate CoT with fixed regex
echo Job ID: $SLURM_JOB_ID
echo Start time: $(date)
echo ==========================================

module purge
module load 2025
module load Anaconda3/2025.06-1

source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate

cd /projects/prjs1800/msc-thesis

# Re-evaluate HotpotQA Reranker+CoT
echo ==========================================
echo Re-evaluating HotpotQA Reranker+CoT
echo ==========================================

python -u scripts/day5/reevaluate_cot.py \
    --results_dir /projects/prjs1800/results/day5/hotpotqa_2026_02_07_12_43_reranker_cot_qwen25_hotpotqa

# Re-evaluate MuSiQue Reranker+CoT
echo ==========================================
echo Re-evaluating MuSiQue Reranker+CoT
echo ==========================================

python -u scripts/day5/reevaluate_cot.py \
    --results_dir /projects/prjs1800/results/day5/musique_2026_02_07_13_23_reranker_cot_qwen25_musique

echo ==========================================
echo Re-evaluation complete
echo End time: $(date)
echo ==========================================
