#!/bin/bash
#SBATCH --job-name=d4_flare
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/results/day4/logs/flare_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 4: FLARE -- Forward-Looking Active Retrieval
# Note: FLARE is sequential, expect 3-5 hours total

echo ==========================================
echo Day 4: FLARE Active Retrieval
echo Job ID: $SLURM_JOB_ID
echo Node: $SLURM_NODELIST
echo Start time: $(date)
echo ==========================================

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate

export PYTHONPATH=/projects/prjs1800/external/FlashRAG:$PYTHONPATH
export HF_HOME=/projects/prjs1800/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

echo GPU Info:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo

cd /projects/prjs1800/msc-thesis

# Experiment 1: FLARE on HotpotQA
echo ==========================================
echo Running FLARE on HotpotQA
echo Start: $(date)
echo ==========================================

python scripts/day4/run_flare_rag.py     --config configs/day4/flare_qwen25_hotpotqa.yaml     --threshold 0.2     --look_ahead_steps 64     --max_generation_length 256     --max_iter_num 5

echo FLARE HotpotQA completed: $(date)

# Experiment 2: FLARE on MuSiQue
echo ==========================================
echo Running FLARE on MuSiQue
echo Start: $(date)
echo ==========================================

python scripts/day4/run_flare_rag.py     --config configs/day4/flare_qwen25_musique.yaml     --threshold 0.2     --look_ahead_steps 64     --max_generation_length 256     --max_iter_num 5

echo FLARE MuSiQue completed: $(date)

echo ==========================================
echo Day 4 FLARE complete
echo End time: $(date)
echo ==========================================
