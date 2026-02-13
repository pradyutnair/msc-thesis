#!/bin/bash
#SBATCH --job-name=d4_ircot
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/results/day4/logs/ircot_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 4: IRCoT -- Iterative Retrieval + Chain-of-Thought
# Runs IRCoT on HotpotQA and MuSiQue with per-round tracking

echo ==========================================
echo Day 4: IRCoT Iterative Retrieval
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

# Enable parallel FAISS on CPU
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# GPU check
echo GPU Info:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo

cd /projects/prjs1800/msc-thesis

# Experiment 1: IRCoT on HotpotQA
echo ==========================================
echo Running IRCoT on HotpotQA
echo Start: $(date)
echo ==========================================

python scripts/day4/run_ircot_rag.py     --config configs/day4/ircot_qwen25_hotpotqa.yaml     --max_iter 5

echo IRCoT HotpotQA completed: $(date)

# Experiment 2: IRCoT on MuSiQue
echo ==========================================
echo Running IRCoT on MuSiQue
echo Start: $(date)
echo ==========================================

python scripts/day4/run_ircot_rag.py     --config configs/day4/ircot_qwen25_musique.yaml     --max_iter 5

echo IRCoT MuSiQue completed: $(date)

echo ==========================================
echo Day 4 IRCoT complete
echo End time: $(date)
echo ==========================================
