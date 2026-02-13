#!/bin/bash
#SBATCH --job-name=d4_ircot_msq
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day4/ircot_msq_rerun_%j.log
#SBATCH --error=/projects/prjs1800/msc-thesis/jobs/output/day4/ircot_msq_rerun_%j.log

echo "=========================================="
echo "Day 4: IRCoT MuSiQue Re-run"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load 2024
module load Python/3.10.4-GCCcore-11.3.0

source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate

export HF_HOME=/projects/prjs1800/.cache/huggingface
export TRANSFORMERS_CACHE=/projects/prjs1800/.cache/huggingface
export PYTHONPATH=/projects/prjs1800/external/FlashRAG:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Thread settings for FAISS
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16

# CRITICAL: FAISS pip pkg ships broken single-threaded OpenBLAS
# Override with system multi-threaded OpenBLAS (67x speedup!)
export LD_PRELOAD=/usr/lib64/libopenblaso.so.0

cd /projects/prjs1800/msc-thesis

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "=========================================="
echo "Running IRCoT on MuSiQue"
echo "Start: $(date)"
echo "=========================================="

python scripts/day4/run_ircot_rag.py \
    --config configs/day4/ircot_qwen25_musique.yaml \
    --max_iter 5

echo "IRCoT MuSiQue completed: $(date)"
echo "=========================================="
echo "All experiments complete: $(date)"
