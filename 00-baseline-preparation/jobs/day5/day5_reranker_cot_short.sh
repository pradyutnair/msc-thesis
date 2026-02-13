#!/bin/bash
#SBATCH --job-name=d5_rerank_cot_short
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day5/reranker_cot_short_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 5: Reranker + CoT with max_tokens=32
# Controls for max_tokens confound: same CoT prompt but short output like Day 1/2

echo ==========================================
echo "Day 5: Reranker + CoT max_tokens=32"
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
export HF_HUB_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

# Enable parallel FAISS on CPU
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export FAISS_NUM_THREADS=16

# Force MKL as BLAS backend
export LD_PRELOAD=/projects/prjs1800/venvs/FlashRAG-venv/lib/libmkl_rt.so.2
export MKL_THREADING_LAYER=GNU

echo GPU Info:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo

cd /projects/prjs1800/msc-thesis

# Experiment 1: Reranker + CoT (short) on HotpotQA
echo ==========================================
echo Running Reranker + CoT short on HotpotQA
echo "Start: $(date)"
echo ==========================================

python -u scripts/day5/run_reranker_cot_rag.py \
    --config configs/day5/reranker_cot_short_qwen25_hotpotqa.yaml

echo "Reranker+CoT short HotpotQA completed: $(date)"

# Experiment 2: Reranker + CoT (short) on MuSiQue
echo ==========================================
echo Running Reranker + CoT short on MuSiQue
echo "Start: $(date)"
echo ==========================================

python -u scripts/day5/run_reranker_cot_rag.py \
    --config configs/day5/reranker_cot_short_qwen25_musique.yaml

echo "Reranker+CoT short MuSiQue completed: $(date)"

echo ==========================================
echo Day 5 Reranker+CoT short complete
echo "End time: $(date)"
echo ==========================================
