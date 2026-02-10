#!/bin/bash
#SBATCH --job-name=d6_rnk_2wiki
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day6/reranker_2wiki_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 6: Reranker on 2WikiMultihopQA (two-phase approach to avoid OOM)

echo ==========================================
echo "Day 6: Reranker on 2WikiMultihopQA"
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

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export FAISS_NUM_THREADS=16

export LD_PRELOAD=/projects/prjs1800/venvs/FlashRAG-venv/lib/libmkl_rt.so.2
export MKL_THREADING_LAYER=GNU

echo GPU Info:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

cd /projects/prjs1800/msc-thesis

python -u scripts/day6/run_reranker_rag.py \
    --config configs/day6/reranker_qwen25_2wiki.yaml

echo ==========================================
echo "Reranker 2Wiki complete"
echo "End time: $(date)"
echo ==========================================
