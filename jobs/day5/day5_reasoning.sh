#!/bin/bash
#SBATCH --job-name=d5_reasoning
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day5/reasoning_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 5: ReasoningPipeline (Search-R1 style) + SelfAsk
# Advanced single-agent reasoning approaches

echo ==========================================
echo Day 5: ReasoningPipeline + SelfAsk
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
export TOKENIZERS_PARALLELISM=false

# Enable parallel FAISS on CPU — CRITICAL for performance
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export FAISS_NUM_THREADS=16

# Force MKL as BLAS backend (bypasses numpy OpenBLAS MAX_THREADS=2 limitation)
export LD_PRELOAD=/projects/prjs1800/venvs/FlashRAG-venv/lib/libmkl_rt.so.2
export MKL_THREADING_LAYER=GNU

# GPU check
echo GPU Info:
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo

cd /projects/prjs1800/msc-thesis

# Experiment 1: ReasoningPipeline on HotpotQA
echo ==========================================
echo Running ReasoningPipeline on HotpotQA
echo Start: $(date)
echo ==========================================

python -u scripts/day5/run_reasoning_rag.py \
    --config configs/day5/reasoning_qwen25_hotpotqa.yaml \
    --max_retrieval 5

echo ReasoningPipeline HotpotQA completed: $(date)

# Experiment 2: ReasoningPipeline on MuSiQue
echo ==========================================
echo Running ReasoningPipeline on MuSiQue
echo Start: $(date)
echo ==========================================

python -u scripts/day5/run_reasoning_rag.py \
    --config configs/day5/reasoning_qwen25_musique.yaml \
    --max_retrieval 5

echo ReasoningPipeline MuSiQue completed: $(date)

# Experiment 3: SelfAsk on HotpotQA (multi-hop mode)
echo ==========================================
echo Running SelfAsk on HotpotQA
echo Start: $(date)
echo ==========================================

python -u scripts/day5/run_selfask_rag.py \
    --config configs/day5/selfask_qwen25_hotpotqa.yaml \
    --max_iter 5

echo SelfAsk HotpotQA completed: $(date)

# Experiment 4: SelfAsk on MuSiQue (multi-hop mode)
echo ==========================================
echo Running SelfAsk on MuSiQue
echo Start: $(date)
echo ==========================================

python -u scripts/day5/run_selfask_rag.py \
    --config configs/day5/selfask_qwen25_musique.yaml \
    --max_iter 5

echo SelfAsk MuSiQue completed: $(date)

echo ==========================================
echo Day 5 Reasoning+SelfAsk complete
echo End time: $(date)
echo ==========================================
