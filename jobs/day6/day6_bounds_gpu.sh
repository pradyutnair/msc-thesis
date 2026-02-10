#!/bin/bash
#SBATCH --job-name=d6_bounds
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day6/bounds_gpu_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 6 GPU Job: Naive Generation (x3) + Gold Context (x3) + Standard RAG 2Wiki + Reranker 2Wiki

echo ==========================================
echo "Day 6: Bounding Experiments (GPU)"
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

# Create output and results directories
mkdir -p /projects/prjs1800/msc-thesis/jobs/output/day6
mkdir -p /projects/prjs1800/results/day6

# ── A1: Naive Generation (no retrieval) ─────────────────────────────────────

echo ==========================================
echo "A1a: Naive Generation on HotpotQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_naive_generation.py \
    --config configs/day6/naive_gen_qwen25_hotpotqa.yaml

echo "Naive Gen HotpotQA completed: $(date)"

echo ==========================================
echo "A1b: Naive Generation on MuSiQue"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_naive_generation.py \
    --config configs/day6/naive_gen_qwen25_musique.yaml

echo "Naive Gen MuSiQue completed: $(date)"

echo ==========================================
echo "A1c: Naive Generation on 2WikiMultihopQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_naive_generation.py \
    --config configs/day6/naive_gen_qwen25_2wiki.yaml

echo "Naive Gen 2Wiki completed: $(date)"

# ── A2: Gold Context (perfect retrieval) ────────────────────────────────────

echo ==========================================
echo "A2a: Gold Context on HotpotQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_gold_context.py \
    --config configs/day6/gold_context_qwen25_hotpotqa.yaml

echo "Gold Context HotpotQA completed: $(date)"

echo ==========================================
echo "A2b: Gold Context on MuSiQue"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_gold_context.py \
    --config configs/day6/gold_context_qwen25_musique.yaml

echo "Gold Context MuSiQue completed: $(date)"

echo ==========================================
echo "A2c: Gold Context on 2WikiMultihopQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_gold_context.py \
    --config configs/day6/gold_context_qwen25_2wiki.yaml

echo "Gold Context 2Wiki completed: $(date)"

# ── A3: Standard RAG on 2WikiMultihopQA ─────────────────────────────────────

echo ==========================================
echo "A3: Standard RAG on 2WikiMultihopQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_standard_pipeline.py \
    --config configs/day6/standard_rag_qwen25_2wiki.yaml

echo "Standard RAG 2Wiki completed: $(date)"

# ── A4: Reranker on 2WikiMultihopQA ─────────────────────────────────────────

echo ==========================================
echo "A4: Reranker on 2WikiMultihopQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_standard_pipeline.py \
    --config configs/day6/reranker_qwen25_2wiki.yaml

echo "Reranker 2Wiki completed: $(date)"

echo ==========================================
echo "Day 6 GPU experiments complete"
echo "End time: $(date)"
echo ==========================================

# List all Day 6 results
echo "Day 6 result directories:"
ls -la /projects/prjs1800/results/day6/
echo
echo "Metric scores:"
for d in /projects/prjs1800/results/day6/*/; do
    if [ -f "$d/metric_score.txt" ]; then
        echo "--- $d ---"
        cat "$d/metric_score.txt"
    fi
done
