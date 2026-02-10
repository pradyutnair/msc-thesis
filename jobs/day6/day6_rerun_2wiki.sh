#!/bin/bash
#SBATCH --job-name=d6_rerun_2wiki
#SBATCH --partition=gpu_a100
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/day6/rerun_2wiki_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# Day 6 Re-run: Gold Context 2Wiki (fixed extractor) + Standard RAG 2Wiki + Reranker 2Wiki

echo ==========================================
echo "Day 6 Re-run: 2Wiki experiments"
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
echo

cd /projects/prjs1800/msc-thesis

# Remove broken 2Wiki gold context result
echo "Removing broken 2Wiki gold context result..."
rm -rf /projects/prjs1800/results/day6/*gold*2wiki*

# 1. Gold Context on 2WikiMultihopQA (fixed extractor)
echo ==========================================
echo "Gold Context on 2WikiMultihopQA (FIXED)"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_gold_context.py \
    --config configs/day6/gold_context_qwen25_2wiki.yaml

echo "Gold Context 2Wiki completed: $(date)"

# 2. Standard RAG on 2WikiMultihopQA
echo ==========================================
echo "Standard RAG on 2WikiMultihopQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_standard_pipeline.py \
    --config configs/day6/standard_rag_qwen25_2wiki.yaml

echo "Standard RAG 2Wiki completed: $(date)"

# 3. Reranker on 2WikiMultihopQA
echo ==========================================
echo "Reranker on 2WikiMultihopQA"
echo "Start: $(date)"
echo ==========================================

python -u scripts/day6/run_standard_pipeline.py \
    --config configs/day6/reranker_qwen25_2wiki.yaml

echo "Reranker 2Wiki completed: $(date)"

echo ==========================================
echo "Day 6 Re-run complete"
echo "End time: $(date)"
echo ==========================================

# Show results
echo "Metric scores:"
for d in /projects/prjs1800/results/day6/*/; do
    if [ -f "$d/metric_score.txt" ]; then
        echo "--- $(basename $d) ---"
        cat "$d/metric_score.txt"
    fi
done
