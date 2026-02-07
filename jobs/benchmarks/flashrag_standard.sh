#!/bin/bash
#SBATCH --job-name=fr_std
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH --output=/projects/prjs1800/msc-thesis/jobs/output/flashrag_standard_%j.log
#SBATCH --mail-user=pradyut.nair@student.uva.nl
#SBATCH --mail-type=END,FAIL

# FlashRAG Standard RAG (retrieve + generate) on NQ, TriviaQA, HotpotQA, 2Wiki, PopQA, WebQA

echo "=========================================="
echo "FlashRAG Standard RAG Baseline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

cd /projects/prjs1800
source venvs/FlashRAG-venv/bin/activate

FLASHRAG_ROOT="/projects/prjs1800/external/FlashRAG"
FLASHRAG_METHODS="$FLASHRAG_ROOT/examples/methods"
export PYTHONPATH="$FLASHRAG_ROOT:$PYTHONPATH"
export HF_HOME="/projects/prjs1800/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
# Config file in msc-thesis for version control
FLASHRAG_CONFIG="/projects/prjs1800/msc-thesis/configs/benchmarks/standard_rag.yaml"
DATASETS="nq triviaqa hotpotqa 2wikimultihopqa popqa web_questions"

cd "$FLASHRAG_METHODS"

# Enable parallel FAISS on CPU (FAISS GPU disabled due to memory constraints)
export OMP_NUM_THREADS=16
export FAISS_OPT_LEVEL=avx2

# Debug: Check GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

for DATASET in $DATASETS; do
  echo "----------------------------------------"
  echo "Dataset: $DATASET (standard_rag)"
  echo "----------------------------------------"
  python run_exp.py \
    --config "$FLASHRAG_CONFIG" \
    --method_name standard_rag \
    --dataset_name "$DATASET" \
    --split test \
    --gpu_id "0,1" || exit 1
done

echo "=========================================="
echo "Completed FlashRAG Standard RAG (all datasets)"
echo "End time: $(date)"
echo "=========================================="
