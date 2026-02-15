#!/bin/bash
#SBATCH --job-name=arag_expA
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/prjs1800/results/logs/%j_arag_expA.out
#SBATCH --error=/projects/prjs1800/results/logs/%j_arag_expA.err

set -euo pipefail

echo "=== ARAG Experiment A - $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# --- Environment ---
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate

export LD_PRELOAD=/projects/prjs1800/venvs/FlashRAG-venv/lib/libmkl_rt.so.2
export MKL_THREADING_LAYER=GNU
export FAISS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export TRANSFORMERS_CACHE=/projects/prjs1800/.cache/huggingface
export HF_HOME=/projects/prjs1800/.cache/huggingface

ROOT=/projects/prjs1800
EXP=$ROOT/msc-thesis/01-arag-reproduction
ARAG=$ROOT/external/arag
RESULTS=$ROOT/results/arag-expA

mkdir -p $RESULTS/logs
mkdir -p $EXP/data/questions
mkdir -p $EXP/data/index

# --- Step 0: Install arag + sentence-transformers ---
echo "=== Step 0: Installing dependencies - $(date) ==="
pip install -e $ARAG 2>&1 | tail -5
pip install sentence-transformers 2>&1 | tail -5
python -c "import arag; print(f'arag {arag.__version__} OK')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"

# --- Step 1: Build FTS5 index (if not exists) ---
if [ ! -f "$EXP/data/index/wiki18_fts.db" ]; then
    echo "=== Step 1: Building FTS5 index - $(date) ==="
    python -u $EXP/scripts/06_build_keyword_fts_index.py \
        --corpus_jsonl $ROOT/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl \
        --id_offset_json $EXP/data/index/wiki18_id_offset.json \
        --sqlite_db $EXP/data/index/wiki18_fts.db
    echo "FTS5 index done - $(date)"
else
    echo "=== Step 1: FTS5 index already exists, skipping ==="
fi

# --- Step 2: Prepare questions ---
echo "=== Step 2: Preparing questions - $(date) ==="
for ds in hotpotqa musique 2wikimultihopqa; do
    if [ ! -f "$EXP/data/questions/${ds}.json" ]; then
        python -u $EXP/scripts/01_prepare_questions.py \
            --dataset $ds \
            --output $EXP/data/questions/${ds}.json
        echo "  $ds: $(python -c "import json; print(len(json.load(open('$EXP/data/questions/${ds}.json'))))" ) questions"
    else
        echo "  $ds: already prepared"
    fi
done

# --- Step 3: Start vLLM server ---
echo "=== Step 3: Starting vLLM server - $(date) ==="
MODEL_PATH="$ROOT/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "Qwen2.5-7B-Instruct" \
    --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-log-requests \
    > $RESULTS/logs/vllm_server.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after $((i*5))s - $(date)"
        VLLM_READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check $RESULTS/logs/vllm_server.log"
        tail -50 $RESULTS/logs/vllm_server.log
        exit 1
    fi
    sleep 5
done

if [ "${VLLM_READY:-0}" != "1" ]; then
    echo "ERROR: vLLM failed to start within 600s"
    tail -50 $RESULTS/logs/vllm_server.log
    exit 1
fi

# Verify it's actually serving
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool || {
    echo "ERROR: vLLM not responding"
    cat $RESULTS/logs/vllm_server.log | tail -30
    exit 1
}

# --- Step 4: Run experiments ---
echo "=== Step 4: Running experiments - $(date) ==="
for ds in hotpotqa musique 2wikimultihopqa; do
    echo "--- Dataset: $ds - $(date) ---"
    python -u $EXP/scripts/03_run_expA.py \
        --dataset $ds \
        --workers 4
    echo "--- $ds done - $(date) ---"
done

# --- Step 5: Evaluate ---
echo "=== Step 5: Evaluating - $(date) ==="
for ds in hotpotqa musique 2wikimultihopqa; do
    PRED_FILE="$RESULTS/$ds/predictions.jsonl"
    if [ -f "$PRED_FILE" ]; then
        echo "--- $ds ---"
        python -u $EXP/scripts/04_eval_em_f1.py \
            --predictions "$PRED_FILE" \
            --output "$RESULTS/$ds/eval_em_f1.json"
    else
        echo "WARNING: No predictions for $ds"
    fi
done

# --- Cleanup ---
echo "=== Cleanup - $(date) ==="
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=== DONE - $(date) ==="
echo "Results in: $RESULTS"
