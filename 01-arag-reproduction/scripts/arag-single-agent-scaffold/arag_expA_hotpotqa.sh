#!/bin/bash
#SBATCH --job-name=arag_hotpotqa
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --output=/projects/prjs1800/results/logs/%j_arag_hotpotqa.out
#SBATCH --error=/projects/prjs1800/results/logs/%j_arag_hotpotqa.err

set -euo pipefail

DATASET="hotpotqa"
PORT="8001"

echo "=== ARAG Experiment A ($DATASET) - $(date) ==="
echo "Job ID: $SLURM_JOB_ID, Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# --- Environment ---
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate

export LD_PRELOAD=/projects/prjs1800/venvs/FlashRAG-venv/lib/libmkl_rt.so.2
export MKL_THREADING_LAYER=GNU
export FAISS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export HF_HOME=/projects/prjs1800/.cache/huggingface

ROOT=/projects/prjs1800
EXP=$ROOT/msc-thesis/01-arag-reproduction
ARAG=$ROOT/external/arag
RESULTS=$ROOT/results/arag-expA

mkdir -p $RESULTS/$DATASET $RESULTS/logs

# --- Quick dep check ---
python -c "import arag; print(f'arag {arag.__version__} OK')" || {
    pip install -e $ARAG 2>&1 | tail -3
    pip install sentence-transformers 2>&1 | tail -3
}

# --- Clear old broken results for this dataset ---
rm -f $RESULTS/$DATASET/predictions.jsonl

# --- Write per-dataset config with correct port ---
CFG=$RESULTS/$DATASET/config.yaml
cat > $CFG << CFGEOF
llm:
  model: "Qwen2.5-7B-Instruct"
  api_key: "dummy"
  base_url: "http://127.0.0.1:${PORT}/v1"
  temperature: 0.0
  max_tokens: 1024

agent:
  max_loops: 10
  max_token_budget: 128000
  verbose: false

data:
  corpus_jsonl: "$ROOT/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl"
  id_offset_json: "$EXP/data/index/wiki18_id_offset.json"
  sqlite_db: "$EXP/data/index/wiki18_fts.db"
  faiss_index_path: "$ROOT/datasets/flashrag/indexes/e5_Flat.index"
  embedding_model: "intfloat/e5-base-v2"
  embedding_device: "cuda:0"

output:
  results_dir: "$RESULTS"
CFGEOF

# --- Start vLLM server ---
echo "=== Starting vLLM server on port $PORT - $(date) ==="
MODEL_PATH="$ROOT/.cache/huggingface/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "Qwen2.5-7B-Instruct" \
    --port $PORT \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --disable-log-requests \
    > $RESULTS/logs/vllm_${DATASET}.log 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

VLLM_READY=0
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
        echo "vLLM ready after $((i*5))s - $(date)"
        VLLM_READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM died"; tail -30 $RESULTS/logs/vllm_${DATASET}.log; exit 1
    fi
    sleep 5
done
[ "$VLLM_READY" = "1" ] || { echo "ERROR: vLLM timeout"; exit 1; }

# --- Run experiment ---
echo "=== Running $DATASET - $(date) ==="
python -u $EXP/scripts/03_run_expA.py \
    --dataset $DATASET \
    --workers 4 \
    --config $CFG
echo "=== $DATASET inference done - $(date) ==="

# --- Evaluate ---
PRED_FILE="$RESULTS/$DATASET/predictions.jsonl"
if [ -f "$PRED_FILE" ]; then
    python -u $EXP/scripts/04_eval_em_f1.py \
        --predictions "$PRED_FILE" \
        --output "$RESULTS/$DATASET/eval_em_f1.json"
fi

# --- Cleanup ---
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "=== DONE ($DATASET) - $(date) ==="
