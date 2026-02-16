#!/bin/bash
# Submit A-RAG (Qwen3-8B via local vLLM) for musique/hotpotqa/2wikimultihop.
#
# Defaults are set for full reruns to avoid checkpoint-resume skipping.
# Override with env vars when needed, e.g.:
#   FORCE_RERUN=0 LIMIT=50 WORKERS=4 bash jobs/3_expA_qwen3_arag.sh

set -euo pipefail

ROOT="/projects/prjs1800/msc-thesis/01-arag-reproduction"
JOBS_DIR="$ROOT/jobs"
OUT_DIR="$ROOT/jobs/output"
mkdir -p "$JOBS_DIR" "$OUT_DIR"

PARTITION="${PARTITION:-gpu_a100}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
WORKERS="${WORKERS:-2}"
FORCE_RERUN="${FORCE_RERUN:-1}"
LIMIT="${LIMIT:-}"

submit_one() {
  local dataset="$1"
  local cfg="$2"
  local qfile="$3"
  local run_out="$4"

  [[ -f "$cfg" ]] || { echo "Missing config: $cfg" >&2; return 1; }
  [[ -f "$qfile" ]] || { echo "Missing questions: $qfile" >&2; return 1; }

  local job_file="$JOBS_DIR/auto_${dataset}_qwen3_vllm.job"

  cat > "$job_file" <<JOB
#!/bin/bash
#SBATCH --job-name=arag_q3_${dataset}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus=1
#SBATCH --mem=${MEM}
#SBATCH --output=${OUT_DIR}/arag_q3_${dataset}_%j.log

set -euo pipefail

module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

cd /projects/prjs1800/external/arag
source /projects/prjs1800/venvs/arag-venv/bin/activate
export PYTHONNOUSERSITE=1

export HF_HOME="/projects/prjs1800/.cache/huggingface"
export HF_HUB_CACHE="\$HF_HOME"
export TRANSFORMERS_CACHE="\$HF_HOME"
export HF_DATASETS_CACHE="\$HF_HOME"
export TOKENIZERS_PARALLELISM=false

export ARAG_API_KEY="dummy"
export ARAG_BASE_URL="http://127.0.0.1:8000/v1"
export ARAG_MODEL="Qwen3-8B"

MODEL_ID="Qwen/Qwen3-8B"
SERVED_NAME="Qwen3-8B"
HOST="127.0.0.1"
PORT="8000"
RUN_OUT="${run_out}"

mkdir -p "\$RUN_OUT"

# Remove checkpoint file to force a fresh pass (prevents Completed=N/Pending=0 no-op runs).
if [[ "${FORCE_RERUN}" == "1" ]]; then
  rm -f "\$RUN_OUT/predictions.jsonl"
fi

cleanup() {
  if [[ -n "\${VLLM_PID:-}" ]] && kill -0 "\$VLLM_PID" 2>/dev/null; then
    kill "\$VLLM_PID" || true
    wait "\$VLLM_PID" || true
  fi
}
trap cleanup EXIT TERM INT

vllm serve "\$MODEL_ID" \
  --served-model-name "\$SERVED_NAME" \
  --host "\$HOST" \
  --port "\$PORT" \
  --dtype auto \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code &
VLLM_PID=\$!

READY_URL="http://\$HOST:\$PORT/v1/models"
MAX_RETRIES=360
for i in \$(seq 1 "\$MAX_RETRIES"); do
  if ! kill -0 "\$VLLM_PID" 2>/dev/null; then
    echo "vLLM process exited before readiness."
    wait "\$VLLM_PID" || true
    exit 1
  fi

  resp="\$(curl -fsS "\$READY_URL" 2>/dev/null || true)"
  if [[ -n "\$resp" ]] && echo "\$resp" | python -c 'import json,sys; d=json.load(sys.stdin); print(any(m.get("id")=="Qwen3-8B" for m in d.get("data",[])))' | grep -q True; then
    echo "vLLM ready for ${dataset}"
    break
  fi
  sleep 5
done

curl -fsS "\$READY_URL" >/dev/null

CMD=(
  uv run --active python /projects/prjs1800/external/arag/scripts/batch_runner.py
  --config "${cfg}"
  --questions "${qfile}"
  --output "\$RUN_OUT"
  --workers "${WORKERS}"
)
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

"\${CMD[@]}"
JOB

  chmod +x "$job_file"
  local jid
  jid=$(sbatch "$job_file" | awk '{print $4}')
  echo "${dataset}: ${jid} (${job_file})"
}

submit_one \
  "musique" \
  "$ROOT/configs/arag_qwen3_vllm_e5_musique.yaml" \
  "/projects/prjs1800/external/arag/data/musique/questions.json" \
  "$ROOT/results/qwen3-8b-vllm/musique"

submit_one \
  "hotpotqa" \
  "$ROOT/configs/arag_qwen3_vllm_e5_hotpotqa.yaml" \
  "/projects/prjs1800/external/arag/data/hotpotqa/questions.json" \
  "$ROOT/results/qwen3-8b-vllm/hotpotqa"

submit_one \
  "2wikimultihop" \
  "$ROOT/configs/arag_qwen3_vllm_e5_2wikimultihop.yaml" \
  "/projects/prjs1800/external/arag/data/2wikimultihop/questions.json" \
  "$ROOT/results/qwen3-8b-vllm/2wikimultihop"

echo "Submitted all dataset jobs."
