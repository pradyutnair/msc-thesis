#!/bin/bash
set -euo pipefail

ROOT="/projects/prjs1800/msc-thesis/01-arag-reproduction"
JOB_DIR="$ROOT/jobs/generated_b0_nonagentic_q3"
OUT_LOG_DIR="$ROOT/jobs/output"
mkdir -p "$JOB_DIR" "$OUT_LOG_DIR"

DATASETS=("hotpotqa" "musique" "2wikimultihop")

MODELS=(
  "q3_8b|Qwen/Qwen3-8B|Qwen3-8B|gpu_a100|12:00:00|arag_qwen3_vllm_e5|b0-qwen3-8b-e5-deepseekr1"
  "q3_30b|Qwen/Qwen3-30B-A3B|Qwen3-30B-A3B|gpu_h100|12:00:00|arag_qwen3_30b_vllm_e5|b0-qwen3-30b-e5-deepseekr1"
)

question_file_for_dataset() {
  local ds="$1"
  echo "/projects/prjs1800/external/arag/data/${ds}/questions.json"
}

submit_chain() {
  local model_key="$1"
  local model_id="$2"
  local served_name="$3"
  local gen_partition="$4"
  local gen_time="$5"
  local config_prefix="$6"
  local result_root_name="$7"
  local ds="$8"

  local qfile
  qfile=$(question_file_for_dataset "$ds")
  local cfg="$ROOT/configs/${config_prefix}_${ds}.yaml"
  local result_dir="$ROOT/results/${result_root_name}/${ds}"

  local gen_job="$JOB_DIR/gen_${model_key}_${ds}.job"
  local eval_job="$JOB_DIR/eval_${model_key}_${ds}.job"

  cat > "$gen_job" <<JOB
#!/bin/bash
#SBATCH --job-name=b0g_${model_key}_${ds}
#SBATCH --partition=${gen_partition}
#SBATCH --time=${gen_time}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=${OUT_LOG_DIR}/gen_b0_${model_key}_${ds}_%j.log
set -euo pipefail
module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0
cd /projects/prjs1800/external/arag
source /projects/prjs1800/venvs/arag-venv/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=/projects/prjs1800/.cache/huggingface
export HF_HUB_CACHE=\$HF_HOME
export TRANSFORMERS_CACHE=\$HF_HOME
export HF_DATASETS_CACHE=\$HF_HOME
export TOKENIZERS_PARALLELISM=false

export ARAG_API_KEY=dummy
HOST="127.0.0.1"
PORT="\${ARAG_PORT:-\$((7000 + (SLURM_JOB_ID % 1000)))}"
export ARAG_BASE_URL="http://\$HOST:\$PORT/v1"
export ARAG_MODEL="${served_name}"

MODEL_ID="${model_id}"
SERVED_NAME="${served_name}"
RESULT_DIR="${result_dir}"
mkdir -p "\$RESULT_DIR"

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
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --trust-remote-code &
VLLM_PID=\$!

READY_URL="http://\$HOST:\$PORT/v1/models"
for i in \$(seq 1 360); do
  if ! kill -0 "\$VLLM_PID" 2>/dev/null; then
    wait "\$VLLM_PID" || true
    exit 1
  fi
  resp="\$(curl -fsS "\$READY_URL" 2>/dev/null || true)"
  if [[ -n "\$resp" ]] && echo "\$resp" | python -c "import json,sys; d=json.load(sys.stdin); print(any(m.get('id')=='\$SERVED_NAME' for m in d.get('data',[])))" | grep -q True; then
    break
  fi
  sleep 5
done
curl -fsS "\$READY_URL" >/dev/null

uv run --active python /projects/prjs1800/msc-thesis/01-arag-reproduction/scripts/b0_non_agentic_runner.py \
  --config "${cfg}" \
  --questions "${qfile}" \
  --output "\$RESULT_DIR" \
  --top-k 5 \
  --max-answer-tokens 128 \
  --overwrite
JOB

  cat > "$eval_job" <<JOB
#!/bin/bash
#SBATCH --job-name=b0e_${model_key}_${ds}
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=${OUT_LOG_DIR}/eval_b0_${model_key}_${ds}_%j.log
set -euo pipefail
module purge
module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0
cd /projects/prjs1800/external/arag
source /projects/prjs1800/venvs/arag-venv/bin/activate
export PYTHONNOUSERSITE=1
export HF_HOME=/projects/prjs1800/.cache/huggingface
export HF_HUB_CACHE=\$HF_HOME
export TRANSFORMERS_CACHE=\$HF_HOME
export HF_DATASETS_CACHE=\$HF_HOME
export TOKENIZERS_PARALLELISM=false

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
SERVED_NAME="DeepSeek-R1-Distill-Qwen-32B"
HOST="127.0.0.1"
PORT="\${ARAG_PORT:-\$((8000 + (SLURM_JOB_ID % 1000)))}"

export ARAG_API_KEY=dummy
export ARAG_BASE_URL="http://\$HOST:\$PORT/v1"
export ARAG_MODEL="\$SERVED_NAME"
RESULT_DIR="${result_dir}"

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
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code &
VLLM_PID=\$!

READY_URL="http://\$HOST:\$PORT/v1/models"
for i in \$(seq 1 600); do
  if ! kill -0 "\$VLLM_PID" 2>/dev/null; then
    wait "\$VLLM_PID" || true
    exit 1
  fi
  resp="\$(curl -fsS "\$READY_URL" 2>/dev/null || true)"
  if [[ -n "\$resp" ]] && echo "\$resp" | python -c "import json,sys; d=json.load(sys.stdin); print(any(m.get('id')=='\$SERVED_NAME' for m in d.get('data',[])))" | grep -q True; then
    break
  fi
  sleep 5
done
curl -fsS "\$READY_URL" >/dev/null

uv run --active python /projects/prjs1800/external/arag/scripts/eval.py \
  --predictions "\$RESULT_DIR/predictions.jsonl" \
  --workers 10 \
  --output "\$RESULT_DIR"
JOB

  chmod +x "$gen_job" "$eval_job"

  local gen_id eval_id
  gen_id=$(sbatch "$gen_job" | awk '{print $4}')
  eval_id=$(sbatch --dependency=afterok:${gen_id} "$eval_job" | awk '{print $4}')

  echo "${model_key}/${ds}: gen=${gen_id} eval=${eval_id}"
}

echo "Submitting B0 non-agentic Q3-only pipelines"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r model_key model_id served_name gen_partition gen_time config_prefix result_root_name <<< "$entry"
  for ds in "${DATASETS[@]}"; do
    submit_chain "$model_key" "$model_id" "$served_name" "$gen_partition" "$gen_time" "$config_prefix" "$result_root_name" "$ds"
  done
done

echo "All Q3 B0 jobs submitted."
