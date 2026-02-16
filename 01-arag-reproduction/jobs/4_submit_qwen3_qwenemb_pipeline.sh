#!/bin/bash
set -euo pipefail

ROOT="/projects/prjs1800/msc-thesis/01-arag-reproduction"
OUT_DIR="$ROOT/jobs/output"
GEN_DIR="$ROOT/jobs/generated_qwen3_qwenemb"
RESULT_ROOT="$ROOT/results/qwen3-8b-qwen-emb-vllm"
mkdir -p "$OUT_DIR" "$GEN_DIR" "$RESULT_ROOT"

GEN_WORKERS="${GEN_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-10}"
PART_A100="${PART_A100:-gpu_a100}"
PART_H100="${PART_H100:-gpu_h100}"

submit_dataset_chain() {
  local ds="$1"
  local chunks="$2"
  local questions="$3"
  local index_dir="$4"
  local config="$5"
  local result_dir="$6"

  local build_job="$GEN_DIR/build_${ds}_qwenemb.job"
  local gen_job="$GEN_DIR/gen_${ds}_qwen3_qwenemb.job"
  local eval_job="$GEN_DIR/eval_${ds}_qwen3_30b.job"

  cat > "$build_job" <<JOB
#!/bin/bash
#SBATCH --job-name=build_${ds}_qemb
#SBATCH --partition=${PART_A100}
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=${OUT_DIR}/build_${ds}_qemb_%j.log
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
mkdir -p "${index_dir}"
uv run --active python /projects/prjs1800/external/arag/scripts/build_index.py \
  --chunks "${chunks}" \
  --output "${index_dir}" \
  --model "Qwen/Qwen3-Embedding-0.6B" \
  --device cuda:0
JOB

  cat > "$gen_job" <<JOB
#!/bin/bash
#SBATCH --job-name=gen_${ds}_q3qemb
#SBATCH --partition=${PART_A100}
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=${OUT_DIR}/gen_${ds}_q3qemb_%j.log
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
export ARAG_BASE_URL=http://127.0.0.1:8000/v1
export ARAG_MODEL=Qwen3-8B
MODEL_ID="Qwen/Qwen3-8B"
SERVED_NAME="Qwen3-8B"
HOST="127.0.0.1"
PORT="8000"
mkdir -p "${result_dir}"
rm -f "${result_dir}/predictions.jsonl"
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
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code &
VLLM_PID=\$!
READY_URL="http://\$HOST:\$PORT/v1/models"
for i in \$(seq 1 360); do
  if ! kill -0 "\$VLLM_PID" 2>/dev/null; then
    wait "\$VLLM_PID" || true
    exit 1
  fi
  resp="\$(curl -fsS "\$READY_URL" 2>/dev/null || true)"
  if [[ -n "\$resp" ]] && echo "\$resp" | python -c 'import json,sys; d=json.load(sys.stdin); print(any(m.get("id")=="Qwen3-8B" for m in d.get("data",[])))' | grep -q True; then
    break
  fi
  sleep 5
done
curl -fsS "\$READY_URL" >/dev/null
uv run --active python /projects/prjs1800/external/arag/scripts/batch_runner.py \
  --config "${config}" \
  --questions "${questions}" \
  --output "${result_dir}" \
  --workers "${GEN_WORKERS}"
JOB

  cat > "$eval_job" <<JOB
#!/bin/bash
#SBATCH --job-name=eval_${ds}_q330b
#SBATCH --partition=${PART_H100}
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --output=${OUT_DIR}/eval_${ds}_q330b_%j.log
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
MODEL_ID="Qwen/Qwen3-30B-A3B"
SERVED_NAME="Qwen3-30B-A3B"
HOST="127.0.0.1"
PORT="8000"
export ARAG_API_KEY=dummy
export ARAG_BASE_URL="http://\$HOST:\$PORT/v1"
export ARAG_MODEL="\$SERVED_NAME"
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
  if [[ -n "\$resp" ]] && echo "\$resp" | python -c 'import json,sys; d=json.load(sys.stdin); print(any(m.get("id")=="Qwen3-30B-A3B" for m in d.get("data",[])))' | grep -q True; then
    break
  fi
  sleep 5
done
curl -fsS "\$READY_URL" >/dev/null
uv run --active python /projects/prjs1800/external/arag/scripts/eval.py \
  --predictions "${result_dir}/predictions.jsonl" \
  --workers "${EVAL_WORKERS}" \
  --output "${result_dir}"
JOB

  chmod +x "$build_job" "$gen_job" "$eval_job"

  local bjid gjid ejid
  bjid=$(sbatch "$build_job" | awk '{print $4}')
  gjid=$(sbatch --dependency="afterok:${bjid}" "$gen_job" | awk '{print $4}')
  ejid=$(sbatch --dependency="afterok:${gjid}" "$eval_job" | awk '{print $4}')

  echo "${ds}: build=${bjid} gen=${gjid} eval=${ejid}"
}

submit_dataset_chain \
  "musique" \
  "/projects/prjs1800/external/arag/data/musique/chunks.json" \
  "/projects/prjs1800/external/arag/data/musique/questions.json" \
  "/projects/prjs1800/external/arag/data/musique/index_qwen3_embedding_06b" \
  "/projects/prjs1800/msc-thesis/01-arag-reproduction/configs/arag_qwen3_vllm_qwenemb_musique.yaml" \
  "/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-8b-qwen-emb-vllm/musique"

submit_dataset_chain \
  "hotpotqa" \
  "/projects/prjs1800/external/arag/data/hotpotqa/chunks.json" \
  "/projects/prjs1800/external/arag/data/hotpotqa/questions.json" \
  "/projects/prjs1800/external/arag/data/hotpotqa/index_qwen3_embedding_06b" \
  "/projects/prjs1800/msc-thesis/01-arag-reproduction/configs/arag_qwen3_vllm_qwenemb_hotpotqa.yaml" \
  "/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-8b-qwen-emb-vllm/hotpotqa"

submit_dataset_chain \
  "2wikimultihop" \
  "/projects/prjs1800/external/arag/data/2wikimultihop/chunks.json" \
  "/projects/prjs1800/external/arag/data/2wikimultihop/questions.json" \
  "/projects/prjs1800/external/arag/data/2wikimultihop/index_qwen3_embedding_06b" \
  "/projects/prjs1800/msc-thesis/01-arag-reproduction/configs/arag_qwen3_vllm_qwenemb_2wikimultihop.yaml" \
  "/projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-8b-qwen-emb-vllm/2wikimultihop"
