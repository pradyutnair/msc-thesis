#!/bin/bash
# ==========================================================================
# Submit all MA²RAG experiment jobs to SLURM.
#
# Usage:
#   # First launch vLLM server
#   VLLM_JOB=$(sbatch --parsable jobs/vllm_server.job)
#
#   # Then submit experiments (they wait for vLLM)
#   bash jobs/submit_all.sh $VLLM_JOB
#
#   # Or for LMCache experiments (M3):
#   VLLM_JOB=$(LMCACHE=1 sbatch --parsable jobs/vllm_server.job)
#   bash jobs/submit_all.sh $VLLM_JOB m3
# ==========================================================================

set -euo pipefail

VLLM_JOB_ID="${1:-}"
EXPERIMENT_FILTER="${2:-all}"  # "all", "m1", "m2", "m3", "m4", "ablations"

PROJECT_DIR="/projects/prjs1800/msc-thesis/02-arag-multi-agent"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
CONFIGS_DIR="$PROJECT_DIR/configs"

DATASETS=("hotpotqa" "musique" "2wiki")
DATASET_PATHS=(
    "/projects/prjs1800/msc-thesis/02-arag-multi-agent/data/hotpotqa/questions.json"
    "/projects/prjs1800/msc-thesis/02-arag-multi-agent/data/musique/questions.json"
    "/projects/prjs1800/msc-thesis/02-arag-multi-agent/data/2wikimultihop/questions.json"
)
LOCAL_DATA_DIRS=(
    "hotpotqa"
    "musique"
    "2wikimultihop"
)

LIMIT=1000
CONCURRENT=5

submit_experiment() {
    local exp_id="$1"
    local config="$2"
    local dep_arg=""

    if [ -n "$VLLM_JOB_ID" ]; then
        dep_arg="--dependency=afterok:$VLLM_JOB_ID"
    fi

    for i in "${!DATASETS[@]}"; do
        local ds="${DATASETS[$i]}"
        local questions="${DATASET_PATHS[$i]}"
        local data_dir="${LOCAL_DATA_DIRS[$i]}"
        local chunks="$PROJECT_DIR/data/$data_dir/chunks.json"
        local index_dir="$PROJECT_DIR/data/$data_dir/index_e5_base_v2"
        local output="$PROJECT_DIR/results/$exp_id/$ds/full"

        echo "Submitting $exp_id/$ds..."

        sbatch $dep_arg \
            --job-name="${exp_id}-${ds}" \
            --partition=gpu \
            --gres=gpu:1 \
            --time=08:00:00 \
            --mem=48G \
            --cpus-per-task=8 \
            --output="$PROJECT_DIR/results/logs/%j_${exp_id}_${ds}.out" \
            --wrap="
module load 2024
module load Python/3.11.5-GCCcore-13.2.0
source /projects/prjs1800/venvs/flashrag-venv/bin/activate
export HF_HOME=/projects/prjs1800/models
export TRANSFORMERS_CACHE=/projects/prjs1800/models
cd $PROJECT_DIR

# Wait for vLLM server
for attempt in \$(seq 1 60); do
    if curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
        break
    fi
    sleep 10
done

python $SCRIPTS_DIR/multi_agent_runner.py \
    --config $CONFIGS_DIR/$config \
    --questions $questions \
    --chunks-file $chunks \
    --index-dir $index_dir \
    --output $output \
    --limit $LIMIT \
    --concurrent $CONCURRENT
"
    done
}

# Primary experiments
if [ "$EXPERIMENT_FILTER" = "all" ] || [ "$EXPERIMENT_FILTER" = "m1" ]; then
    submit_experiment "m1" "m1_multi_agent.yaml"
fi

if [ "$EXPERIMENT_FILTER" = "all" ] || [ "$EXPERIMENT_FILTER" = "m2" ]; then
    submit_experiment "m2" "m2_doc_cache.yaml"
fi

if [ "$EXPERIMENT_FILTER" = "all" ] || [ "$EXPERIMENT_FILTER" = "m3" ]; then
    submit_experiment "m3" "m3_kv_cache.yaml"
fi

if [ "$EXPERIMENT_FILTER" = "all" ] || [ "$EXPERIMENT_FILTER" = "m4" ]; then
    submit_experiment "m4" "m4_single_2x.yaml"
fi

# Ablations
if [ "$EXPERIMENT_FILTER" = "all" ] || [ "$EXPERIMENT_FILTER" = "ablations" ]; then
    submit_experiment "a1" "ablations/a1_no_decomposer.yaml"
    submit_experiment "a2" "ablations/a2_no_aggregator.yaml"
    submit_experiment "a3" "ablations/a3_sequential.yaml"
    submit_experiment "a4" "ablations/a4_no_verify.yaml"
fi

echo "All jobs submitted."
