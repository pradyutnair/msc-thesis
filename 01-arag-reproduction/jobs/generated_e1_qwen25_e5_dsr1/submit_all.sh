#!/bin/bash
set -euo pipefail
JOB_DIR="/projects/prjs1800/msc-thesis/01-arag-reproduction/jobs/generated_e1_qwen25_e5_dsr1"

echo "Submitting E1 rerun: Qwen2.5-7B + E5 + DeepSeek judge"
GEN_HP=$(sbatch "$JOB_DIR/gen_hotpotqa.job" | awk '{print $4}')
GEN_MU=$(sbatch "$JOB_DIR/gen_musique.job" | awk '{print $4}')
GEN_2W=$(sbatch "$JOB_DIR/gen_2wikimultihop.job" | awk '{print $4}')

echo "Generation jobs:"
echo "  hotpotqa:      $GEN_HP"
echo "  musique:       $GEN_MU"
echo "  2wikimultihop: $GEN_2W"

EVAL_HP=$(sbatch --dependency=afterok:${GEN_HP} "$JOB_DIR/eval_hotpotqa.job" | awk '{print $4}')
EVAL_MU=$(sbatch --dependency=afterok:${GEN_MU} "$JOB_DIR/eval_musique.job" | awk '{print $4}')
EVAL_2W=$(sbatch --dependency=afterok:${GEN_2W} "$JOB_DIR/eval_2wikimultihop.job" | awk '{print $4}')

echo "Eval jobs (after generation):"
echo "  hotpotqa:      $EVAL_HP (after $GEN_HP)"
echo "  musique:       $EVAL_MU (after $GEN_MU)"
echo "  2wikimultihop: $EVAL_2W (after $GEN_2W)"
