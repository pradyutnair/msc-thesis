#\!/bin/bash
set -euo pipefail

JOB_DIR="/projects/prjs1800/msc-thesis/01-arag-reproduction/jobs/generated_qwen3_30b_e5_dsr1"

echo "=== Submitting Qwen3-30B + E5 + DeepSeek-R1 Pipeline ==="
echo "Generator: Qwen3-30B-A3B | Embedder: E5-base-v2 | Judge: DeepSeek-R1-Distill-Qwen-32B"
echo ""

# Submit 3 generation jobs in parallel (no dependencies between them)
GEN_HP=$(sbatch "$JOB_DIR/gen_hotpotqa.job" | awk "{print \$4}")
GEN_MU=$(sbatch "$JOB_DIR/gen_musique.job" | awk "{print \$4}")
GEN_2W=$(sbatch "$JOB_DIR/gen_2wikimultihop.job" | awk "{print \$4}")

echo "Generation jobs submitted:"
echo "  hotpotqa:      $GEN_HP"
echo "  musique:       $GEN_MU"
echo "  2wikimultihop: $GEN_2W"
echo ""

# Submit eval jobs as dependencies (each eval waits for its gen job)
EVAL_HP=$(sbatch --dependency=afterok:${GEN_HP} "$JOB_DIR/eval_hotpotqa.job" | awk "{print \$4}")
EVAL_MU=$(sbatch --dependency=afterok:${GEN_MU} "$JOB_DIR/eval_musique.job" | awk "{print \$4}")
EVAL_2W=$(sbatch --dependency=afterok:${GEN_2W} "$JOB_DIR/eval_2wikimultihop.job" | awk "{print \$4}")

echo "Eval jobs submitted (with dependencies):"
echo "  hotpotqa:      $EVAL_HP (after $GEN_HP)"
echo "  musique:       $EVAL_MU (after $GEN_MU)"
echo "  2wikimultihop: $EVAL_2W (after $GEN_2W)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Results: /projects/prjs1800/msc-thesis/01-arag-reproduction/results/qwen3-30b-e5-deepseekr1/"
