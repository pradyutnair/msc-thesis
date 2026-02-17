#!/bin/bash
set -euo pipefail
JOB_DIR="/projects/prjs1800/msc-thesis/01-arag-reproduction/jobs/generated_rejudge_dsr1"

echo "Submitting DeepSeek re-judge jobs for E2 and E3"
J1=$(sbatch "$JOB_DIR/eval_e2_hotpotqa.job" | awk '{print $4}')
J2=$(sbatch "$JOB_DIR/eval_e2_musique.job" | awk '{print $4}')
J3=$(sbatch "$JOB_DIR/eval_e2_2wikimultihop.job" | awk '{print $4}')
J4=$(sbatch "$JOB_DIR/eval_e3_hotpotqa.job" | awk '{print $4}')
J5=$(sbatch "$JOB_DIR/eval_e3_musique.job" | awk '{print $4}')
J6=$(sbatch "$JOB_DIR/eval_e3_2wikimultihop.job" | awk '{print $4}')

echo "E2 hotpotqa:      $J1"
echo "E2 musique:       $J2"
echo "E2 2wikimultihop: $J3"
echo "E3 hotpotqa:      $J4"
echo "E3 musique:       $J5"
echo "E3 2wikimultihop: $J6"
