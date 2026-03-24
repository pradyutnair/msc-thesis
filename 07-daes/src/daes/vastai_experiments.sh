#!/bin/bash
set -e
source .venv/bin/activate
export PYTHONPATH=$(pwd)/dllm:$PYTHONPATH
SCRIPTS=msc-thesis/07-daes/src/daes
RESULTS=results
mkdir -p $RESULTS

echo "=== Experiment 1: AR candidate comparison (CRITICAL) ==="
echo "  dLLM candidates vs Qwen3-8B candidates vs baseline. 50q MuSiQue."
python -u $SCRIPTS/ablation_ar_candidates.py \
    --data_dir data --n_questions 50 \
    2>&1 | tee $RESULTS/ablation_ar_candidates.log

echo "=== Experiment 2: maskgit-plus baseline ==="
echo "  Dream-7B with alg=maskgit_plus. 50q MuSiQue."
python -u $SCRIPTS/spread_reproduce.py \
    --dataset musique --n_questions 50 --mode baseline \
    --L 512 --T 128 --index_name index_e5_musique_full \
    2>&1 | tee $RESULTS/maskgit_plus_baseline.log

echo "=== Experiment 3: Candidate count ablation ==="
echo "  n=1,2,3,5 candidates. 50q MuSiQue."
for n in 1 2 3 5; do
    echo "--- n_candidates=$n ---"
    python -u $SCRIPTS/ablation_candidate_count.py \
        --data_dir data --n_questions 50 --n_candidates $n \
        2>&1 | tee $RESULTS/ablation_ncandidates_${n}.log
done

echo "=== All experiments complete ==="
