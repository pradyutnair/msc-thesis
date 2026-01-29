#!/bin/bash

# Submit all naive vector search baseline jobs
# This script submits SLURM jobs for all four datasets

echo "=========================================="
echo "Submitting Naive Vector Search Baselines"
echo "=========================================="

# Navigate to project root
cd "$(dirname "$0")/../.."

# Create logs directory if it doesn't exist
mkdir -p jobs/logs

# Submit jobs
echo "Submitting 2WikiMultihopQA baseline..."
JOB1=$(sbatch jobs/baselines/naive_2wikimultihopqa.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting HotpotQA baseline..."
JOB2=$(sbatch jobs/baselines/naive_hotpotqa.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting TriviaQA baseline..."
JOB3=$(sbatch jobs/baselines/naive_triviaqa.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

echo "Submitting Natural Questions baseline..."
JOB4=$(sbatch jobs/baselines/naive_natural_questions.sh | awk '{print $4}')
echo "  Job ID: $JOB4"

echo "=========================================="
echo "All jobs submitted!"
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Check status with: squeue -u \$USER"
echo "Check logs in: jobs/logs/"
echo "=========================================="
