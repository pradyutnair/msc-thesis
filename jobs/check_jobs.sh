#!/bin/bash
# Utility script to check status of all jobs

echo "=========================================="
echo "Snellius Job Status Checker"
echo "=========================================="
echo ""

# Check if running on Snellius
if ! command -v squeue &> /dev/null; then
    echo "ERROR: This script must be run on Snellius"
    exit 1
fi

echo "Current Jobs:"
echo "-------------"
squeue -u $USER --format="%.18i %.9P %.30j %.8T %.10M %.6D %R"
echo ""

echo "Recent Job History (last 24 hours):"
echo "------------------------------------"
sacct --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,Partition,State,Elapsed,MaxRSS,ExitCode --user=$USER
echo ""

echo "Disk Usage:"
echo "-----------"
echo "Home directory: $(du -sh $HOME 2>/dev/null | cut -f1)"
echo "Datasets: $(du -sh /projects/prjs1800/datasets 2>/dev/null | cut -f1)"
echo "Results: $(du -sh /projects/prjs1800/results 2>/dev/null | cut -f1)"
echo ""

echo "Conda Environments:"
echo "-------------------"
if [ -d "/projects/prjs1800/conda_envs/multi_agentic_rag" ]; then
    echo "✓ multi_agentic_rag environment exists"
else
    echo "✗ multi_agentic_rag environment not found"
fi
echo ""

echo "Datasets Status:"
echo "----------------"
for dataset in hotpotqa 2wikimultihopqa triviaqa natural_questions; do
    if [ -d "/projects/prjs1800/datasets/$dataset" ]; then
        num_files=$(find "/projects/prjs1800/datasets/$dataset" -name "*.json" | wc -l)
        echo "✓ $dataset: $num_files JSON files"
    else
        echo "✗ $dataset: not downloaded"
    fi
done
echo ""

echo "Recent Log Files:"
echo "-----------------"
if [ -d "jobs/logs" ]; then
    echo "Latest 5 log files:"
    ls -lht jobs/logs/*.out 2>/dev/null | head -5
else
    echo "No log files found"
fi
echo ""

echo "=========================================="
