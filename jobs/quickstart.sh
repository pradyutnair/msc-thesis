#!/bin/bash
# Quick start script for setting up the multi-agentic RAG environment on Snellius

echo "=========================================="
echo "Multi-Agentic RAG Quick Start"
echo "=========================================="
echo ""

# Check if on Snellius
if ! command -v sbatch &> /dev/null; then
    echo "ERROR: This script must be run on Snellius"
    exit 1
fi

echo "This script will:"
echo "1. Submit a job to create the conda environment"
echo "2. Submit jobs to download all datasets"
echo "3. Set up the directory structure"
echo ""

read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p /projects/prjs1800/datasets
mkdir -p /projects/prjs1800/results
mkdir -p /projects/prjs1800/conda_envs
mkdir -p jobs/output

# Check for API keys
echo ""
echo "Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY is not set"
    echo "You will need to set it before running benchmarks:"
    echo "  export OPENAI_API_KEY='your-api-key'"
    echo ""
fi

# Submit environment setup job
echo "Submitting conda environment setup job..."
ENV_JOB_ID=$(sbatch --parsable jobs/setup/setup_conda_env.sh)
echo "Job ID: $ENV_JOB_ID"
echo "Monitor with: tail -f jobs/output/setup_conda_env_${ENV_JOB_ID}.log"
echo ""

echo "Submitting FlashRAG installation job (after env ready)..."
FLASHRAG_JOB_ID=$(sbatch --parsable --dependency=afterok:$ENV_JOB_ID jobs/setup/install_flashrag.sh)
echo "Job ID: $FLASHRAG_JOB_ID"
echo "Monitor with: tail -f jobs/output/install_flashrag_${FLASHRAG_JOB_ID}.log"
echo ""

# Ask if user wants to download datasets
read -p "Do you want to download all datasets now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Submitting dataset download job (will wait for environment setup)..."
    DATASET_JOB_ID=$(sbatch --parsable --dependency=afterok:$ENV_JOB_ID jobs/datasets/download_all_datasets.sh)
    echo "Job ID: $DATASET_JOB_ID"
    echo "This job will start after the environment setup completes."
    echo "Monitor with: tail -f jobs/output/download_all_datasets_${DATASET_JOB_ID}.log"
    echo ""
fi

echo "=========================================="
echo "Setup initiated!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Wait for jobs to complete (check with: squeue -u \$USER)"
echo "2. Set your OpenAI API key: export OPENAI_API_KEY='your-key'"
echo "3. Run baseline benchmark: sbatch jobs/benchmarks/baseline_hotpotqa.sh"
echo "4. Run multi-agent benchmark: sbatch jobs/benchmarks/multiagentic_hotpotqa.sh"
echo ""
echo "Useful commands:"
echo "  squeue -u \$USER              # Check job status"
echo "  bash jobs/check_jobs.sh       # Comprehensive status check"
echo "  scancel <JOB_ID>              # Cancel a job"
echo ""
echo "For more information, see jobs/README.md"
echo ""
