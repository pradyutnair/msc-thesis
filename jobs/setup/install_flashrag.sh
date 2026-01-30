#!/bin/bash
#SBATCH --job-name=install_flashrag
#SBATCH --partition=cbuild
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=jobs/logs/install_flashrag_%j.log
#SBATCH --error=jobs/logs/install_flashrag_%j.err

# Install FlashRAG into the shared conda environment

echo "=========================================="
echo "Installing FlashRAG (flashrag-dev --pre)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load 2023
module load Miniconda3/23.5.2-0

ENV_PATH="/projects/prjs1800/conda_envs/multi_agentic_rag"

if [ ! -d "$ENV_PATH" ]; then
    echo "ERROR: Conda environment not found at $ENV_PATH"
    echo "Run jobs/setup/setup_conda_env.sh first."
    exit 1
fi

source activate "$ENV_PATH"
pip install --upgrade pip
pip install "flashrag-dev" --pre

echo "FlashRAG installation complete."
python -c "import flashrag; print('FlashRAG version:', getattr(flashrag, '__version__', 'unknown'))"

echo "End time: $(date)"
echo "=========================================="
