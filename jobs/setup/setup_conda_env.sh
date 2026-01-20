#!/bin/bash
#SBATCH --job-name=setup_conda_env
#SBATCH --partition=cbuild
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=jobs/logs/setup_conda_env_%j.out
#SBATCH --error=jobs/logs/setup_conda_env_%j.err

# Snellius Conda Environment Setup for Multi-Agentic RAG
# This script creates a conda environment with all necessary dependencies

echo "=========================================="
echo "Starting Conda Environment Setup"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load conda module
module purge
module load 2023
module load Miniconda3/23.5.2-0

# Project-scoped paths
PROJECT_ROOT="/projects/prjs1800"
ENV_NAME="multi_agentic_rag"
ENV_PATH="$PROJECT_ROOT/conda_envs/$ENV_NAME"
mkdir -p "$PROJECT_ROOT/conda_envs"

# Remove existing environment if it exists
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing environment: $ENV_PATH"
    conda env remove -p "$ENV_PATH" -y || rm -rf "$ENV_PATH"
fi

# Create new conda environment
echo "Creating new conda environment: $ENV_PATH"
conda create -p "$ENV_PATH" python=3.11 -y

# Activate environment
source activate "$ENV_PATH"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core LLM and AI frameworks
echo "Installing LLM and AI frameworks..."
pip install openai==1.12.0
pip install anthropic==0.18.1
pip install transformers==4.38.1
pip install langchain==0.1.9
pip install langgraph==0.0.26
pip install langchain-community==0.0.24
pip install langchain-openai==0.0.6

# Install vector stores and embeddings
echo "Installing vector stores and embeddings..."
pip install faiss-cpu==1.7.4
pip install chromadb==0.4.22
pip install sentence-transformers==2.3.1
pip install tiktoken==0.6.0

# Install graph databases and knowledge graph tools
echo "Installing graph databases..."
pip install neo4j==5.17.0
pip install networkx==3.2.1
pip install pyvis==0.3.2

# Install data processing libraries
echo "Installing data processing libraries..."
pip install numpy==1.26.4
pip install pandas==2.2.0
pip install datasets==2.17.1
pip install huggingface-hub==0.20.3

# Install web scraping and search
echo "Installing web scraping and search tools..."
pip install requests==2.31.0
pip install beautifulsoup4==4.12.3
pip install duckduckgo-search==4.4.1

# Install utilities
echo "Installing utilities..."
pip install python-dotenv==1.0.1
pip install pydantic==2.6.1
pip install tqdm==4.66.1
pip install pyyaml==6.0.1

# Install evaluation and metrics
echo "Installing evaluation libraries..."
pip install scikit-learn==1.4.0
pip install scipy==1.12.0
pip install nltk==3.8.1
pip install rouge-score==0.1.2
pip install bert-score==0.3.13

# Install testing and development tools
echo "Installing testing and development tools..."
pip install pytest==8.0.0
pip install pytest-asyncio==0.23.5
pip install black==24.2.0
pip install flake8==7.0.0
pip install ipython==8.21.0
pip install jupyter==1.0.0

# Install specific RAG-related libraries
echo "Installing RAG-related libraries..."
pip install llama-index==0.10.12
pip install ragas==0.1.5

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import langchain; print(f'LangChain version: {langchain.__version__}')"
python -c "import faiss; print(f'FAISS installed successfully')"
python -c "import sentence_transformers; print(f'Sentence Transformers installed successfully')"

# Save environment information
echo "=========================================="
echo "Saving environment information..."
echo "=========================================="
conda env export > "$PROJECT_ROOT/conda_envs/multi_agentic_rag_env.yml"
pip list > "$PROJECT_ROOT/conda_envs/multi_agentic_rag_pip_list.txt"

echo "Environment setup complete!"
echo "Environment name: $ENV_NAME"
echo "Environment path: $ENV_PATH"
echo "Environment YAML: $PROJECT_ROOT/conda_envs/multi_agentic_rag_env.yml"
echo "Pip packages list: $PROJECT_ROOT/conda_envs/multi_agentic_rag_pip_list.txt"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
