# Snellius Job Scripts for Multi-Agentic RAG

This directory contains SBATCH job scripts for running experiments on the Snellius supercomputer (Dutch National Supercomputer).

## Directory Structure

```
jobs/
├── setup/              # Environment setup scripts
├── datasets/           # Dataset download and preparation scripts
├── benchmarks/         # Baseline and evaluation scripts
├── logs/              # Job output and error logs
└── README.md          # This file
```

## Prerequisites

1. **Snellius Account**: You need an active account on Snellius
2. **Project Allocation**: Ensure you have compute hours allocated
3. **API Keys**: Set up required API keys (e.g., OpenAI) in your environment

## Quick Start

### Step 1: Setup Conda Environment

First, create the conda environment with all dependencies:

```bash
cd ~/msc-thesis
sbatch jobs/setup/setup_conda_env.sh
```

This will create a conda environment named `multi_agentic_rag` with all required packages.

**Monitor the job:**
```bash
squeue -u $USER
tail -f jobs/logs/setup_conda_env_<JOB_ID>.out
```

### Step 2: Download Datasets

After the environment is set up, download all datasets:

```bash
# Download all datasets at once (recommended)
sbatch jobs/datasets/download_all_datasets.sh

# Or download individual datasets
sbatch jobs/datasets/download_hotpotqa.sh
sbatch jobs/datasets/download_2wikimultihopqa.sh
sbatch jobs/datasets/download_triviaqa.sh
sbatch jobs/datasets/download_naturalquestions.sh
```

Datasets will be downloaded to `/projects/prjs1800/datasets/` with the following structure:
- `/projects/prjs1800/datasets/hotpotqa/`
- `/projects/prjs1800/datasets/2wikimultihopqa/`
- `/projects/prjs1800/datasets/triviaqa/`
- `/projects/prjs1800/datasets/natural_questions/`

### Step 3: Run Baseline Benchmarks

Before running the benchmarks, set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Run baseline evaluations:

```bash
# Baseline single-agent RAG
sbatch jobs/benchmarks/baseline_hotpotqa.sh

# Multi-agentic RAG
sbatch jobs/benchmarks/multiagentic_hotpotqa.sh
```

Results will be saved to `/projects/prjs1800/results/` directory.

## Partition Information

Snellius has different partitions for different workloads:

### CPU Partitions
- **cbuild**: For building and setup tasks (used in this project)
  - Time limit: 1 hour
  - Suitable for: Environment setup, dataset downloads

### GPU Partitions
- **gpu-a100**: NVIDIA A100 GPUs (used for benchmarks)
  - Time limit: Varies by allocation
  - Suitable for: Model inference, training, evaluation

## Job Scripts Overview

### Setup Scripts

#### `setup/setup_conda_env.sh`
- **Purpose**: Create conda environment with all dependencies
- **Partition**: cbuild
- **Time**: 1 hour
- **Memory**: 16GB
- **Output**: `/projects/prjs1800/conda_envs/multi_agentic_rag/`

**Installed packages:**
- PyTorch with CUDA support
- LangChain and LangGraph
- OpenAI and Anthropic clients
- Vector stores (FAISS, ChromaDB)
- Sentence Transformers
- Graph databases (Neo4j, NetworkX)
- Evaluation libraries (NLTK, ROUGE, BERTScore)

### Dataset Scripts

#### `datasets/download_hotpotqa.sh`
- **Purpose**: Download HotpotQA dataset
- **Partition**: cbuild
- **Time**: 2 hours
- **Memory**: 32GB
- **Output**: `/projects/prjs1800/datasets/hotpotqa/`

**Downloaded files:**
- `hotpot_train_v1.1.json` (Training set)
- `hotpot_dev_distractor_v1.json` (Dev set with distractors)
- `hotpot_dev_fullwiki_v1.json` (Dev set with full Wikipedia)
- `hotpot_test_fullwiki_v1.json` (Test set)
- `hotpot_dev_small.json` (100 examples for quick testing)
- `hotpot_dev_tiny.json` (10 examples for debugging)

#### `datasets/download_2wikimultihopqa.sh`
- **Purpose**: Download 2WikiMultiHopQA dataset
- **Partition**: cbuild
- **Time**: 2 hours
- **Memory**: 32GB
- **Output**: `/projects/prjs1800/datasets/2wikimultihopqa/`

#### `datasets/download_triviaqa.sh`
- **Purpose**: Download TriviaQA dataset
- **Partition**: cbuild
- **Time**: 3 hours
- **Memory**: 64GB
- **Output**: `/projects/prjs1800/datasets/triviaqa/`

#### `datasets/download_naturalquestions.sh`
- **Purpose**: Download Natural Questions dataset
- **Partition**: cbuild
- **Time**: 3 hours
- **Memory**: 64GB
- **Output**: `/projects/prjs1800/datasets/natural_questions/`

#### `datasets/download_all_datasets.sh`
- **Purpose**: Download all datasets in one job
- **Partition**: cbuild
- **Time**: 8 hours
- **Memory**: 128GB
- **Output**: All datasets in `/projects/prjs1800/datasets/`

### Benchmark Scripts

#### `benchmarks/baseline_hotpotqa.sh`
- **Purpose**: Run baseline single-agent RAG on HotpotQA
- **Partition**: gpu-a100
- **Time**: 12 hours
- **GPUs**: 1x A100
- **Memory**: 64GB
- **Output**: `/projects/prjs1800/results/baseline_hotpotqa/`

**Features:**
- Dense retrieval with sentence-transformers
- FAISS indexing
- Simple answer generation

#### `benchmarks/multiagentic_hotpotqa.sh`
- **Purpose**: Run multi-agentic RAG framework on HotpotQA
- **Partition**: gpu-a100
- **Time**: 24 hours
- **GPUs**: 1x A100
- **Memory**: 128GB
- **Output**: `/projects/prjs1800/results/multiagentic_hotpotqa/`

**Features:**
- Hierarchical multi-agent orchestration
- Query decomposition
- Multi-source retrieval
- Answer synthesis with confidence scoring

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View job details
```bash
scontrol show job <JOB_ID>
```

### Cancel a job
```bash
scancel <JOB_ID>
```

### View job output in real-time
```bash
tail -f jobs/logs/<job_name>_<JOB_ID>.out
```

### View job errors
```bash
tail -f jobs/logs/<job_name>_<JOB_ID>.err
```

## Environment Variables

### Required
- `OPENAI_API_KEY`: Your OpenAI API key (for benchmarks using GPT models)

### Optional
- `ANTHROPIC_API_KEY`: For using Claude models
- `HUGGINGFACE_TOKEN`: For accessing gated models on Hugging Face

### Setting environment variables
Add to your `~/.bashrc` or set before submitting jobs:

```bash
export OPENAI_API_KEY='your-key-here'
export ANTHROPIC_API_KEY='your-key-here'
```

## Resource Allocation Guidelines

### CPU Jobs (cbuild partition)
- **Setup tasks**: 4 CPUs, 16GB RAM
- **Dataset downloads**: 4-8 CPUs, 32-128GB RAM

### GPU Jobs (gpu-a100 partition)
- **Baseline benchmarks**: 1 GPU, 8 CPUs, 64GB RAM
- **Multi-agent benchmarks**: 1 GPU, 16 CPUs, 128GB RAM
- **Large-scale experiments**: 2-4 GPUs, 32 CPUs, 256GB RAM

## Troubleshooting

### Job fails immediately
- Check partition availability: `sinfo`
- Verify resource requests are within limits
- Check log files for errors

### Out of memory errors
- Increase `--mem` parameter
- Use smaller dataset splits for testing
- Reduce batch size in evaluation scripts

### CUDA errors
- Verify CUDA module is loaded: `module list`
- Check GPU availability: `nvidia-smi`
- Ensure `CUDA_VISIBLE_DEVICES` is set correctly

### Dataset download fails
- Check internet connectivity from compute nodes
- Verify Hugging Face is accessible
- Try downloading to a different directory

### API rate limits
- Reduce number of API calls
- Add delays between requests
- Use smaller dataset splits for testing

## Best Practices

1. **Test with small datasets first**: Use `_tiny.json` or `_small.json` splits
2. **Monitor resource usage**: Use `seff <JOB_ID>` after job completion
3. **Save intermediate results**: Checkpoint your experiments
4. **Use job arrays**: For running multiple configurations
5. **Clean up logs**: Regularly remove old log files

## Job Arrays

To run multiple experiments with different configurations:

```bash
#SBATCH --array=1-10

# Use $SLURM_ARRAY_TASK_ID to vary parameters
CONFIG_FILE="configs/config_${SLURM_ARRAY_TASK_ID}.yaml"
```

## Extending the Scripts

### Adding a new dataset
1. Create a new script in `jobs/datasets/`
2. Follow the template of existing scripts
3. Use `cbuild` partition for downloads
4. Save to `/projects/prjs1800/datasets/<dataset_name>/`

### Adding a new benchmark
1. Create a new script in `jobs/benchmarks/`
2. Use `gpu-a100` partition for GPU jobs
3. Load the conda environment
4. Save results to `/projects/prjs1800/results/<benchmark_name>/`

## Support

For Snellius-specific issues:
- Documentation: https://servicedesk.surf.nl/wiki/display/WIKI/Snellius
- Support: servicedesk@surf.nl

For project-specific issues:
- Check the main README in the project root
- Review the technical report in `/multi_agentic_rag_research/`

## Citation

If you use these scripts in your research, please cite:

```
@misc{multi_agentic_rag_2025,
  title={Multi-Agentic RAG Framework for Complex Question Answering},
  author={Your Name},
  year={2025},
  institution={Your University}
}
```
