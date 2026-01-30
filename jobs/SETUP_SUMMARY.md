# Snellius Job Setup Summary

All job scripts for running multi-agentic RAG experiments on Snellius have been created and pushed to your repository.

## 📦 What's Been Created

### Directory Structure
```
jobs/
├── setup/
│   └── setup_conda_env.sh          # Conda environment setup
├── datasets/
│   ├── download_hotpotqa.sh        # HotpotQA dataset
│   ├── download_2wikimultihopqa.sh # 2WikiMultiHopQA dataset
│   ├── download_triviaqa.sh        # TriviaQA dataset
│   ├── download_naturalquestions.sh # Natural Questions dataset
│   └── download_all_datasets.sh    # Download all datasets
├── benchmarks/
│   ├── baseline_hotpotqa.sh        # Baseline RAG benchmark
│   └── multiagentic_hotpotqa.sh    # Multi-agent RAG benchmark
├── logs/                            # Job output logs (created automatically)
├── README.md                        # Comprehensive documentation
├── check_jobs.sh                    # Job status checker
├── quickstart.sh                    # Quick start script
├── config_template.yaml             # Configuration template
└── SETUP_SUMMARY.md                 # This file
```

## 🚀 Quick Start on Snellius

### Step 1: Clone Your Repository
```bash
cd ~
git clone https://github.com/pradyutnair/msc-thesis.git
cd msc-thesis
```

### Step 2: Run Quick Start
```bash
bash jobs/quickstart.sh
```

This will:
- Create necessary directories
- Submit the conda environment setup job
- Optionally submit dataset download jobs

### Step 3: Set API Keys
```bash
# Add to ~/.bashrc for persistence
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Monitor Progress
```bash
# Check job status
squeue -u $USER

# Check comprehensive status
bash jobs/check_jobs.sh

# View logs
tail -f jobs/output/setup_conda_env_*.log
```

## 📋 Job Scripts Details

### 1. Environment Setup

**Script**: `jobs/setup/setup_conda_env.sh`

**Usage**:
```bash
sbatch jobs/setup/setup_conda_env.sh
sbatch --dependency=afterok:<ENV_JOB_ID> jobs/setup/install_flashrag.sh
```

**What it does**:
- Creates conda environment: `multi_agentic_rag`
- Installs PyTorch with CUDA 12.1
- Installs LangChain, LangGraph, OpenAI, Anthropic
- Installs FAISS, ChromaDB, Sentence Transformers
- Installs evaluation libraries (NLTK, ROUGE, BERTScore)
- Downloads NLTK data
- Saves environment to `~/multi_agentic_rag_env.yml`

**Resources**:
- Partition: `cbuild`
- Time: 1 hour
- CPUs: 4
- Memory: 16GB

### 2. Dataset Downloads

#### HotpotQA
**Script**: `jobs/datasets/download_hotpotqa.sh`

**Usage**:
```bash
sbatch jobs/datasets/download_hotpotqa.sh
```

**Downloads**:
- Training set (113k examples)
- Dev set with distractors (7,405 examples)
- Dev set full wiki (7,405 examples)
- Test set (7,405 examples)
- Small split (100 examples)
- Tiny split (10 examples)

**Output**: `/projects/prjs1800/datasets/hotpotqa/`

#### 2WikiMultiHopQA
**Script**: `jobs/datasets/download_2wikimultihopqa.sh`

**Downloads from Hugging Face**:
- Train, dev, test splits
- Small and tiny splits

**Output**: `/projects/prjs1800/datasets/2wikimultihopqa/`

#### TriviaQA
**Script**: `jobs/datasets/download_triviaqa.sh`

**Downloads**:
- Unfiltered version (650k+ examples)
- Train, validation, test splits
- Small and tiny splits

**Output**: `/projects/prjs1800/datasets/triviaqa/`

#### Natural Questions
**Script**: `jobs/datasets/download_naturalquestions.sh`

**Downloads**:
- Train and validation splits (307k+ examples)
- Small and tiny splits

**Output**: `/projects/prjs1800/datasets/natural_questions/`

#### All Datasets
**Script**: `jobs/datasets/download_all_datasets.sh`

**Usage**:
```bash
sbatch jobs/datasets/download_all_datasets.sh
```

Downloads all four datasets in one job.

**Resources**:
- Partition: `cbuild`
- Time: 8 hours
- CPUs: 8
- Memory: 128GB

### 3. Baseline Benchmarks

#### Baseline RAG
**Script**: `jobs/benchmarks/baseline_hotpotqa.sh`

**Usage**:
```bash
sbatch jobs/benchmarks/baseline_hotpotqa.sh
```

**What it does**:
- Implements simple single-agent RAG
- Uses dense retrieval (sentence-transformers)
- FAISS indexing
- Evaluates on HotpotQA small split

**Resources**:
- Partition: `gpu-a100`
- Time: 12 hours
- GPUs: 1x A100
- CPUs: 8
- Memory: 64GB

**Output**: `/projects/prjs1800/results/baseline_hotpotqa/`

#### Multi-Agentic RAG
**Script**: `jobs/benchmarks/multiagentic_hotpotqa.sh`

**Usage**:
```bash
sbatch jobs/benchmarks/multiagentic_hotpotqa.sh
```

**What it does**:
- Uses the multi-agentic RAG framework
- Hierarchical orchestration
- Query decomposition
- Multi-source retrieval
- Answer synthesis with confidence

**Resources**:
- Partition: `gpu-a100`
- Time: 24 hours
- GPUs: 1x A100
- CPUs: 16
- Memory: 128GB

**Output**: `/projects/prjs1800/results/multiagentic_hotpotqa/`

## 🛠️ Utility Scripts

### Job Status Checker
**Script**: `jobs/check_jobs.sh`

**Usage**:
```bash
bash jobs/check_jobs.sh
```

**Shows**:
- Current running jobs
- Recent job history (24 hours)
- Disk usage
- Conda environment status
- Dataset download status
- Recent log files

### Quick Start
**Script**: `jobs/quickstart.sh`

**Usage**:
```bash
bash jobs/quickstart.sh
```

**Does**:
- Interactive setup
- Submits environment setup job
- Optionally submits dataset downloads
- Creates directory structure

## 📊 Expected Results Structure

After running experiments, your results will be organized as:

```
/projects/prjs1800/results/
├── baseline_hotpotqa/
│   ├── baseline_results_20250120_143022.json
│   └── ...
├── multiagentic_hotpotqa/
│   ├── multiagentic_results_20250120_150045.json
│   └── ...
└── ...
```

Each results file contains:
- Dataset information
- System configuration
- Evaluation metrics (exact match, F1, etc.)
- Individual predictions
- Metadata (confidence scores, sources, etc.)

## 🔧 Configuration

### Environment Variables Required
```bash
export OPENAI_API_KEY='your-openai-key'
export ANTHROPIC_API_KEY='your-anthropic-key'  # Optional
```

### Configuration Template
Use `jobs/config_template.yaml` as a starting point for your experiments.

## 📈 Monitoring Jobs

### Check Queue
```bash
squeue -u $USER
```

### View Job Details
```bash
scontrol show job <JOB_ID>
```

### Check Resource Usage
```bash
seff <JOB_ID>  # After job completes
```

### Cancel Job
```bash
scancel <JOB_ID>
```

### View Logs in Real-Time
```bash
tail -f jobs/output/<job_name>_<JOB_ID>.log
```

## 🎯 Recommended Workflow

### For Initial Setup
1. Run `bash jobs/quickstart.sh`
2. Wait for environment setup (check with `squeue -u $USER`)
3. Wait for dataset downloads
4. Verify with `bash jobs/check_jobs.sh`

### For Testing
1. Use tiny splits (10 examples) for quick tests
2. Use small splits (100 examples) for validation
3. Use full datasets for final evaluation

### For Experiments
1. Copy `config_template.yaml` to `configs/experiment_1.yaml`
2. Modify configuration
3. Update job script to use your config
4. Submit job: `sbatch jobs/benchmarks/your_experiment.sh`

## 🐛 Troubleshooting

### Job Fails Immediately
- Check partition availability: `sinfo`
- Verify resource limits
- Check logs in `jobs/output/`

### Out of Memory
- Increase `--mem` in job script
- Use smaller dataset splits
- Reduce batch size

### CUDA Errors
- Verify CUDA module loaded: `module list`
- Check GPU: `nvidia-smi`
- Ensure correct partition: `gpu-a100`

### Dataset Download Fails
- Check internet connectivity
- Verify Hugging Face access
- Try individual dataset scripts

### API Rate Limits
- Use smaller dataset splits
- Add delays between API calls
- Consider using local models

## 📚 Additional Resources

### Snellius Documentation
- Main docs: https://servicedesk.surf.nl/wiki/display/WIKI/Snellius
- SLURM guide: https://slurm.schedmd.com/documentation.html

### Project Documentation
- Framework README: `multi_agentic_rag/README.md`
- Implementation guide: `multi_agentic_rag/IMPLEMENTATION_GUIDE.md`
- Technical report: See deliverables from previous research

## 🎓 Next Steps

1. **Complete Setup**
   - Run quickstart script
   - Verify environment and datasets

2. **Run Baselines**
   - Start with tiny splits for testing
   - Run baseline RAG benchmark
   - Run multi-agentic RAG benchmark

3. **Analyze Results**
   - Compare baseline vs multi-agentic
   - Identify strengths and weaknesses
   - Plan improvements

4. **Develop Novel Approaches**
   - Implement ReFrag
   - Add collective reasoning
   - Enhance memory system
   - Integrate process-level supervision

5. **Scale Up**
   - Run on full datasets
   - Compare with SOTA
   - Prepare for publication

## ✅ Checklist

- [ ] Repository cloned on Snellius
- [ ] Conda environment created
- [ ] All datasets downloaded
- [ ] API keys configured
- [ ] Baseline benchmark completed
- [ ] Multi-agentic benchmark completed
- [ ] Results analyzed
- [ ] Novel approach implemented
- [ ] Full evaluation completed
- [ ] Paper draft started

## 📝 Notes

- All scripts use absolute paths (`$HOME`) for portability
- Logs are saved to `jobs/output/` with job ID
- Results include timestamps for tracking
- Small and tiny splits available for quick testing
- Job dependencies can be chained with `--dependency=afterok:<JOB_ID>`

## 🤝 Support

For issues:
- Snellius: servicedesk@surf.nl
- Project: Check main README and technical report
- Framework: See implementation guide

Good luck with your experiments! 🚀
