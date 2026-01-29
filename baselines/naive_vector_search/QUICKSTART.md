# Quick Start Guide

Get the naive vector search baseline running in 5 minutes.

## Prerequisites

1. Access to the remote server with datasets at `/projects/prjs1800/datasets/`
2. Conda environment activated: `multi_agentic_rag`
3. GPU access (optional but recommended)

## Step 1: Install Dependencies

```bash
cd ~/msc-thesis
pip install -r baselines/naive_vector_search/requirements.txt
```

For GPU support:
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Step 2: Test on Small Dataset

Create a test config for quick validation:

```bash
cat > baselines/naive_vector_search/configs/test.yaml << 'EOF'
dataset:
  name: "hotpotqa"
  dir: "/projects/prjs1800/datasets/hotpotqa"
  split: "dev"
  limit: 100  # Only 100 questions for testing

retriever:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"
  top_k: 5
  batch_size: 32

output:
  dir: "./test_results"
  save_index: false
EOF
```

Run the test:
```bash
python -m baselines.naive_vector_search.run_baseline \
    --config baselines/naive_vector_search/configs/test.yaml
```

Expected output:
```
==========================================
Starting Naive Vector Search Baseline
==========================================
Dataset: hotpotqa
Split: dev
Model: sentence-transformers/all-MiniLM-L6-v2
...
Evaluation Results:
  Exact Match: 0.XXXX
  F1 Score: 0.XXXX
  Num Examples: 100
==========================================
```

## Step 3: Submit Full Job

Once the test works, submit the full evaluation:

```bash
sbatch jobs/baselines/naive_hotpotqa.sh
```

Check status:
```bash
squeue -u $USER
```

Monitor progress:
```bash
tail -f jobs/logs/naive_hotpot_*.log
```

## Step 4: Run All Datasets

Submit all baseline jobs at once:

```bash
bash jobs/baselines/run_all_naive_baselines.sh
```

This will submit 4 jobs (one per dataset).

## Step 5: View Results

Results are saved to:
```
/projects/prjs1800/results/naive_baseline/
├── 2wikimultihopqa/
│   └── 2wikimultihopqa_dev_results.json
├── hotpotqa/
│   └── hotpotqa_dev_results.json
├── triviaqa/
│   └── triviaqa_dev_results.json
└── natural_questions/
    └── natural_questions_dev_results.json
```

View metrics:
```bash
python -c "
import json
with open('/projects/prjs1800/results/naive_baseline/hotpotqa/hotpotqa_dev_results.json') as f:
    data = json.load(f)
    print('Metrics:', data['metrics'])
"
```

## Troubleshooting

**Error: Dataset not found**
- Check dataset path in config file
- Verify datasets are downloaded: `ls /projects/prjs1800/datasets/`

**Error: Out of memory**
- Reduce `batch_size` in config (try 16 or 8)
- Use CPU: set `device: "cpu"` in config

**Error: CUDA not available**
- Check GPU allocation: `nvidia-smi`
- Fall back to CPU or request GPU node

**Job pending for long time**
- Check queue: `squeue`
- Try different partition: `--partition=gpu` instead of `gpu_a100`

## Next Steps

1. Analyze results in the JSON output files
2. Compare performance across datasets
3. Experiment with different embedding models
4. Implement improved baselines or multi-agentic approaches

## Common Modifications

**Use different embedding model:**
```yaml
retriever:
  model_name: "sentence-transformers/all-mpnet-base-v2"  # Larger model
```

**Retrieve more documents:**
```yaml
retriever:
  top_k: 20  # Retrieve top-20 instead of top-10
```

**Process only a subset:**
```yaml
dataset:
  limit: 1000  # Process first 1000 questions
```

**Run on training set:**
```yaml
dataset:
  split: "train"
```

## Support

For issues or questions:
1. Check the main [README.md](README.md) for detailed documentation
2. Review logs in `jobs/logs/`
3. Check dataset format and structure
4. Verify conda environment and dependencies
