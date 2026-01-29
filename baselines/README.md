# Baselines

This directory contains baseline implementations for evaluating multi-agentic RAG systems on multi-hop question answering tasks.

## Available Baselines

### Naive Vector Search

A simple single-step retrieval baseline using cosine similarity. This serves as the lower bound for performance comparison.

**Location:** `naive_vector_search/`

**Key Features:**
- Sentence-transformers for embeddings
- FAISS for efficient cosine similarity search
- No multi-hop reasoning or query decomposition
- Supports 4 datasets: 2WikiMultihopQA, HotpotQA, TriviaQA, Natural Questions

**Quick Start:**
```bash
# Run on HotpotQA
python -m baselines.naive_vector_search.run_baseline \
    --config baselines/naive_vector_search/configs/hotpotqa.yaml

# Submit SLURM job
sbatch jobs/baselines/naive_hotpotqa.sh
```

See [naive_vector_search/README.md](naive_vector_search/README.md) for detailed documentation.

## Adding New Baselines

To add a new baseline:

1. Create a new directory: `baselines/your_baseline_name/`
2. Implement the baseline with a consistent interface
3. Add configuration files in `configs/`
4. Create SLURM job scripts in `jobs/baselines/`
5. Document in a README.md

Recommended structure:
```
baselines/your_baseline_name/
├── README.md
├── requirements.txt
├── __init__.py
├── run_baseline.py
├── configs/
│   ├── dataset1.yaml
│   └── dataset2.yaml
└── [implementation files]
```

## Datasets

All baselines use datasets from:
```
/projects/prjs1800/datasets/
├── 2wikimultihopqa/
├── hotpotqa/
├── triviaqa/
└── natural_questions/
```

## Results

Results are saved to:
```
/projects/prjs1800/results/
├── naive_baseline/
│   ├── 2wikimultihopqa/
│   ├── hotpotqa/
│   ├── triviaqa/
│   └── natural_questions/
└── [other baselines]/
```

## Evaluation Metrics

All baselines should report:
- **Exact Match (EM)**: Normalized exact string match
- **F1 Score**: Token-level F1 between prediction and ground truth
- **Retrieval Metrics** (when applicable): Precision, Recall, F1 for retrieved documents

## Job Management

Submit all baseline jobs:
```bash
bash jobs/baselines/run_all_naive_baselines.sh
```

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
ls jobs/logs/
tail -f jobs/logs/naive_hotpot_*.log
```

## Comparison

Performance comparison across baselines will be documented here as more baselines are implemented.

| Baseline | 2WikiMultihopQA EM | HotpotQA EM | TriviaQA EM | NQ EM |
|----------|-------------------|-------------|-------------|-------|
| Naive Vector Search | TBD | TBD | TBD | TBD |
| Multi-Agentic RAG | TBD | TBD | TBD | TBD |

## References

- [DecEx-RAG](https://github.com/sdsxdxl/DecEx-RAG)
- [MA-RAG](https://github.com/thangylvp/MA-RAG)
- [CoRAG](https://github.com/microsoft/LMOps/tree/main/corag)
- [HM-RAG](https://github.com/ocean-luna/HMRAG)
