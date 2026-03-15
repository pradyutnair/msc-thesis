# MSc Thesis: Incremental RAG Component Analysis

## Directory Structure

```
msc-thesis/
├── configs/              # Configuration files organized by experiment day
│   ├── day1/            # Standard RAG configs
│   ├── day2/            # Reranker configs
│   ├── day3/            # Refiner configs (RECOMP, SC)
│   ├── day4/            # Iterative retrieval configs (IRCoT, FLARE)
│   ├── day5/            # Reasoning configs (CoT, ReasoningPipeline, SelfAsk)
│   └── benchmarks/      # Benchmark configs
│
├── scripts/             # Experiment execution scripts organized by day
│   ├── day1/           # Standard RAG scripts
│   ├── day2/           # Reranker scripts
│   ├── day3/           # Refiner scripts
│   ├── day4/           # Iterative retrieval scripts
│   ├── day5/           # Reasoning scripts
│   └── utils/          # Utility scripts (index building, evaluation, logging)
│
├── jobs/                # SLURM job scripts organized by experiment day
│   ├── day1/            # Standard RAG baseline
│   ├── day2/            # Reranker experiments
│   ├── day3/            # Refiner experiments (RECOMP, Selective-Context)
│   ├── day4/            # Iterative retrieval (IRCoT, FLARE)
│   ├── day5/            # Reasoning approaches
│   ├── benchmarks/      # Baseline benchmarks
│   ├── marag/           # Multi-agent RAG experiments
│   └── setup/           # Environment and dataset setup
│
├── results/             # All experiment results consolidated
│   ├── day1/           # Standard RAG results
│   │   ├── hotpotqa/
│   │   ├── musique/
│   │   └── logs/
│   ├── day2/           # Reranker results
│   ├── day3/           # Refiner results
│   ├── day4/           # Iterative retrieval results
│   ├── day5/           # Reasoning results
│   ├── marag/          # Multi-agent RAG results
│   └── benchmarks/     # Baseline benchmark results
│
├── analysis/           # Analysis scripts and outputs
│   ├── scripts/        # Analysis Python scripts
│   │   ├── analyze_retrieval.py
│   │   └── analyze_ircot_retrieval.py
│   └── outputs/        # Analysis results and summaries
│       ├── day4_analysis_results.json
│       └── day4_summary.md
│
├── docs/               # Documentation and reports
│   ├── claude-technical-report.md
│   └── manus-technical-report.md
│
├── plans/              # Planning documents and findings
│   ├── feb-5-standard-rag-incremental.md
│   └── feb5-key-findings.md
│
└── experiments/        # Reserved for future experiment organization
    ├── day1-standard-rag/
    ├── day2-reranker/
    ├── day3-refiners/
    ├── day4-iterative/
    └── day5-reasoning/
```

## Quick Navigation

- **Experiment Scripts**: `scripts/` - Python scripts to run experiments
- **Job Scripts**: `jobs/day*/` - SLURM job scripts for Snellius
- **Results**: `results/day*/` - All experiment outputs, metrics, and logs
- **Analysis**: `analysis/` - Scripts and outputs for analyzing results
- **Configs**: `configs/day*/` - YAML configuration files organized by experiment day
- **Scripts**: `scripts/day*/` - Python execution scripts organized by experiment day
- **Documentation**: `docs/` and `plans/` - Reports and planning docs

## Experiment Days Overview

- **Day 1**: Standard RAG baseline (E5-base-v2 retriever + Qwen2.5-7B)
- **Day 2**: + BGE Reranker (retrieve top-20, rerank to top-5)
- **Day 3**: + Refiners (RECOMP abstractive, Selective-Context perplexity filter)
- **Day 4**: Iterative retrieval (IRCoT, FLARE)
- **Day 5**: Reasoning approaches (CoT, ReasoningPipeline, SelfAsk)

## Results Structure

Each day's results folder contains:

- `hotpotqa/` - HotpotQA dataset results (metrics, configs)
- `musique/` - MuSiQue dataset results (metrics, configs)
- `logs/` - SLURM job logs (.log, .err, .out files)

## Running Experiments

See `jobs/README.md` for detailed instructions on running experiments on Snellius.

## Related Works:

- DualRAG: [https://arxiv.org/pdf/2504.18243, https://github.com/cbxgss/rag](https://arxiv.org/pdf/2504.18243)
- PRISMA: [https://arxiv.org/pdf/2601.05465](https://arxiv.org/pdf/2601.05465)
- OPERA: [https://arxiv.org/pdf/2508.16438](https://arxiv.org/pdf/2508.16438), [https://github.com/Ameame1/OPERA](https://github.com/Ameame1/OPERA)
- PRISM: [https://arxiv.org/pdf/2510.14278](https://arxiv.org/pdf/2510.14278)
- Adaptive-RAG: [https://arxiv.org/pdf/2403.14403](https://arxiv.org/pdf/2403.14403)
- Frugal-RAG: [https://arxiv.org/pdf/2507.07634](https://arxiv.org/pdf/2507.07634)
- ReAgent: [https://aclanthology.org/2025.emnlp-main.202.pdf](https://aclanthology.org/2025.emnlp-main.202.pdf)
- ARAG: [https://arxiv.org/pdf/2602.03442](https://arxiv.org/pdf/2602.03442)

