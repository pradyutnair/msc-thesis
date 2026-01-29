# Naive Vector Search Baseline

A simple baseline implementation for multi-hop question answering using cosine similarity-based retrieval. This baseline serves as a reference point for evaluating more sophisticated multi-agentic RAG systems.

## Overview

This baseline implements a straightforward retrieval approach without any multi-hop reasoning capabilities. It uses sentence-transformers to encode documents and queries, then performs top-k retrieval using FAISS for efficient cosine similarity search.

**Key Characteristics:**
- Single-step retrieval (no query decomposition)
- Cosine similarity matching
- No multi-hop reasoning
- No answer generation (uses simple heuristics)
- Efficient batched processing

## Architecture

The baseline consists of three main components:

### 1. Dataset Loader (`dataset_loader.py`)

Provides unified interface for loading multi-hop QA datasets:
- **2WikiMultihopQA**: Multi-hop questions requiring reasoning over Wikipedia articles
- **HotpotQA**: Multi-hop questions with supporting facts annotations
- **TriviaQA**: Single-hop trivia questions
- **Natural Questions**: Questions from Google search queries

Each dataset is normalized into a common format with `Question` and `Document` objects.

### 2. Vector Retriever (`retriever.py`)

Implements naive vector search using:
- **Encoder**: sentence-transformers (default: `all-MiniLM-L6-v2`)
- **Index**: FAISS IndexFlatIP with L2-normalized vectors for cosine similarity
- **Retrieval**: Top-k documents based on query-document similarity

Features:
- Batched encoding for efficiency
- Index persistence (save/load)
- GPU acceleration support

### 3. Evaluator (`evaluator.py`)

Computes standard QA metrics:
- **Exact Match (EM)**: Binary match after normalization
- **F1 Score**: Token-level overlap between prediction and ground truth
- **Retrieval Metrics**: Precision, recall, and F1 for retrieved documents (when supporting facts are available)

## Installation

### Prerequisites

Ensure you have access to the datasets on the remote server:
```bash
/projects/prjs1800/datasets/
├── 2wikimultihopqa/
├── hotpotqa/
├── triviaqa/
└── natural_questions/
```

### Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

For GPU support with FAISS:
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Usage

### Local Execution

Run the baseline on a specific dataset:

```bash
python -m baselines.naive_vector_search.run_baseline \
    --config baselines/naive_vector_search/configs/hotpotqa.yaml
```

### Remote Server Execution

Submit SLURM jobs on the remote server:

```bash
# Single dataset
sbatch jobs/baselines/naive_hotpotqa.sh

# All datasets
bash jobs/baselines/run_all_naive_baselines.sh
```

Check job status:
```bash
squeue -u $USER
```

View logs:
```bash
tail -f jobs/logs/naive_hotpot_<job_id>.log
```

## Configuration

Each dataset has a YAML configuration file in `configs/`:

```yaml
dataset:
  name: "hotpotqa"
  dir: "/projects/prjs1800/datasets/hotpotqa"
  split: "dev"
  limit: null  # Set to limit dataset size for testing

retriever:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cuda"
  top_k: 10
  batch_size: 64

output:
  dir: "/projects/prjs1800/results/naive_baseline/hotpotqa"
  save_index: true
  index_dir: "/projects/prjs1800/results/naive_baseline/hotpotqa/index"
```

### Configuration Parameters

**Dataset:**
- `name`: Dataset identifier (2wikimultihopqa, hotpotqa, triviaqa, natural_questions)
- `dir`: Path to dataset directory
- `split`: Dataset split (train/dev/test)
- `limit`: Maximum number of questions (null for all)

**Retriever:**
- `model_name`: Sentence-transformer model from HuggingFace
- `device`: Computation device (cuda/cpu)
- `top_k`: Number of documents to retrieve
- `batch_size`: Batch size for encoding

**Output:**
- `dir`: Results output directory
- `save_index`: Whether to save the FAISS index
- `index_dir`: Directory for saving/loading index

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "config": { ... },
  "metrics": {
    "exact_match": 0.42,
    "f1": 0.58,
    "retrieval_precision": 0.65,
    "retrieval_recall": 0.72,
    "retrieval_f1": 0.68,
    "num_examples": 7405
  },
  "stats": {
    "num_questions": 7405,
    "num_documents": 15234,
    "retrieval_time": 45.2,
    "avg_time_per_query": 0.0061
  },
  "results": [
    {
      "question_id": "5a7a06935542990198eaf050",
      "question": "Which film was released first, Kistimaat or I Karunaikuzhiyil?",
      "predicted_answer": "I Karunaikuzhiyil",
      "ground_truth": "I Karunaikuzhiyil",
      "retrieved_docs": [
        {
          "title": "I Karunaikuzhiyil",
          "score": 0.85,
          "text": "I Karunaikuzhiyil is a 1977 Indian Malayalam film..."
        }
      ]
    }
  ]
}
```

## Limitations

This is a **naive baseline** with several known limitations:

1. **No Multi-Hop Reasoning**: Retrieves documents in a single step without considering multi-hop relationships
2. **Simple Answer Extraction**: Uses basic heuristics instead of proper answer generation
3. **No Query Decomposition**: Treats all questions uniformly regardless of complexity
4. **No Re-ranking**: Uses initial retrieval scores without refinement
5. **Limited Context**: Does not aggregate information across multiple documents

These limitations are intentional, as this baseline serves as a lower bound for comparison with more sophisticated approaches.

## Expected Performance

Approximate performance ranges on dev sets (will vary based on exact dataset versions):

| Dataset | EM | F1 | Retrieval F1 |
|---------|----|----|--------------|
| 2WikiMultihopQA | 0.15-0.25 | 0.25-0.35 | 0.45-0.55 |
| HotpotQA | 0.20-0.30 | 0.30-0.40 | 0.50-0.60 |
| TriviaQA | 0.35-0.45 | 0.45-0.55 | N/A |
| Natural Questions | 0.25-0.35 | 0.35-0.45 | N/A |

Note: These are rough estimates. Actual performance depends on dataset preprocessing and exact splits used.

## Extending the Baseline

To improve upon this baseline, consider:

1. **Better Embeddings**: Use larger models (e.g., `e5-large`, `instructor-xl`)
2. **Query Decomposition**: Break complex questions into sub-questions
3. **Multi-Hop Retrieval**: Iterative retrieval following reasoning chains
4. **Answer Generation**: Use LLMs to generate answers from retrieved context
5. **Re-ranking**: Add a cross-encoder for more accurate document scoring
6. **Hybrid Search**: Combine dense and sparse retrieval (BM25 + dense)

## Troubleshooting

**Out of Memory:**
- Reduce `batch_size` in config
- Use CPU instead of GPU for encoding
- Process dataset in chunks with `limit` parameter

**FAISS Index Issues:**
- Ensure FAISS-GPU is installed correctly
- Check CUDA compatibility
- Fall back to CPU version if needed

**Dataset Not Found:**
- Verify dataset paths in config files
- Check dataset directory structure
- Ensure datasets are downloaded on the server

## References

This baseline follows standard practices from:
- [DecEx-RAG](https://github.com/sdsxdxl/DecEx-RAG)
- [MA-RAG](https://github.com/thangylvp/MA-RAG)
- [HotpotQA Paper](https://arxiv.org/abs/1809.09600)
- [2WikiMultihopQA Paper](https://arxiv.org/abs/1910.09753)

## License

Part of the MSc Thesis project. See repository root for license information.
