"""Run FlashRAG Standard/Reranker Pipeline.

Day 6: Runs standard RAG or reranker pipeline on 2WikiMultihopQA,
replicating the exact same setup as Day 1 (standard) and Day 2 (reranker).

This uses FlashRAG's built-in SequentialPipeline, which handles:
- Retrieval (E5 dense retriever)
- Optional reranking (if use_reranker=True in config)
- Generation (vLLM-based Qwen2.5-7B)
- Evaluation (EM, F1)

Usage:
    python -u scripts/day6/run_standard_pipeline.py --config configs/day6/standard_rag_qwen25_2wiki.yaml
    python -u scripts/day6/run_standard_pipeline.py --config configs/day6/reranker_qwen25_2wiki.yaml
"""

import argparse
import os
import time

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline


def main():
    parser = argparse.ArgumentParser(description="Run FlashRAG standard/reranker pipeline")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, top-k: {config['retrieval_topk']}")
    print(f"Use reranker: {config['use_reranker']}")
    if config['use_reranker']:
        print(f"  Model: {config['rerank_model_name']}, rerank top-k: {config['rerank_topk']}")
    print(f"Max tokens: {config['generation_params']['max_tokens']}")
    print(f"Save dir: {config['save_dir']}")

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    import faiss
    print(f"FAISS threads: {faiss.omp_get_max_threads()}")

    t_start = time.time()
    pipeline = SequentialPipeline(config)
    result = pipeline.run(test_data, do_eval=True)
    t_total = time.time() - t_start

    print(f"\nTotal time: {t_total:.1f}s ({t_total/len(test_data):.2f}s/example)")
    print(f"Results saved to {config['save_dir']}")


if __name__ == "__main__":
    main()
