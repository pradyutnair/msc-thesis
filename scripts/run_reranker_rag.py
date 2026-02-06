"""Run Standard RAG + Reranker with FlashRAG — split retrieval/generation phases.

Phase 1: Retrieval + Reranking with full CPU parallelism (before vLLM).
Phase 2: Generation with vLLM + evaluation.
"""

import argparse
import os
import time

# CRITICAL: Set threading BEFORE any library import
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.retriever import DenseRetriever

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, initial top-k: {config['retrieval_topk']}")
    print(f"Reranker: {config['rerank_model_name']}, rerank top-k: {config['rerank_topk']}")
    print(f"Save dir: {config['save_dir']}")

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    # Phase 1: Retrieval + Reranking (uses FAISS with full CPU parallelism)
    import faiss
    print(f"\n=== Phase 1: Retrieval + Reranking (FAISS threads: {faiss.omp_get_max_threads()}) ===")

    retriever = DenseRetriever(config)
    questions = [item.question for item in test_data]

    t0 = time.time()
    retrieval_results = retriever.batch_search(questions, num=config["retrieval_topk"])
    t_retrieval = time.time() - t0
    print(f"Retrieval + Reranking: {t_retrieval:.1f}s ({t_retrieval/len(questions):.3f}s/query)")

    # Store results into dataset items
    for i, item in enumerate(test_data):
        item.update_output("retrieval_result", retrieval_results[i])

    # Free retriever memory
    del retriever
    import gc; gc.collect()
    print("Retriever freed from memory")

    # Phase 2: Generation
    print(f"\n=== Phase 2: Generation + Evaluation ===")
    from flashrag.pipeline import SequentialPipeline
    pipeline = SequentialPipeline(config)

    start_time = time.time()
    output = pipeline.run(test_data, do_eval=True)
    t_gen = time.time() - start_time

    t_total = t_retrieval + t_gen
    print(f"\nRetrieval+Rerank: {t_retrieval:.1f}s | Generation+Eval: {t_gen:.1f}s | Total: {t_total:.1f}s")
    print(f"Per example: {t_total/len(test_data):.2f}s")
    print(f"Results saved to {config['save_dir']}")

if __name__ == "__main__":
    main()
