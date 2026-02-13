"""Run ReasoningPipeline (Search-R1 style) with FlashRAG.

Day 5: Advanced single-agent reasoning with interleaved search.
The model reasons step by step, generating search queries when uncertain,
then continues reasoning with retrieved documents.

Uses FlashRAG's built-in ReasoningPipeline which handles:
  - Step-by-step reasoning with <think></think> tags
  - Search triggering via <|begin_of_query|>...<|end_of_query|> tokens
  - Document injection via <|begin_of_documents|>...<|end_of_documents|>
  - Answer extraction via <answer>...</answer> tags

Both retriever and generator are loaded simultaneously (required for
interleaved retrieval during reasoning).
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
from flashrag.pipeline import ReasoningPipeline
import faiss


def main():
    parser = argparse.ArgumentParser(description="Run ReasoningPipeline (Search-R1 style)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_retrieval", type=int, default=5,
                        help="Max number of search queries the model can issue (default: 5)")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, top-k: {config['retrieval_topk']}")
    print(f"Max retrieval rounds: {args.max_retrieval}")
    print(f"Max tokens: {config['generation_params']['max_tokens']}")
    print(f"Save dir: {config['save_dir']}")

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    # Initialize ReasoningPipeline (loads both retriever + generator)
    print(f"\n=== Initializing ReasoningPipeline ===")
    t_init = time.time()
    pipeline = ReasoningPipeline(config, max_retrieval_num=args.max_retrieval)
    t_init_done = time.time() - t_init
    print(f"Pipeline initialized in {t_init_done:.1f}s")

    # Fix FAISS threads after vLLM/PyTorch init
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "16"))
    print(f"FAISS OMP threads before fix: {faiss.omp_get_max_threads()}")
    faiss.omp_set_num_threads(omp_threads)
    print(f"FAISS OMP threads after fix: {faiss.omp_get_max_threads()}")

    # Run pipeline
    print(f"\n=== Running ReasoningPipeline ===")
    t_start = time.time()
    output_dataset = pipeline.run(test_data, do_eval=True)
    t_total = time.time() - t_start

    # Print timing and stats
    print(f"\n=== Summary ===")
    print(f"Total time: {t_total:.1f}s ({t_total/len(test_data):.2f}s/example)")

    # Analyze retrieval behavior
    n_retrieved = sum(1 for item in output_dataset if hasattr(item, 'retrieved_times') and item.retrieved_times > 0)
    avg_retrievals = 0
    if n_retrieved > 0:
        total_retrievals = sum(item.retrieved_times for item in output_dataset if hasattr(item, 'retrieved_times'))
        avg_retrievals = total_retrievals / len(output_dataset)
    print(f"Items that triggered search: {n_retrieved}/{len(output_dataset)} ({100*n_retrieved/len(output_dataset):.1f}%)")
    print(f"Average search queries per item: {avg_retrievals:.2f}")

    # Analyze finish reasons
    finish_reasons = {}
    for item in output_dataset:
        reason = getattr(item, 'finish_reason', 'unknown')
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    print(f"Finish reasons: {finish_reasons}")

    # Show sample outputs
    print(f"\n--- Sample outputs (first 3) ---")
    for i in range(min(3, len(output_dataset))):
        item = output_dataset[i]
        print(f"\nQ: {item.question}")
        print(f"Pred: {item.pred[:200] if item.pred else 'None'}")
        print(f"Gold: {item.golden_answers}")
        print(f"Retrievals: {getattr(item, 'retrieved_times', 0)}")

    print(f"\nResults saved to {config['save_dir']}")


if __name__ == "__main__":
    main()
