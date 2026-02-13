"""Run SelfAsk Pipeline with FlashRAG.

Day 5: Question decomposition via self-ask.
The model generates follow-up sub-questions, retrieves for each,
then combines intermediate answers into a final answer.

Uses FlashRAG's built-in SelfAskPipeline with multi-hop exemplars.
Both retriever and generator are loaded simultaneously.
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
from flashrag.utils import get_dataset, selfask_pred_parse
from flashrag.pipeline import SelfAskPipeline
import faiss


def main():
    parser = argparse.ArgumentParser(description="Run SelfAsk Pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_iter", type=int, default=5,
                        help="Max self-ask iterations (default: 5)")
    parser.add_argument("--single_hop", action="store_true", default=False,
                        help="Use single-hop exemplars (default: multi-hop)")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, top-k: {config['retrieval_topk']}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Single-hop mode: {args.single_hop}")
    print(f"Save dir: {config['save_dir']}")

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    # Initialize SelfAskPipeline
    print(f"\n=== Initializing SelfAskPipeline ===")
    t_init = time.time()
    pipeline = SelfAskPipeline(
        config,
        max_iter=args.max_iter,
        single_hop=args.single_hop,
    )
    t_init_done = time.time() - t_init
    print(f"Pipeline initialized in {t_init_done:.1f}s")

    # Fix FAISS threads after vLLM/PyTorch init
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "16"))
    print(f"FAISS OMP threads before fix: {faiss.omp_get_max_threads()}")
    faiss.omp_set_num_threads(omp_threads)
    print(f"FAISS OMP threads after fix: {faiss.omp_get_max_threads()}")

    # Run pipeline
    print(f"\n=== Running SelfAskPipeline ===")
    t_start = time.time()
    output_dataset = pipeline.run(test_data, do_eval=True, pred_process_fun=selfask_pred_parse)
    t_total = time.time() - t_start

    print(f"\n=== Summary ===")
    print(f"Total time: {t_total:.1f}s ({t_total/len(test_data):.2f}s/example)")

    # Show sample outputs
    print(f"\n--- Sample outputs (first 3) ---")
    for i in range(min(3, len(output_dataset))):
        item = output_dataset[i]
        print(f"\nQ: {item.question}")
        print(f"Pred: {item.pred[:200] if item.pred else 'None'}")
        print(f"Gold: {item.golden_answers}")

    print(f"\nResults saved to {config['save_dir']}")


if __name__ == "__main__":
    main()
