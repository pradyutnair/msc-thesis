"""Run FLARE (Forward-Looking Active Retrieval) with FlashRAG.

FLARE (Jiang et al., EMNLP 2023) generates text and triggers retrieval when
the model's token-level confidence drops below a threshold. Unlike IRCoT
(planned retrieval), FLARE is reactive — it retrieves only when uncertain.

Key difference from IRCoT:
  - IRCoT: fixed interleaving pattern (think → retrieve → think → retrieve)
  - FLARE: uncertainty-driven (generate → check confidence → retrieve if needed)

Note: FLARE processes items sequentially (no batching), so it's slower than
IRCoT. Both retriever and generator must be in memory simultaneously.
"""

import argparse
import os
import time

# Set threading BEFORE imports for FAISS parallelism
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import FLAREPipeline


def main():
    parser = argparse.ArgumentParser(description="Run FLARE active retrieval")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Confidence threshold for triggering retrieval (default: 0.2)")
    parser.add_argument("--look_ahead_steps", type=int, default=64,
                        help="Tokens to look ahead for confidence (default: 64)")
    parser.add_argument("--max_generation_length", type=int, default=256,
                        help="Max total generation length (default: 256)")
    parser.add_argument("--max_iter_num", type=int, default=5,
                        help="Max retrieval iterations per item (default: 5)")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, top-k: {config['retrieval_topk']}")
    print(f"FLARE params: threshold={args.threshold}, look_ahead={args.look_ahead_steps}, "
          f"max_gen_len={args.max_generation_length}, max_iter={args.max_iter_num}")
    print(f"Save dir: {config['save_dir']}")

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")
    print(f"WARNING: FLARE processes items sequentially (no batching). "
          f"Estimated time: {len(test_data) * 1.5 / 60:.0f}-{len(test_data) * 5 / 60:.0f} minutes")

    # Create FLARE pipeline (loads retriever + generator simultaneously)
    print("\nInitializing FLARE pipeline...")
    t_init = time.time()
    pipeline = FLAREPipeline(
        config,
        threshold=args.threshold,
        look_ahead_steps=args.look_ahead_steps,
        max_generation_length=args.max_generation_length,
        max_iter_num=args.max_iter_num,
    )
    print(f"Pipeline initialized in {time.time() - t_init:.1f}s")

    # Run FLARE
    print(f"\n=== Running FLARE on {len(test_data)} examples ===")
    t_start = time.time()
    output = pipeline.run(test_data, do_eval=True)
    t_total = time.time() - t_start

    print(f"\n=== Timing ===")
    print(f"Total: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Per example: {t_total/len(test_data):.2f}s")
    print(f"Results saved to {config['save_dir']}")


if __name__ == "__main__":
    main()
