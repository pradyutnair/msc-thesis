"""Run Standard RAG + Reranker with FlashRAG — split retrieval/generation phases.

Phase 1: Retrieval + Reranking with full CPU parallelism (before vLLM).
Phase 2: Generation with vLLM + evaluation (no pipeline — direct generator call).

The SequentialPipeline always re-creates retriever+reranker, causing GPU OOM
when vLLM is also loaded. So we bypass the pipeline and call generator directly.
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

    # Phase 1: Retrieval + Reranking
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

    # Free retriever + reranker GPU memory before vLLM
    import torch, gc
    if hasattr(retriever, 'reranker') and retriever.reranker is not None:
        if hasattr(retriever.reranker, 'ranker'):
            retriever.reranker.ranker.cpu()
            del retriever.reranker.ranker
        del retriever.reranker
    del retriever
    gc.collect()
    torch.cuda.empty_cache()

    gpu_free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"Retriever+Reranker freed. GPU free: {gpu_free:.1f} GiB")

    # Phase 2: Direct generation + evaluation (bypass SequentialPipeline)
    print(f"\n=== Phase 2: Generation + Evaluation ===")
    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_generator
    from flashrag.evaluator import Evaluator

    prompt_template = PromptTemplate(config)
    generator = get_generator(config)
    evaluator = Evaluator(config)

    # Build prompts from pre-retrieved + reranked docs
    input_prompts = [
        prompt_template.get_string(question=q, retrieval_result=r)
        for q, r in zip(test_data.question, test_data.retrieval_result)
    ]
    test_data.update_output("prompt", input_prompts)

    # Generate
    t_gen_start = time.time()
    pred_answer_list = generator.generate(input_prompts)
    test_data.update_output("pred", pred_answer_list)
    t_gen = time.time() - t_gen_start
    print(f"Generation: {t_gen:.1f}s")

    # Evaluate
    eval_result = evaluator.evaluate(test_data)
    print(eval_result)

    t_total = t_retrieval + t_gen
    print(f"\nRetrieval+Rerank: {t_retrieval:.1f}s | Generation: {t_gen:.1f}s | Total: {t_total:.1f}s")
    print(f"Per example: {t_total/len(test_data):.2f}s")
    print(f"Results saved to {config['save_dir']}")

if __name__ == "__main__":
    main()
