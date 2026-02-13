"""Run Standard RAG + Refiner with FlashRAG — split retrieval/refining/generation phases.

Phase 1: Retrieval with full CPU parallelism (before any GPU model).
Phase 2: Load refiner (RECOMP T5 or SelectiveContext GPT-2) on GPU, process, then free it.
Phase 3: Generation with vLLM + evaluation (direct generator call, no pipeline).

Bypasses SequentialPipeline to avoid GPU OOM from pipeline loading refiner + vLLM together.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = Config(config_file_path=args.config)
    all_split = get_dataset(config)
    test_data = all_split["test"]

    # ── Phase 1: Retrieval (CPU-heavy, no GPU needed) ──────────────────────
    print(f"\n=== Phase 1: Retrieval ({len(test_data)} examples) ===")
    import faiss
    from flashrag.retriever import DenseRetriever

    retriever = DenseRetriever(config)
    questions = [item.question for item in test_data]

    t0 = time.time()
    retrieval_results = retriever.batch_search(questions, num=config["retrieval_topk"])
    t_retrieval = time.time() - t0
    print(f"Retrieval: {t_retrieval:.1f}s")

    for i, item in enumerate(test_data):
        item.update_output("retrieval_result", retrieval_results[i])

    # Free retriever
    del retriever
    import gc
    gc.collect()

    # ── Phase 2: Refining (small GPU model) ────────────────────────────────
    print(f"\n=== Phase 2: Refining with {config['refiner_name']} ===")
    import torch
    from flashrag.utils import get_refiner

    refiner = get_refiner(config)

    t1 = time.time()
    refine_results = refiner.batch_run(test_data)
    t_refine = time.time() - t1
    print(f"Refining: {t_refine:.1f}s")

    test_data.update_output("refine_result", refine_results)

    # Measure compression: avg input tokens vs output tokens (approximate by char count)
    total_input_chars = 0
    total_output_chars = 0
    for item, refined in zip(test_data, refine_results):
        for doc in item.retrieval_result:
            total_input_chars += len(doc["contents"])
        total_output_chars += len(refined)
    compression_ratio = total_output_chars / max(total_input_chars, 1)
    print(f"Compression: {total_input_chars} -> {total_output_chars} chars ({compression_ratio:.2%} retained)")

    # Free refiner GPU memory before vLLM
    if hasattr(refiner, 'model'):
        refiner.model.cpu()
        del refiner.model
    if hasattr(refiner, 'refiner') and hasattr(refiner.refiner, 'model'):
        refiner.refiner.model.cpu()
        del refiner.refiner.model
    del refiner
    gc.collect()
    torch.cuda.empty_cache()

    gpu_free = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU free after refiner cleanup: {gpu_free:.1f} GiB")

    # ── Phase 3: Generation + Evaluation (vLLM) ───────────────────────────
    print(f"\n=== Phase 3: Generation with vLLM ===")
    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_generator
    from flashrag.evaluator import Evaluator

    prompt_template = PromptTemplate(config)
    generator = get_generator(config)
    evaluator = Evaluator(config)

    # Build prompts using refined text (formatted_reference, not retrieval_result)
    input_prompts = [
        prompt_template.get_string(question=q, formatted_reference=r)
        for q, r in zip(test_data.question, refine_results)
    ]
    test_data.update_output("prompt", input_prompts)

    t2 = time.time()
    pred_answer_list = generator.generate(input_prompts)
    t_gen = time.time() - t2
    print(f"Generation: {t_gen:.1f}s")

    test_data.update_output("pred", pred_answer_list)

    # Evaluate
    eval_result = evaluator.evaluate(test_data)
    print(f"\n{eval_result}")

    t_total = t_retrieval + t_refine + t_gen
    print(f"\nRetrieval: {t_retrieval:.1f}s | Refining: {t_refine:.1f}s | Generation: {t_gen:.1f}s | Total: {t_total:.1f}s")
    print(f"Compression ratio: {compression_ratio:.2%}")
    print(f"Per example: {t_total/len(test_data):.2f}s")


if __name__ == "__main__":
    main()
