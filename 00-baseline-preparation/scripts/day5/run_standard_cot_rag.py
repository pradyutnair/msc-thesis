"""Run Standard RAG + CoT Prompting (NO reranker) with FlashRAG.

Completes the 2x2 factorial design:
  - Day 1: Standard RAG (no reranker, no CoT)     — baseline
  - Day 2: + Reranker (no CoT)                     — reranker effect
  - Day 5: + Reranker + CoT                        — reranker + CoT effect
  - THIS:  Standard RAG + CoT (no reranker)        — CoT-only effect

Uses the same CoT prompt as Day 5 Reranker+CoT but with standard E5 top-5
retrieval (no reranking). This isolates the CoT effect from the reranker.

Phase 1: Retrieval with E5 top-5 (before vLLM).
Phase 2: Generation with CoT prompt + answer extraction + evaluation.
"""

import argparse
import os
import re
import time

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.retriever import DenseRetriever


# Same CoT prompt as Day 5 Reranker+CoT
COT_SYSTEM_PROMPT = (
    "Answer the question based on the given documents. "
    "Think step by step:\n"
    "1. What information do I need to answer this question?\n"
    "2. Which documents contain relevant information?\n"
    "3. How do I combine information from multiple documents?\n"
    "4. What is my final answer?\n\n"
    "After your reasoning, provide your final answer in the format: "
    "\"So the answer is: <answer>\"\n"
    "IMPORTANT: The <answer> must be as concise as possible — "
    "a single entity, name, number, yes/no, or short phrase. "
    "Do NOT repeat the question or write a full sentence.\n\n"
    "The following are given documents.\n\n{reference}"
)

COT_USER_PROMPT = "Question: {question}"


def _clean_extracted(answer):
    answer = answer.strip().rstrip(".")
    answer = re.sub(r"^that\s+(the\s+)?", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s*\(.*?\)\s*$", "", answer)
    for stop in [" because ", " since ", " as we ", " which ", " due to "]:
        if stop in answer.lower():
            idx = answer.lower().index(stop)
            answer = answer[:idx].strip().rstrip(",")
    return answer.strip().rstrip(".")


def extract_answer_from_cot(cot_output):
    """Fixed extraction: period only terminates at end-of-string or before newline."""
    text = cot_output.strip()
    matches = list(re.finditer(r"[Ss]o the answer is:?\s*(.+?)(?:\.\s*(?:\n|$)|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))
    matches = list(re.finditer(r"[Tt]he answer is:?\s*(.+?)(?:\.\s*(?:\n|$)|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))
    match = re.search(r"[Ff]inal answer:?\s*(.+?)(?:\.\s*(?:\n|$)|\n|$)", text)
    if match:
        return _clean_extracted(match.group(1))
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return _clean_extracted(lines[-1])
    return text


def main():
    parser = argparse.ArgumentParser(description="Run Standard RAG + CoT (no reranker)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, top-k: {config['retrieval_topk']}")
    print(f"Reranker: {config['use_reranker']}")
    print(f"Max tokens: {config['generation_params']['max_tokens']}")

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    # Phase 1: Retrieval (no reranking)
    import faiss
    print(f"\n=== Phase 1: Retrieval (FAISS threads: {faiss.omp_get_max_threads()}) ===")

    retriever = DenseRetriever(config)
    questions = [item.question for item in test_data]

    t0 = time.time()
    retrieval_results = retriever.batch_search(questions, num=config["retrieval_topk"])
    t_retrieval = time.time() - t0
    print(f"Retrieval: {t_retrieval:.1f}s ({t_retrieval/len(questions):.3f}s/query)")

    for i, item in enumerate(test_data):
        item.update_output("retrieval_result", retrieval_results[i])

    # Free retriever memory
    import torch, gc
    del retriever
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: Generation with CoT
    print(f"\n=== Phase 2: Generation with CoT Prompt ===")
    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_generator
    from flashrag.evaluator import Evaluator

    prompt_template = PromptTemplate(
        config=config,
        system_prompt=COT_SYSTEM_PROMPT,
        user_prompt=COT_USER_PROMPT,
    )

    generator = get_generator(config)
    evaluator = Evaluator(config)

    input_prompts = [
        prompt_template.get_string(question=q, retrieval_result=r)
        for q, r in zip(test_data.question, test_data.retrieval_result)
    ]
    test_data.update_output("prompt", input_prompts)

    t_gen_start = time.time()
    raw_cot_outputs = generator.generate(input_prompts)
    t_gen = time.time() - t_gen_start
    print(f"Generation: {t_gen:.1f}s ({t_gen/len(test_data):.3f}s/example)")

    test_data.update_output("raw_cot_output", raw_cot_outputs)

    extracted_answers = [extract_answer_from_cot(cot) for cot in raw_cot_outputs]
    test_data.update_output("pred", extracted_answers)

    # Extraction stats
    n_pattern1 = sum(1 for cot in raw_cot_outputs if re.search(r"[Ss]o the answer is", cot))
    print(f"\nExtraction: {n_pattern1}/{len(raw_cot_outputs)} matched 'So the answer is' ({100*n_pattern1/len(raw_cot_outputs):.1f}%)")

    # Samples
    for i in range(min(3, len(raw_cot_outputs))):
        print(f"\nQ: {test_data[i].question}")
        print(f"Extracted: {extracted_answers[i]}")
        print(f"Gold: {test_data[i].golden_answers}")

    eval_result = evaluator.evaluate(test_data)
    print(f"\n{eval_result}")

    t_total = t_retrieval + t_gen
    print(f"\nRetrieval: {t_retrieval:.1f}s | Generation: {t_gen:.1f}s | Total: {t_total:.1f}s")


if __name__ == "__main__":
    main()
