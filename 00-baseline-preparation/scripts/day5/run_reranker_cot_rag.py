"""Run Standard RAG + Reranker + CoT Prompting with FlashRAG.

Day 5 Priority Experiment: Single-agent ceiling baseline.
Combines best retrieval (reranker from top-20) with CoT reasoning prompt.

Key differences from Day 2 run_reranker_rag.py:
  - CoT system prompt instructs step-by-step reasoning
  - max_tokens increased to 256 (to allow reasoning output)
  - Answer extraction: parses "the answer is:" from CoT output
  - Saves both raw CoT output and extracted answer for analysis

Phase 1: Retrieval + Reranking with full CPU parallelism (before vLLM).
Phase 2: Generation with CoT prompt + answer extraction + evaluation.
"""

import argparse
import os
import re
import time

# CRITICAL: Set threading BEFORE any library import
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.retriever import DenseRetriever


# ── CoT Prompt Template ────────────────────────────────────────────────────
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
    """Clean extracted answer: remove verbose framing, keep only the core answer."""
    answer = answer.strip().rstrip(".")
    # Remove leading "that " / "that the " if model wrote "the answer is that X"
    answer = re.sub(r"^that\s+(the\s+)?", "", answer, flags=re.IGNORECASE)
    # Remove trailing parenthetical remarks
    answer = re.sub(r"\s*\(.*?\)\s*$", "", answer)
    # If answer starts with "Yes/No, ..." and question expects yes/no, keep just yes/no
    # But if there's useful info after, keep the first clause
    # Take text before " because", " since", " as ", " which ", " due to"
    for stop in [" because ", " since ", " as we ", " which ", " due to "]:
        if stop in answer.lower():
            idx = answer.lower().index(stop)
            answer = answer[:idx].strip().rstrip(",")
    return answer.strip().rstrip(".")


def extract_answer_from_cot(cot_output):
    """Extract the final answer from CoT reasoning output.

    Tries multiple patterns in order of specificity.
    Uses the LAST match for "So the answer is" to handle cases
    where the model refines its answer.
    """
    text = cot_output.strip()

    # Pattern 1: "So the answer is: ..." — use LAST occurrence
    matches = list(re.finditer(r"[Ss]o the answer is:?\s*(.+?)(?:\.|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))

    # Pattern 2: "the answer is ..."
    matches = list(re.finditer(r"[Tt]he answer is:?\s*(.+?)(?:\.|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))

    # Pattern 3: "Final answer: ..."
    match = re.search(r"[Ff]inal answer:?\s*(.+?)(?:\.|\n|$)", text)
    if match:
        return _clean_extracted(match.group(1))

    # Pattern 4: Last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return _clean_extracted(lines[-1])

    return text


def main():
    parser = argparse.ArgumentParser(description="Run Standard RAG + Reranker + CoT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, initial top-k: {config['retrieval_topk']}")
    print(f"Reranker: {config['rerank_model_name']}, rerank top-k: {config['rerank_topk']}")
    print(f"Max tokens: {config['generation_params']['max_tokens']}")
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

    # Phase 2: Generation with CoT prompt + evaluation
    print(f"\n=== Phase 2: Generation with CoT Prompt ===")
    from flashrag.prompt import PromptTemplate
    from flashrag.utils import get_generator
    from flashrag.evaluator import Evaluator

    # Use CoT prompt template
    prompt_template = PromptTemplate(
        config=config,
        system_prompt=COT_SYSTEM_PROMPT,
        user_prompt=COT_USER_PROMPT,
    )

    generator = get_generator(config)
    evaluator = Evaluator(config)

    # Build prompts from pre-retrieved + reranked docs
    input_prompts = [
        prompt_template.get_string(question=q, retrieval_result=r)
        for q, r in zip(test_data.question, test_data.retrieval_result)
    ]
    test_data.update_output("prompt", input_prompts)

    # Generate with CoT (longer output due to reasoning)
    t_gen_start = time.time()
    raw_cot_outputs = generator.generate(input_prompts)
    t_gen = time.time() - t_gen_start
    print(f"Generation: {t_gen:.1f}s ({t_gen/len(test_data):.3f}s/example)")

    # Save raw CoT outputs for analysis
    test_data.update_output("raw_cot_output", raw_cot_outputs)

    # Extract answers from CoT outputs
    extracted_answers = [extract_answer_from_cot(cot) for cot in raw_cot_outputs]
    test_data.update_output("pred", extracted_answers)

    # Track extraction stats
    n_pattern1 = sum(1 for cot in raw_cot_outputs if re.search(r"[Ss]o the answer is", cot))
    n_pattern2 = sum(1 for cot in raw_cot_outputs if re.search(r"[Tt]he answer is", cot)) - n_pattern1
    n_pattern3 = sum(1 for cot in raw_cot_outputs if re.search(r"[Ff]inal answer", cot))
    n_fallback = len(raw_cot_outputs) - n_pattern1 - n_pattern2 - n_pattern3
    print(f"\nAnswer extraction stats:")
    print(f"  'So the answer is': {n_pattern1} ({100*n_pattern1/len(raw_cot_outputs):.1f}%)")
    print(f"  'The answer is': {n_pattern2} ({100*n_pattern2/len(raw_cot_outputs):.1f}%)")
    print(f"  'Final answer': {n_pattern3} ({100*n_pattern3/len(raw_cot_outputs):.1f}%)")
    print(f"  Fallback (last line): {n_fallback} ({100*n_fallback/len(raw_cot_outputs):.1f}%)")

    # Show a few examples
    print(f"\n--- Sample CoT outputs (first 3) ---")
    for i in range(min(3, len(raw_cot_outputs))):
        print(f"\nQ: {test_data[i].question}")
        print(f"CoT: {raw_cot_outputs[i][:300]}...")
        print(f"Extracted: {extracted_answers[i]}")
        print(f"Gold: {test_data[i].golden_answers}")

    # Evaluate
    eval_result = evaluator.evaluate(test_data)
    print(f"\n{eval_result}")

    t_total = t_retrieval + t_gen
    print(f"\nRetrieval+Rerank: {t_retrieval:.1f}s | Generation: {t_gen:.1f}s | Total: {t_total:.1f}s")
    print(f"Per example: {t_total/len(test_data):.2f}s")
    print(f"Results saved to {config['save_dir']}")


if __name__ == "__main__":
    main()
