"""Run Naive Generation (No Retrieval) — Lower Bound Baseline.

Day 6: Generate answers using only the question, without any retrieved context.
This establishes the lower bound: how well the LLM can answer from parametric
knowledge alone, quantifying the value of retrieval.

Usage:
    python -u scripts/day6/run_naive_generation.py --config configs/day6/naive_gen_qwen25_hotpotqa.yaml
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path


# ── Evaluation functions (same as FlashRAG) ─────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(pred, gold_list):
    norm_pred = normalize_answer(pred)
    return max(float(norm_pred == normalize_answer(g)) for g in gold_list)


def compute_f1(pred, gold_list):
    best = 0.0
    for gold in gold_list:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_tokens)
        rec = num_same / len(gold_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


# ── Naive prompt (no context) ───────────────────────────────────────────────

NAIVE_SYSTEM_PROMPT = (
    "Answer the following question. Give a concise answer: "
    "a single entity, name, number, yes/no, or short phrase."
)

NAIVE_USER_PROMPT = "Question: {question}"


def build_naive_prompt(question, tokenizer):
    """Build chat-formatted prompt with no retrieval context."""
    messages = [
        {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
        {"role": "user", "content": NAIVE_USER_PROMPT.format(question=question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser(description="Naive generation (no retrieval)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load config (reuse FlashRAG config for dataset/generator settings)
    from flashrag.config import Config
    config = Config(config_file_path=args.config)

    dataset_name = config["dataset_name"]
    save_dir = config["save_dir"]
    generator_model = config["generator_model"]
    max_tokens = config["generation_params"]["max_tokens"]

    print(f"=== Naive Generation (No Retrieval) ===")
    print(f"Dataset: {dataset_name}")
    print(f"Generator: {generator_model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Save dir: {save_dir}")

    # Load dataset
    from flashrag.utils import get_dataset
    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(generator_model)

    # Build prompts (no context)
    print(f"\nBuilding naive prompts (no retrieval context)...")
    prompts = [build_naive_prompt(item.question, tokenizer) for item in test_data]

    # Show sample prompt
    print(f"\n--- Sample prompt ---")
    print(prompts[0][:500])
    print(f"---\n")

    # Generate with vLLM
    print(f"Initializing vLLM generator...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=generator_model,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        max_model_len=16384,
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
    )

    print(f"Generating answers for {len(prompts)} questions...")
    t_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_gen = time.time() - t_start
    print(f"Generation: {t_gen:.1f}s ({t_gen/len(prompts):.3f}s/example)")

    predictions = [o.outputs[0].text.strip() for o in outputs]

    # Evaluate
    ems, f1s = [], []
    for i, item in enumerate(test_data):
        gold = item.golden_answers
        pred = predictions[i]
        ems.append(compute_em(pred, gold))
        f1s.append(compute_f1(pred, gold))

    avg_em = sum(ems) / len(ems)
    avg_f1 = sum(f1s) / len(f1s)

    print(f"\n{'='*60}")
    print(f"NAIVE GENERATION RESULTS (n={len(ems)})")
    print(f"{'='*60}")
    print(f"  EM:  {avg_em:.4f} ({100*avg_em:.2f}%)")
    print(f"  F1:  {avg_f1:.4f} ({100*avg_f1:.2f}%)")

    # Show examples
    print(f"\n--- Sample predictions (first 5) ---")
    for i in range(min(5, len(predictions))):
        print(f"  Q: {test_data[i].question}")
        print(f"  Pred: {predictions[i]}")
        print(f"  Gold: {test_data[i].golden_answers}")
        print(f"  EM={ems[i]:.0f} F1={f1s[i]:.3f}")
        print()

    # Save results
    os.makedirs(save_dir, exist_ok=True)

    # Save metric scores
    metric_path = os.path.join(save_dir, "metric_score.txt")
    with open(metric_path, "w") as f:
        f.write(f"em: {avg_em}\n")
        f.write(f"f1: {avg_f1}\n")
    print(f"Saved metrics to {metric_path}")

    # Save intermediate data (matching FlashRAG format for bootstrap analysis)
    intermediate = []
    for i, item in enumerate(test_data):
        intermediate.append({
            "id": item.id if hasattr(item, 'id') else str(i),
            "question": item.question,
            "golden_answers": item.golden_answers,
            "metadata": item.metadata if hasattr(item, 'metadata') else {},
            "output": {
                "pred": predictions[i],
                "prompt": prompts[i],
            },
        })

    data_path = os.path.join(save_dir, "intermediate_data.json")
    with open(data_path, "w") as f:
        json.dump(intermediate, f, indent=2, ensure_ascii=False)
    print(f"Saved intermediate data to {data_path}")

    # Save per-item scores for bootstrap
    scores_path = os.path.join(save_dir, "per_item_scores.json")
    per_item = [{"id": intermediate[i]["id"], "em": ems[i], "f1": f1s[i]}
                for i in range(len(ems))]
    with open(scores_path, "w") as f:
        json.dump(per_item, f, indent=2)
    print(f"Saved per-item scores to {scores_path}")

    print(f"\nTotal time: {t_gen:.1f}s")
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
