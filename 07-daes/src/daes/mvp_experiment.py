"""
MVP Experiment: Dream-7B sample() vs infill() on multi-hop QA
Tests whether bidirectional evidence synthesis beats left-to-right generation.

Usage:
    python mvp_experiment.py --dataset hotpotqa --n_questions 50 --mode sample
    python mvp_experiment.py --dataset hotpotqa --n_questions 50 --mode infill
"""

import argparse
import json
import os
import sys
import time
import torch

# Add dllm to path
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")

import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig


def load_questions(dataset: str, n: int) -> list[dict]:
    """Load n questions from ARAG dataset."""
    path = f"/projects/prjs1800/external/arag/data/{dataset}/questions.json"
    with open(path) as f:
        questions = json.load(f)
    return questions[:n]


def format_evidence(evidence_list: list) -> str:
    """Format gold evidence passages into a context string."""
    passages = []
    for title, sentences in evidence_list:
        text = " ".join(s.strip() for s in sentences)
        passages.append(f"[{title}] {text}")
    return "\n".join(passages)


def build_prompt(question: str, evidence: str) -> str:
    """Build a simple QA prompt with evidence context."""
    return (
        f"Based on the following evidence, answer the question concisely.\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def run_sample_mode(sampler, tokenizer, prompt: str, config: DreamSamplerConfig) -> str:
    """Generate answer using standard left-to-right sample()."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(sampler.model.device)

    # Generate with sample()
    output = sampler.sample(input_ids, config)
    if hasattr(output, "sequences"):
        output_ids = output.sequences[0]
    else:
        output_ids = output[0]

    # Decode only the new tokens
    generated = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True)
    return generated.strip()


def run_infill_mode(sampler, tokenizer, prompt: str, config: DreamSamplerConfig, n_mask: int = 64) -> str:
    """Generate answer using infill() — bidirectional denoising over masked answer region."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)

    # Append mask tokens for the answer region
    mask_id = tokenizer.mask_token_id
    canvas = input_ids + [mask_id] * n_mask

    # Run infill — only mask tokens get denoised, evidence tokens stay fixed
    output = sampler.infill([canvas], config)
    if hasattr(output, "sequences"):
        output_ids = output.sequences[0]
    else:
        output_ids = output[0]

    # Extract the answer region (after the prompt)
    answer_ids = output_ids[len(input_ids):]
    # Trim at EOS or mask tokens
    eos_id = tokenizer.eos_token_id
    answer_tokens = []
    for tid in answer_ids:
        t = tid.item() if hasattr(tid, "item") else tid
        if t == eos_id or t == mask_id:
            break
        answer_tokens.append(t)

    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["hotpotqa", "musique", "2wikimultihop"])
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", type=str, required=True, choices=["sample", "infill"])
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", type=str, default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n_mask", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")

    # Use dllm's model loader (registers Dream with transformers)
    from dataclasses import dataclass
    @dataclass
    class ModelArgs:
        model_name_or_path: str = args.model_path
    model_args = ModelArgs()
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    sampler = DreamSampler(model=model, tokenizer=tokenizer)

    # Configure sampler
    config = DreamSamplerConfig(
        max_new_tokens=128 if args.mode == "sample" else None,
        steps=args.steps,
        temperature=0.1,
        alg="entropy",
        return_dict=False,
    )

    print(f"Loading {args.n_questions} questions from {args.dataset}...")
    questions = load_questions(args.dataset, args.n_questions)

    predictions = []
    for i, q in enumerate(questions):
        evidence_str = format_evidence(q["evidence"])
        prompt = build_prompt(q["question"], evidence_str)

        t0 = time.time()
        try:
            if args.mode == "sample":
                pred = run_sample_mode(sampler, tokenizer, prompt, config)
            else:
                pred = run_infill_mode(sampler, tokenizer, prompt, config, n_mask=args.n_mask)
        except Exception as e:
            print(f"  ERROR on question {i}: {e}")
            pred = ""
        elapsed = time.time() - t0

        predictions.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": pred,
            "mode": args.mode,
            "time": round(elapsed, 2),
        })
        print(f"[{i+1}/{len(questions)}] ({elapsed:.1f}s) Q: {q['question'][:60]}...")
        print(f"  Gold: {q['answer']}")
        print(f"  Pred: {pred[:100]}")

    # Save predictions
    out_path = os.path.join(args.output_dir, f"mvp_{args.dataset}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved {len(predictions)} predictions to {out_path}")

    # Quick self-eval: exact substring match
    contain_count = sum(
        1 for p in predictions
        if p["gold_answer"].lower() in p["pred_answer"].lower()
    )
    print(f"Quick contain-acc: {contain_count}/{len(predictions)} = {contain_count/len(predictions)*100:.1f}%")


if __name__ == "__main__":
    main()
