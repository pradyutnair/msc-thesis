"""
MVP Experiment: SPREAD algorithm for Dream-7B on multi-hop QA.

Implements Algorithm 1 from "Unlocking the Potentials of RAG for Diffusion LMs"
Key change: at each denoising step, select which tokens to unmask based on
query-relevance (cosine sim of hidden states to query embedding), NOT confidence.

Usage:
    python mvp_spread.py --dataset hotpotqa --n_questions 50 --mode spread
    python mvp_spread.py --dataset hotpotqa --n_questions 50 --mode baseline
    python mvp_spread.py --dataset hotpotqa --n_questions 50 --mode confidence  (standard infill)
"""

import argparse
import json
import os
import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.core.samplers.base import BaseSamplerOutput


def load_questions(dataset: str, n: int) -> list[dict]:
    path = f"/projects/prjs1800/external/arag/data/{dataset}/questions.json"
    with open(path) as f:
        questions = json.load(f)
    return questions[:n]


def format_evidence(evidence_list: list) -> str:
    passages = []
    for title, sentences in evidence_list:
        text = " ".join(s.strip() for s in sentences)
        passages.append(f"[{title}] {text}")
    return "\n".join(passages)


def encode_query(model, tokenizer, query: str, device) -> torch.Tensor:
    """Encode query using the diffusion model's own hidden states (Algorithm 1, line 2)."""
    tokens = tokenizer.encode(query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    # Use last hidden layer, mean-pool over query tokens
    h_q = outputs.hidden_states[-1].mean(dim=1)  # [1, hidden_dim]
    return F.normalize(h_q, dim=-1)


def spread_infill(
    model, tokenizer, sampler, canvas_ids: list[int],
    query: str, steps: int = 64, temperature: float = 0.1,
    mode: str = "spread",
) -> tuple[str, dict]:
    """
    Custom infill loop implementing SPREAD Algorithm 1.

    mode:
        "spread"     — select tokens by query-relevance (SPREAD)
        "confidence" — select tokens by model confidence (standard)
        "baseline"   — select tokens randomly (ablation baseline)
    """
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Encode query for relevance scoring (Algorithm 1, line 2)
    h_q = encode_query(model, tokenizer, query, device)

    # Build canvas: [context tokens ... MASK MASK ... MASK]
    x = torch.tensor([canvas_ids], dtype=torch.long, device=device)
    T = x.shape[1]
    attention_mask = torch.ones((1, T), dtype=torch.long, device=device)

    # Count masks and compute transfer schedule
    mask_index = (x == mask_id)
    n_masks = mask_index.sum().item()
    if n_masks == 0:
        return "", {"n_masks": 0}

    # Compute how many tokens to unmask per step
    tokens_per_step = max(1, n_masks // steps)
    remaining = n_masks

    stats = {"n_masks": n_masks, "steps_used": 0, "mode": mode}

    for step in range(steps):
        if remaining <= 0:
            break

        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        # Step 1: Forward pass to get hidden states AND logits (Algorithm 1, line 6)
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # [1, T, hidden_dim]

        # AR-shift logits (Dream-specific: predict token at position i from position i-1)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Get predicted tokens and confidence at masked positions
        mask_positions = mask_index[0].nonzero(as_tuple=True)[0]  # indices of masks
        mask_logits = logits[0, mask_positions]  # [n_masked, vocab]

        # Sample tokens (what the model wants to put at each position)
        confidence, x0 = sample_tokens(
            mask_logits, temperature=temperature, neg_entropy=True
        )

        # Decide how many to unmask this step
        k = min(tokens_per_step, remaining)
        if step == steps - 1:
            k = remaining  # unmask all remaining on last step

        # Step 2 & 3: Select which positions to unmask (Algorithm 1, lines 9-18)
        if mode == "spread":
            # SPREAD: select by query relevance
            h_masked = hidden_states[0, mask_positions]  # [n_masked, hidden_dim]
            h_masked_norm = F.normalize(h_masked, dim=-1)
            # Cosine similarity to query (Algorithm 1, line 11)
            sim = (h_masked_norm @ h_q.squeeze(0)).float()  # [n_masked]
            # Sigmoid normalization (Algorithm 1, line 12)
            relevance = torch.sigmoid(sim)
            _, topk_idx = torch.topk(relevance, min(k, len(relevance)))
        elif mode == "confidence":
            # Standard: select by model confidence
            _, topk_idx = torch.topk(confidence, min(k, len(confidence)))
        elif mode == "baseline":
            # Random: select randomly
            perm = torch.randperm(len(mask_positions), device=device)
            topk_idx = perm[:min(k, len(mask_positions))]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Commit selected tokens (Algorithm 1, lines 16-18)
        selected_positions = mask_positions[topk_idx]
        x[0, selected_positions] = x0[topk_idx]
        remaining -= len(topk_idx)
        stats["steps_used"] = step + 1

    # Extract answer (everything after the original context)
    n_context = len([t for t in canvas_ids if t != mask_id])
    answer_ids = x[0, n_context:].tolist()

    # Trim at EOS or remaining masks
    trimmed = []
    for tid in answer_ids:
        if tid == eos_id or tid == mask_id:
            break
        trimmed.append(tid)

    answer = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
    return answer, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", type=str, required=True, choices=["spread", "confidence", "baseline"])
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", type=str, default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_mask", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    from dataclasses import dataclass
    @dataclass
    class ModelArgs:
        model_name_or_path: str = args.model_path
    model_args = ModelArgs()
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    sampler = DreamSampler(model=model, tokenizer=tokenizer)

    print(f"Loading {args.n_questions} questions from {args.dataset}...")
    questions = load_questions(args.dataset, args.n_questions)

    predictions = []
    for i, q in enumerate(questions):
        evidence_str = format_evidence(q["evidence"])
        prompt = (
            f"Based on the following evidence, answer the question concisely.\n\n"
            f"Evidence:\n{evidence_str}\n\n"
            f"Question: {q['question']}\n\n"
            f"Answer:"
        )

        # Build canvas: tokenized prompt + mask tokens for answer
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)
        mask_id = tokenizer.mask_token_id
        canvas = input_ids + [mask_id] * args.n_mask

        t0 = time.time()
        try:
            pred, stats = spread_infill(
                model, tokenizer, sampler, canvas,
                query=q["question"],
                steps=args.steps,
                mode=args.mode,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            pred, stats = "", {"error": str(e)}
        elapsed = time.time() - t0

        predictions.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": pred,
            "mode": args.mode,
            "time": round(elapsed, 2),
            "stats": stats,
        })
        print(f"[{i+1}/{len(questions)}] ({elapsed:.1f}s) Q: {q['question'][:60]}...")
        print(f"  Gold: {q['answer']}")
        print(f"  Pred: {pred[:100]}")

    out_path = os.path.join(args.output_dir, f"mvp_{args.dataset}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\nSaved {len(predictions)} predictions to {out_path}")

    contain_count = sum(
        1 for p in predictions
        if p["gold_answer"].lower() in p["pred_answer"].lower()
    )
    print(f"Quick contain-acc: {contain_count}/{len(predictions)} = {contain_count/len(predictions)*100:.1f}%")


if __name__ == "__main__":
    main()
