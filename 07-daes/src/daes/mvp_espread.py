"""
Evidence-Grounded Iterative SPREAD for Multi-Hop QA.

Three modes:
  - "spread"       : vanilla SPREAD (query-relevance only, single-shot retrieval)
  - "espread"      : evidence-grounded SPREAD (query + evidence relevance, single-shot)
  - "iter_espread"  : iterative retrieval + evidence-grounded SPREAD

Usage:
    python mvp_espread.py --dataset musique --n_questions 50 --mode spread
    python mvp_espread.py --dataset musique --n_questions 50 --mode espread
    python mvp_espread.py --dataset musique --n_questions 50 --mode iter_espread
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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_questions(dataset: str, n: int) -> list[dict]:
    path = f"/projects/prjs1800/external/arag/data/{dataset}/questions.json"
    with open(path) as f:
        questions = json.load(f)
    return questions[:n]


def format_evidence(evidence_list: list) -> str:
    """Format gold evidence passages into context string."""
    passages = []
    for title, sentences in evidence_list:
        text = " ".join(s.strip() for s in sentences)
        passages.append(f"[{title}] {text}")
    return "\n".join(passages)


def build_prompt(question: str, evidence: str) -> str:
    return (
        f"Based on the following evidence, answer the question concisely.\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Core: Evidence-Grounded SPREAD denoising
# ---------------------------------------------------------------------------

def encode_query(model, tokenizer, text: str, device) -> torch.Tensor:
    """Encode text using the diffusion model's own hidden states (SPREAD Alg 1 line 2)."""
    tokens = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
    h = outputs.hidden_states[-1].mean(dim=1)  # [1, hidden_dim]
    return F.normalize(h, dim=-1)


def espread_infill(
    model, tokenizer, canvas_ids: list[int],
    query: str,
    evidence_token_positions: list[int] | None = None,
    alpha: float = 0.5,
    steps: int = 128,
    temperature: float = 0.1,
    mode: str = "espread",
) -> tuple[str, dict]:
    """
    Evidence-Grounded SPREAD infilling.

    At each denoising step, select which tokens to unmask based on:
      - "spread":  query-relevance only (vanilla SPREAD)
      - "espread": alpha * query_relevance + (1-alpha) * evidence_relevance
      - "confidence": standard confidence-based (ablation baseline)

    evidence_token_positions: indices in the canvas that correspond to evidence
        tokens (used for computing h_e). If None, evidence grounding is skipped.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Encode query embedding (SPREAD Alg 1 line 2)
    h_q = encode_query(model, tokenizer, query, device)  # [1, D]

    # Build canvas
    x = torch.tensor([canvas_ids], dtype=torch.long, device=device)
    T = x.shape[1]
    attention_mask = torch.ones((1, T), dtype=torch.long, device=device)

    n_masks = (x == mask_id).sum().item()
    if n_masks == 0:
        return "", {"n_masks": 0}

    tokens_per_step = max(1, n_masks // steps)
    remaining = n_masks
    stats = {"n_masks": n_masks, "steps_used": 0, "mode": mode}

    for step in range(steps):
        if remaining <= 0:
            break

        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        # Step 1: Forward pass → hidden states + logits (SPREAD Alg 1 line 6)
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # [1, T, D]

        # AR-shift logits (Dream-specific)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Get masked positions
        mask_positions = mask_index[0].nonzero(as_tuple=True)[0]
        mask_logits = logits[0, mask_positions]

        # Sample tokens at masked positions
        confidence, x0 = sample_tokens(
            mask_logits, temperature=temperature, neg_entropy=True
        )

        k = min(tokens_per_step, remaining)
        if step == steps - 1:
            k = remaining

        # Step 2: Compute relevance scores (SPREAD Alg 1 lines 9-14)
        h_masked = hidden_states[0, mask_positions]  # [n_masked, D]
        h_masked_norm = F.normalize(h_masked, dim=-1)

        # Query relevance
        sim_q = (h_masked_norm @ h_q.squeeze(0)).float()
        rel_q = torch.sigmoid(sim_q)

        if mode == "espread" and evidence_token_positions is not None and len(evidence_token_positions) > 0:
            # Evidence relevance: mean-pool hidden states over evidence positions
            ev_positions = torch.tensor(evidence_token_positions, device=device)
            # Clamp to valid range
            ev_positions = ev_positions[ev_positions < T]
            if len(ev_positions) > 0:
                h_evidence = hidden_states[0, ev_positions].mean(dim=0, keepdim=True)  # [1, D]
                h_evidence_norm = F.normalize(h_evidence, dim=-1)
                sim_e = (h_masked_norm @ h_evidence_norm.squeeze(0)).float()
                rel_e = torch.sigmoid(sim_e)
                # Combined relevance
                relevance = alpha * rel_q + (1 - alpha) * rel_e
            else:
                relevance = rel_q
        elif mode == "spread":
            relevance = rel_q
        else:
            # Fallback to query-only
            relevance = rel_q

        # Step 3: Select top-k by relevance and commit (SPREAD Alg 1 lines 15-18)
        _, topk_idx = torch.topk(relevance, min(k, len(relevance)))
        selected_positions = mask_positions[topk_idx]
        x[0, selected_positions] = x0[topk_idx]
        remaining -= len(topk_idx)
        stats["steps_used"] = step + 1

    # Extract answer from mask region
    n_context = sum(1 for t in canvas_ids if t != mask_id)
    answer_ids = x[0, n_context:].tolist()
    trimmed = []
    for tid in answer_ids:
        if tid == eos_id or tid == mask_id:
            break
        trimmed.append(tid)

    answer = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
    return answer, stats


# ---------------------------------------------------------------------------
# Iterative retrieval (simulated with gold evidence split)
# ---------------------------------------------------------------------------

def simulate_iterative_retrieval(question_data: dict) -> tuple[str, str, list[int]]:
    """
    Simulate 2-round iterative retrieval using gold evidence.

    Round 1: Use first half of evidence passages (simulates initial retrieval).
    Round 2: Use ALL evidence passages (simulates retrieval after extracting bridge entity).

    Returns: (round1_evidence, round2_full_evidence, evidence_token_positions_placeholder)
    """
    evidence = question_data["evidence"]
    n = len(evidence)
    if n <= 1:
        # Single passage — no iteration possible
        return format_evidence(evidence), format_evidence(evidence), []

    # Round 1: first ceil(n/2) passages
    half = (n + 1) // 2
    round1_evidence = evidence[:half]
    # Round 2: all passages
    round2_evidence = evidence

    return format_evidence(round1_evidence), format_evidence(round2_evidence), []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["spread", "espread", "iter_espread", "confidence"])
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", type=str, default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_mask", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for query vs evidence relevance")
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
    mask_id = tokenizer.mask_token_id

    predictions = []
    for i, q in enumerate(questions):

        if args.mode == "iter_espread":
            # --- Iterative retrieval: 2 rounds ---
            r1_evidence, r2_evidence, _ = simulate_iterative_retrieval(q)

            # Round 1: generate preliminary answer with SPREAD (query-only)
            prompt_r1 = build_prompt(q["question"], r1_evidence)
            messages_r1 = [{"role": "user", "content": prompt_r1}]
            input_text_r1 = tokenizer.apply_chat_template(
                messages_r1, tokenize=False, add_generation_prompt=True
            )
            input_ids_r1 = tokenizer.encode(input_text_r1, add_special_tokens=False)
            canvas_r1 = input_ids_r1 + [mask_id] * args.n_mask

            t0 = time.time()
            try:
                prelim_answer, _ = espread_infill(
                    model, tokenizer, canvas_r1,
                    query=q["question"], alpha=args.alpha,
                    steps=args.steps, mode="spread",
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                prelim_answer = ""

            # Round 2: generate final answer with E-SPREAD using ALL evidence
            prompt_r2 = build_prompt(q["question"], r2_evidence)
            messages_r2 = [{"role": "user", "content": prompt_r2}]
            input_text_r2 = tokenizer.apply_chat_template(
                messages_r2, tokenize=False, add_generation_prompt=True
            )
            input_ids_r2 = tokenizer.encode(input_text_r2, add_special_tokens=False)
            canvas_r2 = input_ids_r2 + [mask_id] * args.n_mask

            # Compute evidence token positions (everything before the question in the prompt)
            # Simple heuristic: evidence tokens are roughly the first 80% of the prompt
            n_evidence_tokens = int(len(input_ids_r2) * 0.8)
            evidence_positions = list(range(n_evidence_tokens))

            try:
                pred, stats = espread_infill(
                    model, tokenizer, canvas_r2,
                    query=q["question"],
                    evidence_token_positions=evidence_positions,
                    alpha=args.alpha, steps=args.steps, mode="espread",
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                pred, stats = "", {"error": str(e)}
            elapsed = time.time() - t0
            stats["prelim_answer"] = prelim_answer

        else:
            # --- Single-shot: spread, espread, or confidence ---
            evidence_str = format_evidence(q["evidence"])
            prompt = build_prompt(q["question"], evidence_str)
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)
            canvas = input_ids + [mask_id] * args.n_mask

            # Evidence positions for E-SPREAD
            evidence_positions = None
            if args.mode == "espread":
                n_evidence_tokens = int(len(input_ids) * 0.8)
                evidence_positions = list(range(n_evidence_tokens))

            t0 = time.time()
            try:
                pred, stats = espread_infill(
                    model, tokenizer, canvas,
                    query=q["question"],
                    evidence_token_positions=evidence_positions,
                    alpha=args.alpha, steps=args.steps, mode=args.mode,
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                pred, stats = "", {"error": str(e)}
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
