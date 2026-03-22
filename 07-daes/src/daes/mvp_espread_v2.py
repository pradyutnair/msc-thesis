"""
Evidence-Grounded SPREAD v2 — matches SPREAD paper's exact evaluation protocol.

Key changes from v1:
  - 512 max tokens (not 64)
  - No "concisely" in prompt — match SPREAD's RAG setup
  - Uses sample() with SPREAD token selection (not infill)
  - Computes F1, Precision, Recall (SPREAD's metrics)

Modes:
  - "baseline"    : Dream-7B standard denoising (entropy)
  - "spread"      : vanilla SPREAD (query-relevance token selection)
  - "espread"     : evidence-grounded SPREAD (query + evidence relevance)
  - "iter_espread": iterative retrieval + evidence-grounded SPREAD
"""

import argparse
import json
import os
import sys
import time
import re
import string
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from dllm.core.samplers.base import BaseSamplerOutput


# ---------------------------------------------------------------------------
# Metrics (match SPREAD paper: standard QA token-level P/R/F1)
# ---------------------------------------------------------------------------

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return {"precision": 0, "recall": 0, "f1": 0}
    common = set(pt) & set(gt)
    if not common:
        return {"precision": 0, "recall": 0, "f1": 0}
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    f = 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_questions(dataset: str, n: int) -> list[dict]:
    path = f"/projects/prjs1800/external/arag/data/{dataset}/questions.json"
    with open(path) as f:
        return json.load(f)[:n]


def format_evidence(evidence_list: list) -> str:
    passages = []
    for title, sentences in evidence_list:
        text = " ".join(s.strip() for s in sentences)
        passages.append(f"[{title}] {text}")
    return "\n".join(passages)


def build_prompt(question: str, evidence: str) -> str:
    """Match SPREAD paper setup: context + question, no 'concisely' instruction."""
    return (
        f"Answer the question based on the given information.\n\n"
        f"{evidence}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# SPREAD-style generation using sample() with modified token selection
# ---------------------------------------------------------------------------

def spread_generate(
    model, tokenizer, prompt: str, query: str,
    evidence_text: str = "",
    n_tokens: int = 512,
    steps: int = 128,
    temperature: float = 0.1,
    mode: str = "spread",
    alpha: float = 0.5,
) -> tuple[str, dict]:
    """
    Generate answer using Dream-7B sample() with SPREAD-style token selection.

    This matches SPREAD's actual approach: generate new tokens (not infill),
    but select which tokens to commit based on query/evidence relevance.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Tokenize prompt
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prompt = len(prompt_ids)

    # Build canvas: prompt (fixed) + masked generation region
    canvas_ids = prompt_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas_ids], dtype=torch.long, device=device)
    T = x.shape[1]
    attention_mask = torch.ones((1, T), dtype=torch.long, device=device)

    # Encode query using model's own hidden states (SPREAD Alg 1 line 2)
    query_tokens = tokenizer.encode(query, return_tensors="pt").to(device)
    with torch.no_grad():
        q_out = model(query_tokens, output_hidden_states=True)
    h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)  # [1, D]

    # Evidence token positions (for E-SPREAD): positions in canvas corresponding to evidence
    # Evidence starts after the instruction prefix and ends before "Question:"
    evidence_token_ids = tokenizer.encode(evidence_text, add_special_tokens=False) if evidence_text else []
    # Approximate: evidence tokens are within the prompt region
    # We use all prompt positions as a proxy (evidence dominates the prompt)
    evidence_positions = list(range(n_prompt))

    n_masks = n_tokens
    tokens_per_step = max(1, n_masks // steps)
    remaining = n_masks
    stats = {"mode": mode, "steps_used": 0, "n_prompt": n_prompt}

    for step in range(steps):
        if remaining <= 0:
            break

        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        # Forward pass (SPREAD Alg 1 line 6)
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # [1, T, D]

        # AR-shift logits (Dream-specific)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Masked positions
        mask_positions = mask_index[0].nonzero(as_tuple=True)[0]
        mask_logits = logits[0, mask_positions]

        # Sample tokens
        confidence, x0 = sample_tokens(
            mask_logits, temperature=temperature, neg_entropy=True
        )

        k = min(tokens_per_step, remaining)
        if step == steps - 1:
            k = remaining

        # Token selection strategy
        if mode == "baseline":
            # Standard: select by confidence (Dream default)
            _, topk_idx = torch.topk(confidence, min(k, len(confidence)))

        elif mode == "spread":
            # SPREAD: select by query relevance (Alg 1 lines 9-15)
            h_masked = F.normalize(hidden_states[0, mask_positions], dim=-1)
            sim_q = (h_masked @ h_q.squeeze(0)).float()
            rel_q = torch.sigmoid(sim_q)
            _, topk_idx = torch.topk(rel_q, min(k, len(rel_q)))

        elif mode in ("espread", "iter_espread"):
            # E-SPREAD: query + evidence relevance
            h_masked = F.normalize(hidden_states[0, mask_positions], dim=-1)

            # Query relevance
            sim_q = (h_masked @ h_q.squeeze(0)).float()
            rel_q = torch.sigmoid(sim_q)

            # Evidence relevance
            ev_pos = torch.tensor(evidence_positions, device=device)
            ev_pos = ev_pos[ev_pos < T]
            if len(ev_pos) > 0:
                h_ev = F.normalize(hidden_states[0, ev_pos].mean(dim=0, keepdim=True), dim=-1)
                sim_e = (h_masked @ h_ev.squeeze(0)).float()
                rel_e = torch.sigmoid(sim_e)
                relevance = alpha * rel_q + (1 - alpha) * rel_e
            else:
                relevance = rel_q

            _, topk_idx = torch.topk(relevance, min(k, len(relevance)))

        # Commit tokens
        selected_positions = mask_positions[topk_idx]
        x[0, selected_positions] = x0[topk_idx]
        remaining -= len(topk_idx)
        stats["steps_used"] = step + 1

    # Extract generated text (after prompt) — NO EOS trimming (match SPREAD protocol)
    gen_ids = x[0, n_prompt:].tolist()
    # Keep all non-mask tokens (SPREAD evaluates full denoised output)
    kept = [tid for tid in gen_ids if tid != mask_id]
    answer = tokenizer.decode(kept, skip_special_tokens=True).strip()
    stats["answer_tokens"] = len(kept)
    return answer, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "spread", "espread", "iter_espread"])
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", type=str, default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.5)
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

    print(f"Loading {args.n_questions} questions from {args.dataset}...")
    questions = load_questions(args.dataset, args.n_questions)

    predictions = []
    total_metrics = {"f1": 0, "precision": 0, "recall": 0}

    for i, q in enumerate(questions):
        evidence_str = format_evidence(q["evidence"])

        if args.mode == "iter_espread":
            # Round 1: partial evidence + SPREAD
            evidence = q["evidence"]
            half = max(1, (len(evidence) + 1) // 2)
            r1_evidence = format_evidence(evidence[:half])
            prompt_r1 = build_prompt(q["question"], r1_evidence)

            t0 = time.time()
            try:
                prelim, _ = spread_generate(
                    model, tokenizer, prompt_r1, q["question"],
                    evidence_text=r1_evidence,
                    n_tokens=args.n_tokens, steps=args.steps, mode="spread",
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                prelim = ""

            # Round 2: full evidence + E-SPREAD
            r2_evidence = format_evidence(evidence)
            prompt_r2 = build_prompt(q["question"], r2_evidence)
            try:
                pred, stats = spread_generate(
                    model, tokenizer, prompt_r2, q["question"],
                    evidence_text=r2_evidence,
                    n_tokens=args.n_tokens, steps=args.steps,
                    mode="espread", alpha=args.alpha,
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                pred, stats = "", {"error": str(e)}
            elapsed = time.time() - t0

        else:
            prompt = build_prompt(q["question"], evidence_str)
            t0 = time.time()
            try:
                pred, stats = spread_generate(
                    model, tokenizer, prompt, q["question"],
                    evidence_text=evidence_str,
                    n_tokens=args.n_tokens, steps=args.steps,
                    mode=args.mode, alpha=args.alpha,
                )
            except Exception as e:
                import traceback; traceback.print_exc()
                pred, stats = "", {"error": str(e)}
            elapsed = time.time() - t0

        metrics = compute_f1(pred, q["answer"])
        for k in total_metrics:
            total_metrics[k] += metrics[k]

        predictions.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": pred,
            "mode": args.mode,
            "time": round(elapsed, 2),
            "f1": round(metrics["f1"], 4),
            "precision": round(metrics["precision"], 4),
        })

        print(f"[{i+1}/{len(questions)}] ({elapsed:.1f}s) F1={metrics['f1']:.2f} Q: {q['question'][:55]}...")
        print(f"  Gold: {q['answer']}")
        print(f"  Pred: {pred[:120]}")

    out_path = os.path.join(args.output_dir, f"v2_{args.dataset}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    n = len(predictions)
    contain = sum(1 for p in predictions if p["gold_answer"].lower() in p["pred_answer"].lower())
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset}  Mode: {args.mode}  N={n}")
    print(f"  F1:        {total_metrics['f1']/n*100:.1f}%")
    print(f"  Precision: {total_metrics['precision']/n*100:.1f}%")
    print(f"  Recall:    {total_metrics['recall']/n*100:.1f}%")
    print(f"  Contain:   {contain}/{n} = {contain/n*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
