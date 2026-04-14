"""50-question DNMR contrastive pilot for Snellius."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.environ.get("DLLM_PATH", "/projects/prjs1800/msc-thesis/07-daes/dllm"))
sys.path.insert(0, os.environ.get("DAES_PATH", "/projects/prjs1800/msc-thesis/07-daes/src/daes"))
sys.path.insert(0, os.path.join(os.getcwd(), ".remote_edit/src/daes"))

import dllm
import eamd_v2_wiki18
from dllm.pipelines.dream.sampler import sample_tokens
from eamd_v2_wiki18 import (
    QUESTION_FILES,
    Wiki18Retriever,
    _neg_entropy,
    build_short_prompt,
    compute_em,
    compute_f1,
    decode_answer,
    expand_evidence,
    extract_candidates_agnostic,
    get_mask_id,
    prepare_logits,
)


def compute_contain(pred: str, gold: str) -> float:
    return float(gold.strip().lower() in pred.strip().lower())


def get_model_and_tokenizer(model_name: str, device: str):
    if model_name == "dream":
        model_ref = "Dream-org/Dream-v0-Instruct-7B"
        model_args = SimpleNamespace(model_name_or_path=model_ref)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
        return model, tokenizer

    model_ref = "GSAI-ML/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_ref,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return model, tokenizer


def score_answer(pred: str, gold: str) -> dict:
    _precision, _recall, f1 = compute_f1(pred, gold)
    return {
        "answer": pred,
        "f1": round(f1, 4),
        "em": compute_em(pred, gold),
        "contain": compute_contain(pred, gold),
    }


def build_passage_bias(tokenizer, passages: list[str], device: torch.device, beta: float) -> torch.Tensor:
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    bias = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    token_ids = set()
    for passage in passages:
        token_ids.update(tokenizer.encode(passage, add_special_tokens=False))
    if token_ids:
        bias[torch.tensor(sorted(token_ids), dtype=torch.long, device=device)] = beta
    return bias


@torch.inference_mode()
def simple_decode(
    model,
    tokenizer,
    context: str,
    question: str,
    steps: int = 32,
    n_tokens: int = 32,
    logit_bias: torch.Tensor | None = None,
):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)
    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, masked_local + n_prefix]
        if logit_bias is not None:
            # Model logits may be larger than tokenizer vocab (padding); match dims
            if token_logits.shape[-1] > logit_bias.shape[-1]:
                padded = torch.zeros(token_logits.shape[-1], dtype=logit_bias.dtype, device=logit_bias.device)
                padded[:logit_bias.shape[-1]] = logit_bias
                token_logits = token_logits + padded.unsqueeze(0)
            else:
                token_logits = token_logits + logit_bias[:token_logits.shape[-1]].unsqueeze(0)
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


@torch.inference_mode()
def grounded_simple_decode(
    model,
    tokenizer,
    passages: list[str],
    question: str,
    steps: int = 32,
    n_tokens: int = 32,
    beta: float = 1.5,
):
    context = "\n\n".join(passages)
    bias = build_passage_bias(tokenizer, passages, model.device, beta)
    return simple_decode(
        model,
        tokenizer,
        context,
        question,
        steps=steps,
        n_tokens=n_tokens,
        logit_bias=bias,
    )


@torch.inference_mode()
def extract_candidates_contrastive(
    model,
    tokenizer,
    context: str,
    question: str,
    n_candidates: int = 3,
    n_positions: int = 3,
    n_branch: int = 2,
    n_mask: int = 12,
    extraction_steps: int = 12,
    alpha: float = 0.3,
):
    """Contrastive bridge extraction: use (full - alpha * question-only) logits
    for position selection and initial token seeding, then denoise using
    full-context logits only (1 extra forward pass total, not per step)."""
    device = model.device
    mask_id = get_mask_id(tokenizer)

    prefix_full, n_prefix_full = build_short_prompt(tokenizer, context, question)
    prefix_q, n_prefix_q = build_short_prompt(tokenizer, "", question)

    canvas_full = prefix_full + [mask_id] * n_mask
    canvas_q = prefix_q + [mask_id] * n_mask

    x_full = torch.tensor([canvas_full], dtype=torch.long, device=device)
    x_q = torch.tensor([canvas_q], dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(canvas_full)), dtype=torch.long, device=device)
    attn_q = torch.ones((1, len(canvas_q)), dtype=torch.long, device=device)

    # One forward pass each — the q-only pass is the only extra cost
    logits_full = prepare_logits(model(x_full, attention_mask=attn_full).logits)
    logits_q = prepare_logits(model(x_q, attention_mask=attn_q).logits)
    answer_logits_full = logits_full[0, n_prefix_full:n_prefix_full + n_mask]
    answer_logits_q = logits_q[0, n_prefix_q:n_prefix_q + n_mask]

    # Contrastive logits for position + token selection only
    answer_logits_contrastive = answer_logits_full - alpha * answer_logits_q

    probs = torch.softmax(answer_logits_contrastive / 0.3, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    entropy_positions = torch.topk(entropy, min(n_positions, n_mask)).indices.tolist()
    top_positions = [0]
    for pos in entropy_positions:
        if pos not in top_positions:
            top_positions.append(pos)
        if len(top_positions) >= n_positions + 1:
            break

    # Seed branches using contrastive logits
    branch_canvases = []
    branch_meta = []
    for pos_local in top_positions:
        pos_probs = torch.softmax(answer_logits_contrastive[pos_local] / 0.3, dim=-1)
        top_probs, top_ids = torch.topk(pos_probs, min(n_branch, pos_probs.shape[-1]))
        for token_prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            canvas = list(canvas_full)
            canvas[n_prefix_full + pos_local] = token_id
            branch_canvases.append(canvas)
            branch_meta.append({"position": pos_local, "init_conf": float(token_prob)})

    if not branch_canvases:
        return []

    # Denoising uses full-context logits only (no extra q-only passes)
    x_all = torch.tensor(branch_canvases, dtype=torch.long, device=device)
    attn_batch = torch.ones((len(branch_canvases), x_all.shape[1]), dtype=torch.long, device=device)
    remaining = torch.full((len(branch_canvases),), n_mask - 1, dtype=torch.long, device=device)
    neg_ent = _neg_entropy()

    for step in range(extraction_steps):
        active = remaining > 0
        if not active.any():
            break
        active_idx = active.nonzero(as_tuple=True)[0]
        logits_active = prepare_logits(
            model(x_all[active_idx], attention_mask=attn_batch[: len(active_idx)]).logits
        )
        for j, bi in enumerate(active_idx.tolist()):
            mi = (x_all[bi] == mask_id)
            if not mi.any():
                remaining[bi] = 0
                continue
            mp = mi.nonzero(as_tuple=True)[0]
            conf, sampled = sample_tokens(logits_active[j, mp], temperature=0.1, neg_entropy=neg_ent)
            rem = remaining[bi].item()
            n_commit = min(max(1, rem // extraction_steps), rem)
            if step == extraction_steps - 1:
                n_commit = rem
            _, topk = torch.topk(conf, min(n_commit, len(conf)))
            x_all[bi, mp[topk]] = sampled[topk]
            remaining[bi] -= len(topk)

    candidates = []
    seen = set()
    for bi, meta in enumerate(branch_meta):
        cand_text = tokenizer.decode(
            x_all[bi, n_prefix_full:n_prefix_full + n_mask].tolist(),
            skip_special_tokens=True,
        ).strip()
        cand_text = cand_text.split("\n")[0].split(". ")[0].strip()
        if cand_text and len(cand_text) > 1 and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append(
                {
                    "text": cand_text,
                    "init_conf": meta["init_conf"],
                    "position": meta["position"],
                }
            )
            if len(candidates) >= n_candidates:
                break
    return candidates


def expand_with_candidates(
    retriever: Wiki18Retriever,
    question: str,
    seed_answer: str,
    bridge_cands: list[dict],
    current_passages: list[str],
    expand_top_k: int = 3,
) -> list[str]:
    queries = [f"query: {question} {seed_answer}"]
    for cand in bridge_cands:
        text = cand.get("text", "").strip()
        if text and len(text) > 1:
            queries.append(f"query: {question} {text}")
    results = retriever.retrieve_batch(queries, expand_top_k)
    passages = list(current_passages)
    seen = set(current_passages)
    for result_list in results:
        for passage in result_list:
            if passage not in seen:
                passages.append(passage)
                seen.add(passage)
    return passages


def save_checkpoint(output_path: str, args, started_at: float, results: list[dict]):
    methods = [
        "baseline_10",
        "pool_vanilla",
        "pool_contrastive",
        "pool_grounded",
        "pool_contrastive_grounded",
    ]
    summary = {}
    for method in methods:
        if not results:
            summary[method] = {"f1": 0.0, "em": 0.0, "contain": 0.0}
            continue
        summary[method] = {
            "f1": round(sum(row[method]["f1"] for row in results) / len(results), 4),
            "em": round(sum(row[method]["em"] for row in results) / len(results), 4),
            "contain": round(sum(row[method]["contain"] for row in results) / len(results), 4),
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "results": results,
                "config": vars(args),
                "timing": {"elapsed_sec": round(time.time() - started_at, 1)},
            },
            handle,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", required=True, choices=list(QUESTION_FILES.keys()))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=1.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--question_ids", type=str, default=None,
                        help="Comma-separated question IDs to run (e.g. dev_42,dev_48). If not set, uses start_idx/n_questions.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started_at = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = get_model_and_tokenizer(args.model, args.device)
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model

    retriever = Wiki18Retriever(device=args.device)
    with open(QUESTION_FILES[args.dataset], "r", encoding="utf-8") as handle:
        all_questions = json.load(handle)

    if args.question_ids:
        target_ids = set(args.question_ids.split(","))
        questions = [q for q in all_questions if (q.get("qid") or q.get("id", "")) in target_ids]
        print(f"Filtered to {len(questions)} questions from {len(target_ids)} requested IDs", flush=True)
    else:
        questions = all_questions[args.start_idx : args.start_idx + args.n_questions]

    query_texts = [f"query: {item['question']}" for item in questions]
    initial_passages_batch = retriever.retrieve_batch(query_texts, args.initial_top_k)

    # Also retrieve baseline_10 passages (matched budget control)
    baseline10_queries = [f"query: {item['question']}" for item in questions]
    baseline10_passages_batch = retriever.retrieve_batch(baseline10_queries, 10)

    results = []
    for idx, item in enumerate(questions):
        q_started = time.time()
        question = item["question"]
        gold = item.get("answer") or (item.get("golden_answers") or [""])[0]
        initial_passages = initial_passages_batch[idx]
        initial_context = "\n\n".join(initial_passages)

        # === baseline_10: matched-budget control (10 passages, no bridge extraction) ===
        baseline10_context = "\n\n".join(baseline10_passages_batch[idx])
        baseline10_answer = simple_decode(
            model, tokenizer, baseline10_context, question,
            steps=args.steps, n_tokens=args.answer_tokens,
        )

        # === Extract candidates: vanilla + contrastive ===
        vanilla_candidates = extract_candidates_agnostic(
            model, tokenizer, initial_context, question,
            n_candidates=args.n_candidates, extraction_steps=args.steps,
        )
        contrastive_candidates = extract_candidates_contrastive(
            model, tokenizer, initial_context, question,
            n_candidates=args.n_candidates, extraction_steps=args.steps,
            alpha=args.alpha,
        )

        # === Expand evidence ===
        expanded_vanilla = expand_with_candidates(
            retriever, question, baseline10_answer, vanilla_candidates,
            initial_passages, expand_top_k=args.expand_top_k,
        )
        expanded_contrastive = expand_with_candidates(
            retriever, question, baseline10_answer, contrastive_candidates,
            initial_passages, expand_top_k=args.expand_top_k,
        )

        # === pool_vanilla: standard extraction + standard decode (control) ===
        pool_vanilla_answer = simple_decode(
            model, tokenizer, "\n\n".join(expanded_vanilla), question,
            steps=args.steps, n_tokens=args.answer_tokens,
        )

        # === pool_contrastive: contrastive extraction + standard decode ===
        pool_contrastive_answer = simple_decode(
            model, tokenizer, "\n\n".join(expanded_contrastive), question,
            steps=args.steps, n_tokens=args.answer_tokens,
        )

        # === pool_grounded: vanilla extraction + grounded decode ===
        pool_grounded_answer = grounded_simple_decode(
            model, tokenizer, expanded_vanilla, question,
            steps=args.steps, n_tokens=args.answer_tokens, beta=args.beta,
        )

        # === pool_contrastive_grounded: contrastive extraction + grounded decode ===
        pool_cg_answer = grounded_simple_decode(
            model, tokenizer, expanded_contrastive, question,
            steps=args.steps, n_tokens=args.answer_tokens, beta=args.beta,
        )

        row = {
            "id": item.get("qid") or item.get("id", f"{args.dataset}_{args.start_idx + idx}"),
            "question": question,
            "gold": gold,
            "elapsed": round(time.time() - q_started, 2),
            "vanilla_candidates": [c.get("text", "") for c in vanilla_candidates],
            "contrastive_candidates": [c.get("text", "") for c in contrastive_candidates],
            "n_passages_vanilla": len(expanded_vanilla),
            "n_passages_contrastive": len(expanded_contrastive),
            "baseline_10": score_answer(baseline10_answer, gold),
            "pool_vanilla": score_answer(pool_vanilla_answer, gold),
            "pool_contrastive": score_answer(pool_contrastive_answer, gold),
            "pool_grounded": score_answer(pool_grounded_answer, gold),
            "pool_contrastive_grounded": score_answer(pool_cg_answer, gold),
        }
        results.append(row)

        if (idx + 1) % 5 == 0 or idx == 0:
            n = len(results)
            def _avg(m, k): return sum(r[m][k] for r in results) / n
            print(
                f"[{idx+1}/{len(questions)}] "
                f"bl10_c={_avg('baseline_10','contain'):.2f} "
                f"van_c={_avg('pool_vanilla','contain'):.2f} ctr_c={_avg('pool_contrastive','contain'):.2f} "
                f"gnd_c={_avg('pool_grounded','contain'):.2f} cg_c={_avg('pool_contrastive_grounded','contain'):.2f}",
                flush=True,
            )

        if (idx + 1) % 5 == 0 or idx == len(questions) - 1:
            save_checkpoint(args.output, args, started_at, results)

    save_checkpoint(args.output, args, started_at, results)


if __name__ == "__main__":
    main()
