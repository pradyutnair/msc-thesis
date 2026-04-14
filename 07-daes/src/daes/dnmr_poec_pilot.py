"""DNMR-POEC pilot: Posterior-Optimal Evidence-Curated DNMR.

Methods:
  baseline                  - single decode from C0
  pool                      - single-round DNMR with naive evidence union
  pool_curated_*            - single-round DNMR with posterior-weighted evidence curation
  pool_curated_guided_*     - curated evidence + contrastive guided decode

Example:
  python -u src/daes/dnmr_poec_pilot.py \
      --model dream \
      --dataset musique \
      --n_questions 50 \
      --top_b_values 8 \
      --lambda_values 0.5 \
      --gamma_values 0,0.5 \
      --output results/dnmr_poec_musique_50q.json
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributions as dists

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import eamd_v2_wiki18
from eamd_v2_wiki18 import (
    QUESTION_FILES,
    Wiki18Retriever,
    _neg_entropy,
    build_short_prompt,
    compute_f1,
    decode_answer,
    extract_candidates_generic,
    get_mask_id,
    prepare_logits,
)
from transformers import AutoModel, AutoTokenizer


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def fmt_float(value: float) -> str:
    return str(value).replace(".", "p")


def full_attention_mask(length: int, device):
    if eamd_v2_wiki18.MODEL_TYPE_REF == "dream":
        return "full"
    return torch.ones((1, length), dtype=torch.long, device=device)


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    margin_confidence: bool = False,
    neg_entropy: bool = False,
):
    if temperature > 0:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


def score_answer(pred: str, gold: str) -> dict:
    precision, recall, f1 = compute_f1(pred, gold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": float(pred.strip().lower() == gold.strip().lower()),
        "contain": float(gold.strip().lower() in pred.strip().lower()),
    }


@torch.inference_mode()
def simple_decode(model, tokenizer, context: str, question: str, steps: int = 32, n_tokens: int = 32) -> str:
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = full_attention_mask(len(canvas), device)
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
def guided_decode(
    model,
    tokenizer,
    context: str,
    question: str,
    steps: int = 32,
    n_tokens: int = 32,
    gamma: float = 0.0,
    prior_context: str = "",
) -> str:
    if gamma <= 0.0:
        return simple_decode(model, tokenizer, context, question, steps=steps, n_tokens=n_tokens)

    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ctx, n_prefix_ctx = build_short_prompt(tokenizer, context, question)
    prefix_prior, n_prefix_prior = build_short_prompt(tokenizer, prior_context, question)

    x_ctx = torch.tensor([prefix_ctx + [mask_id] * n_tokens], dtype=torch.long, device=device)
    x_prior = torch.tensor([prefix_prior + [mask_id] * n_tokens], dtype=torch.long, device=device)
    attn_ctx = full_attention_mask(x_ctx.shape[1], device)
    attn_prior = full_attention_mask(x_prior.shape[1], device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    for step in range(steps):
        if remaining <= 0:
            break

        masked_local = (x_ctx[0, n_prefix_ctx:n_prefix_ctx + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        ctx_out = model(x_ctx, attention_mask=attn_ctx)
        prior_out = model(x_prior, attention_mask=attn_prior)
        ctx_logits = prepare_logits(ctx_out.logits)[0, masked_local + n_prefix_ctx]
        prior_logits = prepare_logits(prior_out.logits)[0, masked_local + n_prefix_prior]
        guided_logits = ctx_logits + gamma * (ctx_logits - prior_logits)

        confidence, x0 = sample_tokens(guided_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)

        x_ctx[0, masked_local[topk] + n_prefix_ctx] = x0[topk]
        x_prior[0, masked_local[topk] + n_prefix_prior] = x0[topk]
        remaining -= len(topk)

    return decode_answer(tokenizer, x_ctx[0, n_prefix_ctx:n_prefix_ctx + n_tokens])


def expand_evidence_raw(
    retriever: Wiki18Retriever,
    question: str,
    seed_answer: str,
    bridge_cands: list[dict],
    current_passages: list[str],
    expand_top_k: int = 3,
):
    queries = [f"query: {question} {seed_answer}"]
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if text and len(text) > 1:
            queries.append(f"query: {question} {text}")
    results = retriever.retrieve_batch(queries, expand_top_k)
    existing = set(current_passages)
    new_passages = []
    for result_list in results:
        for passage in result_list:
            if passage not in existing:
                new_passages.append(passage)
                existing.add(passage)
    return list(current_passages) + new_passages, new_passages


def build_query_specs(question: str, seed_answer: str, bridge_cands: list[dict], lambda_ans: float) -> list[dict]:
    lambda_ans = min(max(lambda_ans, 0.0), 1.0)
    valid_bridges = []
    for cand in bridge_cands:
        text = cand.get("text", "").strip() if isinstance(cand, dict) else str(cand).strip()
        if not text or len(text) <= 1:
            continue
        valid_bridges.append(
            {
                "text": text,
                "prob": max(float(cand.get("init_conf", 0.0)), 0.0) if isinstance(cand, dict) else 0.0,
            }
        )

    specs = [{"query": f"query: {question} {seed_answer}", "weight": 1.0, "kind": "seed"}]
    if not valid_bridges:
        return specs

    bridge_mass = sum(item["prob"] for item in valid_bridges)
    seed_weight = lambda_ans
    if bridge_mass > 0:
        bridge_weights = [(1.0 - lambda_ans) * item["prob"] / bridge_mass for item in valid_bridges]
    else:
        uniform = (1.0 - lambda_ans) / len(valid_bridges)
        bridge_weights = [uniform for _ in valid_bridges]

    if seed_weight <= 0 and bridge_mass > 0:
        specs = []

    if seed_weight > 0:
        specs[0]["weight"] = seed_weight

    for item, weight in zip(valid_bridges, bridge_weights):
        if weight <= 0:
            continue
        specs.append(
            {
                "query": f"query: {question} {item['text']}",
                "weight": weight,
                "kind": "bridge",
                "bridge_text": item["text"],
                "bridge_prob": item["prob"],
            }
        )

    total_weight = sum(spec["weight"] for spec in specs)
    if total_weight <= 0:
        return [{"query": f"query: {question} {seed_answer}", "weight": 1.0, "kind": "seed"}]
    for spec in specs:
        spec["weight"] /= total_weight
    return specs


def expand_evidence_curated(
    retriever: Wiki18Retriever,
    question: str,
    seed_answer: str,
    bridge_cands: list[dict],
    current_passages: list[str],
    expand_top_k: int = 3,
    top_b: int = 8,
    lambda_ans: float = 0.5,
):
    query_specs = build_query_specs(question, seed_answer, bridge_cands, lambda_ans)
    scored_results = retriever.retrieve_batch_with_scores([spec["query"] for spec in query_specs], expand_top_k)

    existing = set(current_passages)
    candidate_map = {}
    for spec, result in zip(query_specs, scored_results):
        weight = spec["weight"]
        for hit in result["hits"]:
            passage = hit["passage"]
            if passage in existing:
                continue
            score = max(float(hit["score"]), 0.0)
            contribution = weight * score / (hit["rank"] + 1)
            if passage not in candidate_map:
                candidate_map[passage] = {
                    "passage": passage,
                    "utility": 0.0,
                    "max_score": score,
                    "support": [],
                }
            candidate_map[passage]["utility"] += contribution
            candidate_map[passage]["max_score"] = max(candidate_map[passage]["max_score"], score)
            candidate_map[passage]["support"].append(
                {
                    "query": spec["query"],
                    "weight": weight,
                    "rank": hit["rank"],
                    "score": score,
                    "kind": spec["kind"],
                    "bridge_text": spec.get("bridge_text"),
                }
            )

    ranked = sorted(
        candidate_map.values(),
        key=lambda item: (item["utility"], item["max_score"], len(item["support"])),
        reverse=True,
    )
    selected = ranked[:top_b]
    # Put the highest-utility passage nearest the answer mask.
    ordered_selected = sorted(selected, key=lambda item: item["utility"])
    new_passages = [item["passage"] for item in ordered_selected]
    metadata = {
        "query_specs": query_specs,
        "selected": [
            {
                "utility": round(item["utility"], 6),
                "max_score": round(item["max_score"], 6),
                "support_count": len(item["support"]),
                "passage_preview": item["passage"][:160],
            }
            for item in selected
        ],
        "candidate_pool_size": len(ranked),
    }
    return list(current_passages) + new_passages, new_passages, metadata


def build_method_name(prefix: str, top_b: int, lambda_ans: float, gamma: float | None = None) -> str:
    name = f"{prefix}_b{top_b}_la{fmt_float(lambda_ans)}"
    if gamma is not None:
        name += f"_g{fmt_float(gamma)}"
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--top_b_values", type=str, default="8")
    parser.add_argument("--lambda_values", type=str, default="0.5")
    parser.add_argument("--gamma_values", type=str, default="0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    top_b_values = parse_int_list(args.top_b_values)
    lambda_values = parse_float_list(args.lambda_values)
    gamma_values = parse_float_list(args.gamma_values)

    t_start = time.time()
    print("=== DNMR-POEC Pilot ===", flush=True)
    print(f"Model: {args.model}, Dataset: {args.dataset}", flush=True)
    print(
        f"top_b={top_b_values}, lambda={lambda_values}, gamma={gamma_values}",
        flush=True,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    print("Batch initial retrieval...", flush=True)
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"Initial retrieval done in {time.time() - t_start:.1f}s", flush=True)

    method_names = ["baseline", "pool"]
    for top_b in top_b_values:
        for lambda_ans in lambda_values:
            curated_name = build_method_name("pool_curated", top_b, lambda_ans)
            method_names.append(curated_name)
            for gamma in gamma_values:
                guided_name = build_method_name("pool_curated_guided", top_b, lambda_ans, gamma)
                method_names.append(guided_name)

    totals = {name: {"f1": 0.0, "em": 0.0, "contain": 0.0} for name in method_names}
    results = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        answers = {}
        metrics = {}

        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext, steps=args.steps, n_tokens=args.answer_tokens)
        answers["baseline"] = baseline_ans
        metrics["baseline"] = score_answer(baseline_ans, gold)

        seed_ans = baseline_ans
        pool_cands = extract_candidates_generic(
            model,
            tokenizer,
            old_ctx,
            qtext,
            args.n_candidates,
            extraction_steps=args.extraction_steps,
        )

        pool_passages, raw_new = expand_evidence_raw(
            retriever,
            qtext,
            seed_ans,
            pool_cands,
            initial,
            args.expand_top_k,
        )
        pool_ctx = "\n\n".join(pool_passages)
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext, steps=args.steps, n_tokens=args.answer_tokens)
        answers["pool"] = pool_ans
        metrics["pool"] = score_answer(pool_ans, gold)

        curated_cache = {}
        for top_b in top_b_values:
            for lambda_ans in lambda_values:
                curated_name = build_method_name("pool_curated", top_b, lambda_ans)
                curated_passages, curated_new, curated_meta = expand_evidence_curated(
                    retriever,
                    qtext,
                    seed_ans,
                    pool_cands,
                    initial,
                    expand_top_k=args.expand_top_k,
                    top_b=top_b,
                    lambda_ans=lambda_ans,
                )
                curated_ctx = "\n\n".join(curated_passages)
                curated_ans = simple_decode(
                    model,
                    tokenizer,
                    curated_ctx,
                    qtext,
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                )
                answers[curated_name] = curated_ans
                metrics[curated_name] = score_answer(curated_ans, gold)
                curated_cache[(top_b, lambda_ans)] = {
                    "context": curated_ctx,
                    "answer": curated_ans,
                    "new_passages": curated_new,
                    "meta": curated_meta,
                }

                for gamma in gamma_values:
                    guided_name = build_method_name("pool_curated_guided", top_b, lambda_ans, gamma)
                    if gamma == 0:
                        guided_ans = curated_ans
                    else:
                        guided_ans = guided_decode(
                            model,
                            tokenizer,
                            curated_ctx,
                            qtext,
                            steps=args.steps,
                            n_tokens=args.answer_tokens,
                            gamma=gamma,
                            prior_context="",
                        )
                    answers[guided_name] = guided_ans
                    metrics[guided_name] = score_answer(guided_ans, gold)

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext,
            "gold": gold,
            "elapsed": round(time.time() - tq, 2),
            "seed_answer": seed_ans,
            "pool_candidates": pool_cands,
            "pool_new_passages": len(raw_new),
            "curated_configs": {
                build_method_name("pool_curated", top_b, lambda_ans): {
                    "new_passages": len(curated_cache[(top_b, lambda_ans)]["new_passages"]),
                    "candidate_pool_size": curated_cache[(top_b, lambda_ans)]["meta"]["candidate_pool_size"],
                    "top_selected": curated_cache[(top_b, lambda_ans)]["meta"]["selected"][:3],
                }
                for top_b in top_b_values
                for lambda_ans in lambda_values
            },
        }

        for name in method_names:
            row[name] = {
                "answer": answers[name],
                "f1": round(metrics[name]["f1"], 4),
                "em": metrics[name]["em"],
                "contain": metrics[name]["contain"],
            }
            totals[name]["f1"] += metrics[name]["f1"]
            totals[name]["em"] += metrics[name]["em"]
            totals[name]["contain"] += metrics[name]["contain"]

        results.append(row)

        if (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({row['elapsed']:.1f}s)", flush=True)
            for name in method_names:
                print(
                    f"  {name:36s} {row[name]['answer'][:40]:40s} "
                    f"F1={row[name]['f1']:.3f} EM={row[name]['em']:.0f}",
                    flush=True,
                )

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {
                name: {metric: round(value / max(1, n_done), 4) for metric, value in totals[name].items()}
                for name in method_names
            }
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "results": results,
                        "config": vars(args),
                        "timing": {"elapsed_sec": round(time.time() - t_start, 1)},
                    },
                    f,
                    indent=2,
                )

    summary = {
        name: {metric: round(value / max(1, len(results)), 4) for metric, value in totals[name].items()}
        for name in method_names
    }
    print(f"\n{'Method':<36s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}", flush=True)
    print("-" * 60, flush=True)
    for name in method_names:
        print(
            f"{name:<36s} {summary[name]['f1']:>6.3f} {summary[name]['em']:>6.3f} {summary[name]['contain']:>8.3f}",
            flush=True,
        )
    print(f"\nTotal: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
