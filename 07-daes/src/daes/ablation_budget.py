"""Ablation: bridge candidates vs simply retrieving more passages.

Compares three settings on the same questions and retriever:
  - `baseline_5`: decode from the initial top-5 passages (C0)
  - `baseline_10`: decode from the initial top-10 passages only
  - `dnmr_pool`: start from top-5, then expand via bridge-candidate queries

This isolates whether DNMR gains come from bridge-conditioned expansion rather
than from a larger initial retrieval budget.
"""

import argparse
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/Fast-dLLM/llada")

import dllm
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
    sample_tokens,
)
from generate import (  # type: ignore[reportMissingImports]
    generate as fast_dllm_generate,
    generate_with_dual_cache as fast_dllm_generate_with_dual_cache,
    generate_with_prefix_cache as fast_dllm_generate_with_prefix_cache,
)
from model.modeling_llada import LLaDAModelLM  # type: ignore[reportMissingImports]


@torch.inference_mode()
def simple_decode(
    model,
    tokenizer,
    context,
    question,
    steps=32,
    n_tokens=32,
    decode_backend="vanilla",
    fast_dllm_use_cache=False,
    fast_dllm_dual_cache=False,
    fast_dllm_block_size=32,
    fast_dllm_threshold=None,
):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)

    if decode_backend == "fast-dllm":
        if eamd_v2_wiki18.MODEL_TYPE_REF != "llada":
            raise ValueError("Fast-dLLM decoding is currently supported only for llada in this script.")
        if fast_dllm_dual_cache and not fast_dllm_use_cache:
            raise ValueError("--fast_dllm_dual_cache requires --fast_dllm_use_cache.")
        if fast_dllm_block_size <= 0:
            raise ValueError("--fast_dllm_block_size must be positive.")
        if n_tokens % fast_dllm_block_size != 0:
            raise ValueError(
                f"Fast-dLLM requires answer_tokens ({n_tokens}) to be divisible by "
                f"fast_dllm_block_size ({fast_dllm_block_size})."
            )
        n_blocks = n_tokens // fast_dllm_block_size
        if steps % n_blocks != 0:
            raise ValueError(
                f"Fast-dLLM requires steps ({steps}) to be divisible by the number "
                f"of decode blocks ({n_blocks})."
            )

        prompt = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        generate_fn = fast_dllm_generate
        if fast_dllm_use_cache:
            generate_fn = (
                fast_dllm_generate_with_dual_cache
                if fast_dllm_dual_cache
                else fast_dllm_generate_with_prefix_cache
            )
        output, _ = generate_fn(
            model,
            prompt,
            steps=steps,
            gen_length=n_tokens,
            block_length=fast_dllm_block_size,
            temperature=0.0,
            remasking="low_confidence",
            mask_id=mask_id,
            threshold=fast_dllm_threshold,
        )
        return decode_answer(tokenizer, output[0, n_prefix:n_prefix + n_tokens])

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
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


def expand_evidence(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
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

    return list(current_passages) + new_passages, queries, new_passages


def load_model_and_tokenizer(model_name, decode_backend):
    if model_name == "dream":
        model_args = SimpleNamespace(model_name_or_path="Dream-org/Dream-v0-Instruct-7B")
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        if decode_backend == "fast-dllm":
            model = LLaDAModelLM.from_pretrained(
                "GSAI-ML/LLaDA-8B-Instruct",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).cuda().eval()
        else:
            model = AutoModel.from_pretrained(
                "GSAI-ML/LLaDA-8B-Instruct",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).cuda().eval()
    return model, tokenizer


def contain_metric(answer, gold):
    return float(gold.strip().lower() in answer.strip().lower())


def shorten_answer(answer, max_words=6):
    text = answer.strip()
    if not text:
        return text

    text = text.replace("\r", " ").strip()
    text = text.split("\n", 1)[0].strip()
    text = re.sub(r"^(?:short answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:the answer is|it is|it's)\s+", "", text, flags=re.IGNORECASE)

    for sep in [". ", "; ", " - ", " -- ", " because ", " which ", " who ", " that "]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()

    text = text.strip(" \t\n\r.,;:!?\"'()[]{}")
    words = text.split()
    if max_words > 0 and len(words) > max_words:
        text = " ".join(words[:max_words]).strip(" \t\n\r.,;:!?\"'()[]{}")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dream", choices=["dream", "llada"])
    parser.add_argument("--decode_backend", default="vanilla", choices=["vanilla", "fast-dllm"])
    parser.add_argument("--dataset", default="musique", choices=sorted(QUESTION_FILES))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--ablation_top_k", type=int, default=10)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline_5", "baseline_10", "dnmr_pool"],
        choices=["baseline_5", "baseline_10", "dnmr_pool"],
    )
    parser.add_argument("--fast_dllm_use_cache", action="store_true")
    parser.add_argument("--fast_dllm_dual_cache", action="store_true")
    parser.add_argument("--fast_dllm_block_size", type=int, default=32)
    parser.add_argument("--fast_dllm_threshold", type=float, default=None)
    parser.add_argument("--enforce_short_answer", action="store_true")
    parser.add_argument("--max_answer_words", type=int, default=6)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    if args.decode_backend == "fast-dllm" and args.model != "llada":
        raise ValueError("The Fast-dLLM decode backend is currently supported only for --model llada.")
    if args.fast_dllm_dual_cache and not args.fast_dllm_use_cache:
        raise ValueError("--fast_dllm_dual_cache requires --fast_dllm_use_cache.")

    t_start = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=== Budget Ablation ===", flush=True)
    print(
        f"Model: {args.model} | Decode backend: {args.decode_backend} | Dataset: {args.dataset} | "
        f"C0={args.initial_top_k} | Retrieval ablation={args.ablation_top_k} | "
        f"Expand={args.expand_top_k}",
        flush=True,
    )

    model, tokenizer = load_model_and_tokenizer(args.model, args.decode_backend)
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t_start:.1f}s", flush=True)

    decode_kwargs = {
        "decode_backend": args.decode_backend,
        "fast_dllm_use_cache": args.fast_dllm_use_cache,
        "fast_dllm_dual_cache": args.fast_dllm_dual_cache,
        "fast_dllm_block_size": args.fast_dllm_block_size,
        "fast_dllm_threshold": args.fast_dllm_threshold,
    }

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    max_top_k = max(args.initial_top_k, args.ablation_top_k)
    print(f"Batch initial retrieval (top-{max_top_k})...", flush=True)
    query_texts = [f"query: {q['question']}" for q in questions]
    all_retrieval = retriever.retrieve_batch(query_texts, max_top_k)
    print(f"Initial retrieval done in {time.time() - t_start:.1f}s", flush=True)

    methods = list(args.methods)
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        retrieved = all_retrieval[qi]
        initial_passages = retrieved[:args.initial_top_k]
        ablation_passages = retrieved[:args.ablation_top_k]
        c0_context = "\n\n".join(initial_passages)
        ablation_context = "\n\n".join(ablation_passages)

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext,
            "gold": gold,
            "passage_counts": {
                "baseline_5": len(initial_passages),
                "baseline_10": len(ablation_passages),
            },
        }

        method_answers = {}

        if "baseline_5" in methods or "dnmr_pool" in methods:
            baseline_5_answer = simple_decode(
                model,
                tokenizer,
                c0_context,
                qtext,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                **decode_kwargs,
            )
            if "baseline_5" in methods:
                method_answers["baseline_5"] = baseline_5_answer
            seed_answer = baseline_5_answer
        else:
            seed_answer = ""

        if "baseline_10" in methods:
            method_answers["baseline_10"] = simple_decode(
                model,
                tokenizer,
                ablation_context,
                qtext,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                **decode_kwargs,
            )

        if "dnmr_pool" in methods:
            bridge_cands = extract_candidates_generic(
                model,
                tokenizer,
                c0_context,
                qtext,
                args.n_candidates,
                extraction_steps=args.extraction_steps,
            )
            expanded_passages, queries_used, new_passages = expand_evidence(
                retriever,
                qtext,
                seed_answer,
                bridge_cands,
                initial_passages,
                args.expand_top_k,
            )
            expanded_context = "\n\n".join(expanded_passages)
            method_answers["dnmr_pool"] = simple_decode(
                model,
                tokenizer,
                expanded_context,
                qtext,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                **decode_kwargs,
            )
            row["dnmr_pool_stats"] = {
                "n_candidates": len(bridge_cands),
                "candidate_texts": [cand.get("text", "") for cand in bridge_cands],
                "retrieval_queries": queries_used,
                "n_new_passages": len(new_passages),
                "n_total_passages": len(expanded_passages),
            }

        elapsed = time.time() - tq
        row["elapsed"] = round(elapsed, 2)

        for method, answer in method_answers.items():
            if args.enforce_short_answer:
                answer = shorten_answer(answer, max_words=args.max_answer_words)
            f1_result = compute_f1(answer, gold)
            f1_value = f1_result if isinstance(f1_result, (int, float)) else f1_result[2]
            em = float(answer.strip().lower() == gold.strip().lower())
            contain = contain_metric(answer, gold)
            totals[method]["f1"] += f1_value
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {
                "answer": answer,
                "f1": round(f1_value, 4),
                "em": em,
                "contain": contain,
            }

        results.append(row)

        if (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1:
            print(f"[{qi + 1}/{len(questions)}] {row['id']} ({elapsed:.1f}s)", flush=True)
            for method in methods:
                answer = row[method]["answer"][:50] if method in row else ""
                print(
                    f"  {method:12s} {answer:50s} "
                    f"F1={row[method]['f1']:.3f} EM={row[method]['em']:.0f}",
                    flush=True,
                )

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {
                method: {metric: round(value / max(1, n_done), 4) for metric, value in totals[method].items()}
                for method in methods
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

    n_done = len(results)
    summary = {
        method: {metric: round(value / max(1, n_done), 4) for metric, value in totals[method].items()}
        for method in methods
    }

    print("\nMethod         F1      EM   Contain", flush=True)
    print("-----------------------------------", flush=True)
    for method in methods:
        stats = summary[method]
        print(
            f"{method:12s} {stats['f1']:.4f}  {stats['em']:.4f}  {stats['contain']:.4f}",
            flush=True,
        )
    if "baseline_10" in summary and "dnmr_pool" in summary:
        delta = summary["dnmr_pool"]["f1"] - summary["baseline_10"]["f1"]
        print(f"\nDNMR pool vs baseline_10 delta: {delta:+.4f} F1", flush=True)
    print(f"Saved to {args.output}", flush=True)
    print(f"Total elapsed: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
