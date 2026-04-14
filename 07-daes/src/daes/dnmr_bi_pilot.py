"""DNMR-BI pilot: Bridge-Informed DNMR.

Tests two orthogonal improvements over standard Pool decode:
  1. Bridge-conditioned prompt: feed bridge entities as reasoning cues
  2. Canvas-length optimization: n_tokens=16 vs 32

Methods:
  baseline       - decode from C0 (n=32)
  pool           - standard DNMR decode from C1 (n=32)
  pool_bi        - DNMR + bridge hint in prompt (n=32)
  pool_16        - DNMR with n_tokens=16
  pool_bi_16     - bridge hint + n_tokens=16
  pool_struct    - DNMR with structured context labeling (n=32)

Run: python -u src/daes/dnmr_bi_pilot.py --model dream --dataset musique --n_questions 50
"""
import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
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
    short_user_prompt,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


@torch.inference_mode()
def simple_decode(model, tokenizer, context, question, steps=32, n_tokens=32):
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
    return list(current_passages) + new_passages, new_passages


def clean_bridge_text(text):
    """Extract a clean entity name from a bridge candidate string."""
    import re
    text = re.sub(r"^(?:The answer is:?\s*|The\s+)", "", text, flags=re.IGNORECASE).strip()
    text = text.split("\n")[0].split(". ")[0].strip()
    words = text.split()
    if len(words) > 5:
        text = " ".join(words[:5])
    return text.strip().rstrip(",.")


def build_bridge_hint(bridge_cands):
    """Build a bridge hint string from candidates."""
    clean = []
    seen = set()
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        text = clean_bridge_text(text)
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            clean.append(text)
    if not clean:
        return ""
    return "Clue: " + ", ".join(clean[:3]) + "."


def build_structured_context(initial_passages, new_passages, bridge_cands):
    """Build a structured context that labels initial vs bridge-retrieved passages."""
    initial_ctx = "\n\n".join(initial_passages)
    if not new_passages:
        return initial_ctx
    bridge_names = []
    for cand in bridge_cands[:2]:
        text = clean_bridge_text(cand.get("text", ""))
        if text:
            bridge_names.append(text)
    label = ", ".join(bridge_names) if bridge_names else "related facts"
    bridge_ctx = "\n\n".join(new_passages)
    return f"{initial_ctx}\n\nAdditional evidence ({label}):\n{bridge_ctx}"


def score(pred, gold):
    precision, recall, f1 = compute_f1(pred, gold)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "em": float(pred.strip().lower() == gold.strip().lower()),
        "contain": float(gold.strip().lower() in pred.strip().lower()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== DNMR-BI Pilot ===", flush=True)
    print(f"Model: {args.model}, Dataset: {args.dataset}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

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

    methods = ["baseline", "pool", "pool_bi", "pool_16", "pool_bi_16", "pool_struct"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # Baseline: decode from C0
        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                     steps=args.steps, n_tokens=32)

        # Seed answer for bridge extraction
        seed_ans = baseline_ans

        # Bridge extraction
        bridge_cands = extract_candidates_generic(
            model, tokenizer, old_ctx, qtext,
            args.n_candidates, extraction_steps=args.extraction_steps
        )

        # Evidence expansion
        all_passages, new_passages = expand_evidence(
            retriever, qtext, seed_ans, bridge_cands,
            initial, args.expand_top_k
        )
        pool_ctx = "\n\n".join(all_passages)

        # Build variants
        bridge_hint = build_bridge_hint(bridge_cands)
        struct_ctx = build_structured_context(initial, new_passages, bridge_cands)

        if bridge_hint:
            bi_ctx = f"{bridge_hint}\n\n{pool_ctx}"
        else:
            bi_ctx = pool_ctx

        # pool: standard DNMR (n=32)
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext,
                                 steps=args.steps, n_tokens=32)

        # pool_bi: bridge hint (n=32)
        pool_bi_ans = simple_decode(model, tokenizer, bi_ctx, qtext,
                                    steps=args.steps, n_tokens=32)

        # pool_16: standard DNMR (n=16)
        pool_16_ans = simple_decode(model, tokenizer, pool_ctx, qtext,
                                    steps=args.steps, n_tokens=16)

        # pool_bi_16: bridge hint (n=16)
        pool_bi_16_ans = simple_decode(model, tokenizer, bi_ctx, qtext,
                                       steps=args.steps, n_tokens=16)

        # pool_struct: structured context (n=32)
        pool_struct_ans = simple_decode(model, tokenizer, struct_ctx, qtext,
                                        steps=args.steps, n_tokens=32)

        elapsed = time.time() - tq

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext,
            "gold": gold,
            "elapsed": round(elapsed, 2),
            "bridge_hint": bridge_hint,
            "bridge_cands": [{"text": c.get("text", ""), "conf": round(c.get("init_conf", 0), 4)} for c in bridge_cands],
            "n_passages": len(all_passages),
            "n_new_passages": len(new_passages),
        }

        answers = {
            "baseline": baseline_ans,
            "pool": pool_ans,
            "pool_bi": pool_bi_ans,
            "pool_16": pool_16_ans,
            "pool_bi_16": pool_bi_16_ans,
            "pool_struct": pool_struct_ans,
        }

        for method, ans in answers.items():
            s = score(ans, gold)
            s["answer"] = ans
            row[method] = s
            totals[method]["f1"] += s["f1"]
            totals[method]["em"] += s["em"]
            totals[method]["contain"] += s["contain"]

        results.append(row)

        if (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s) hint={bridge_hint[:50]}", flush=True)
            for m in methods:
                a = row[m]["answer"][:35]
                f = row[m]["f1"]
                print(f"  {m:16s} {a:35s} F1={f:.3f}", flush=True)

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()} for m in methods}
            with open(args.output, "w") as f:
                json.dump({"summary": summary, "results": results, "config": vars(args),
                           "timing": {"elapsed_sec": round(time.time() - t_start, 1)}}, f, indent=2)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\n{'Method':<16s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 40)
    for m in methods:
        s = summary[m]
        print(f"{m:<16s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")
    print(f"\nTotal: {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, n):.1f}s/q)")


if __name__ == "__main__":
    main()
