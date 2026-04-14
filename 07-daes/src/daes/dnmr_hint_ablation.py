"""DNMR Hint Ablation: Test hint placement strategies.

Standard DNMR extraction + hint as soft signal in different positions:
  baseline          - decode from C0
  pool              - standard DNMR (no hint)
  pool_hint_decode  - hint prepended to decode context only
  pool_hint_retr    - hint used in retrieval queries only
  pool_hint_both    - hint in both retrieval queries AND decode context
  pool_hint_poep    - hint + POEP (posterior-ordered evidence)

Run: python -u src/daes/dnmr_hint_ablation.py --model llada --dataset musique --n_questions 50
"""
import argparse
import json
import math
import os
import re
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
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import string


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def score(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return {"f1": 0, "precision": 0, "recall": 0, "em": 0, "contain": 0}
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return {"f1": 0, "precision": 0, "recall": 0, "em": 0, "contain": 0}
    p = overlap / len(pt)
    r = overlap / len(gt)
    f1 = 2 * p * r / (p + r)
    em = float(normalize_answer(pred) == normalize_answer(gold))
    contain = float(gold.strip().lower() in pred.strip().lower())
    return {"f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4),
            "em": em, "contain": contain}


def build_hint(bridge_cands):
    """Build entity hint string from candidates."""
    entities = []
    seen = set()
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        # Light cleaning: strip "The answer is:" prefix
        text = re.sub(r"^(?:The answer is:?\s*)", "", text, flags=re.IGNORECASE).strip()
        text = text.split("\n")[0].strip().rstrip(".,;:")
        # Truncate very long candidates
        words = text.split()
        if len(words) > 5:
            text = " ".join(words[:5])
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            entities.append(text)
    if not entities:
        return ""
    return "Related entities: " + ", ".join(entities[:3]) + "."


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


def expand_evidence_standard(retriever, question, seed_answer, bridge_cands,
                             current_passages, expand_top_k=3):
    """Standard: question + each candidate."""
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


def expand_evidence_with_hint(retriever, question, seed_answer, bridge_cands,
                              hint_str, current_passages, expand_top_k=3):
    """Hint in retrieval: use hint string as an additional retrieval query."""
    queries = [f"query: {question} {seed_answer}"]
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if text and len(text) > 1:
            queries.append(f"query: {question} {text}")
    # Add hint as extra query
    if hint_str:
        queries.append(f"query: {question} {hint_str}")
    results = retriever.retrieve_batch(queries, expand_top_k)
    existing = set(current_passages)
    new_passages = []
    for result_list in results:
        for passage in result_list:
            if passage not in existing:
                new_passages.append(passage)
                existing.add(passage)
    return list(current_passages) + new_passages, new_passages


def expand_evidence_poep(retriever, question, seed_answer, bridge_cands,
                         current_passages, expand_top_k=3):
    """POEP: reverse bridge order so best-posterior passages are last (near answer)."""
    reversed_cands = list(reversed(bridge_cands))
    queries = [f"query: {question} {seed_answer}"]
    for cand in reversed_cands:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== DNMR Hint Ablation ===", flush=True)
    print(f"Model: {args.model}, Dataset: {args.dataset}, N: {args.n_questions}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
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
    print(f"Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool", "pool_hint_decode", "pool_hint_retr",
               "pool_hint_both", "pool_hint_poep"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # Shared: seed decode + candidate extraction
        seed_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        bridge_cands = extract_candidates_generic(
            model, tokenizer, old_ctx, qtext, args.n_candidates,
            extraction_steps=args.extraction_steps
        )

        hint = build_hint(bridge_cands)

        # === baseline: decode from C0 ===
        baseline_ans = seed_ans

        # === pool: standard DNMR (no hint) ===
        pool_passages, pool_new = expand_evidence_standard(
            retriever, qtext, seed_ans, bridge_cands, initial, args.expand_top_k
        )
        pool_ctx = "\n\n".join(pool_passages)
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        # === pool_hint_decode: hint in decode context only ===
        hint_decode_ctx = f"{hint}\n\n{pool_ctx}" if hint else pool_ctx
        hint_decode_ans = simple_decode(model, tokenizer, hint_decode_ctx, qtext,
                                       steps=args.steps, n_tokens=args.answer_tokens)

        # === pool_hint_retr: hint in retrieval queries only ===
        hint_retr_passages, hint_retr_new = expand_evidence_with_hint(
            retriever, qtext, seed_ans, bridge_cands, hint, initial, args.expand_top_k
        )
        hint_retr_ctx = "\n\n".join(hint_retr_passages)
        hint_retr_ans = simple_decode(model, tokenizer, hint_retr_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)

        # === pool_hint_both: hint in retrieval AND decode ===
        hint_both_ctx = f"{hint}\n\n{hint_retr_ctx}" if hint else hint_retr_ctx
        hint_both_ans = simple_decode(model, tokenizer, hint_both_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)

        # === pool_hint_poep: hint + POEP ===
        poep_passages, poep_new = expand_evidence_poep(
            retriever, qtext, seed_ans, bridge_cands, initial, args.expand_top_k
        )
        poep_hint_ctx = f"{hint}\n\n" + "\n\n".join(poep_passages) if hint else "\n\n".join(poep_passages)
        poep_hint_ans = simple_decode(model, tokenizer, poep_hint_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)

        elapsed = time.time() - tq

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext, "gold": gold, "elapsed": round(elapsed, 2),
            "candidates": [c.get("text", "")[:50] for c in bridge_cands],
            "hint": hint,
            "new_passages_std": len(pool_new),
            "new_passages_hint_retr": len(hint_retr_new),
            "new_passages_poep": len(poep_new),
        }

        answers = {
            "baseline": baseline_ans,
            "pool": pool_ans,
            "pool_hint_decode": hint_decode_ans,
            "pool_hint_retr": hint_retr_ans,
            "pool_hint_both": hint_both_ans,
            "pool_hint_poep": poep_hint_ans,
        }

        for method, ans in answers.items():
            s = score(ans, gold)
            s["answer"] = ans
            row[method] = s
            totals[method]["f1"] += s["f1"]
            totals[method]["em"] += s["em"]
            totals[method]["contain"] += s["contain"]

        results.append(row)

        if (qi + 1) % args.log_every == 0:
            print(f"\n[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s)", flush=True)
            print(f"  Gold: {gold}", flush=True)
            print(f"  Cands: {row['candidates']}", flush=True)
            print(f"  Hint: {hint}", flush=True)
            print(f"  New passages: std={len(pool_new)} hint_retr={len(hint_retr_new)} poep={len(poep_new)}", flush=True)
            for m in methods:
                a = row[m]["answer"][:40]
                f = row[m]["f1"]
                c = "Y" if row[m]["contain"] else "."
                print(f"  {m:20s} {a:40s} F1={f:.3f} C={c}", flush=True)

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()}
                       for m in methods}
            with open(args.output, "w") as f:
                json.dump({"summary": summary, "results": results, "config": vars(args),
                           "timing": {"elapsed_sec": round(time.time() - t_start, 1)}}, f, indent=2)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\n{'Method':<20s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 43)
    for m in methods:
        s = summary[m]
        print(f"{m:<20s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")
    print(f"\nTotal: {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, n):.1f}s/q)")


if __name__ == "__main__":
    main()
