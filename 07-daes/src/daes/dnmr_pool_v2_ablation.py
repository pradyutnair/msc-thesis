"""DNMR Pool v2 Ablation: Short canvas + hint2 + short extraction + yes/no detection.

Tests on LLaDA to close the gap with iSPREAD/iARAM.

Methods:
  baseline          - decode from C0 (n_tokens=32)
  pool              - standard DNMR (n_mask=12, n_tokens=32, no hint)
  pool_8            - short canvas only (n_mask=12, n_tokens=8)
  pool_8_hint2      - short canvas + entity hint in context (n_mask=12, n_tokens=8)
  pool_m6_8_hint2   - shorter extraction + short canvas + hint (n_mask=6, n_tokens=8)
  pool_m6_8_hint2_yn - above + yes/no detection (n_tokens=2 for yes/no questions)

Run: python -u src/daes/dnmr_pool_v2_ablation.py --model llada --dataset musique --n_questions 10
"""
import argparse
import json
import math
import os
import re
import sys
import time
import string
from collections import Counter

import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import eamd_v2_wiki18
from eamd_v2_wiki18 import (
    QUESTION_FILES,
    Wiki18Retriever,
    _neg_entropy,
    build_short_prompt,
    decode_answer,
    extract_candidates_agnostic,
    get_mask_id,
    prepare_logits,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
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


def is_yesno_question(question):
    """Detect yes/no (comparison) questions."""
    q = question.strip().lower()
    return q.startswith(("is ", "are ", "was ", "were ", "did ", "do ", "does ",
                         "has ", "have ", "had ", "can ", "could ", "will ", "would ",
                         "should "))


def build_hint_v2(bridge_cands):
    """Build clean entity hint string from candidates."""
    entities = []
    seen = set()
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        # Strip "The answer is:" and similar preambles
        text = re.sub(r"^(?:The answer is:?\s*|The\s+)", "", text, flags=re.IGNORECASE).strip()
        text = text.split("\n")[0].strip().rstrip(".,;:")
        # Truncate very long candidates to 5 words
        words = text.split()
        if len(words) > 5:
            text = " ".join(words[:5])
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            entities.append(text)
    if not entities:
        return ""
    return "Related entities: " + ", ".join(entities[:3]) + "."


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Evidence expansion (standard — same as idnmr_pilot.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== DNMR Pool v2 Ablation ===", flush=True)
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

    methods = ["baseline", "pool", "pool_8", "pool_8_hint2", "pool_m6_8_hint2", "pool_m6_8_hint2_yn"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)
        yn = is_yesno_question(qtext)

        # Shared: seed decode (32 tokens, used for retrieval query)
        seed_ans = simple_decode(model, tokenizer, old_ctx, qtext, steps=args.steps, n_tokens=32)

        # === Extract candidates with n_mask=12 (standard) ===
        cands_12 = extract_candidates_agnostic(
            model, tokenizer, old_ctx, qtext, args.n_candidates,
            n_mask=12, extraction_steps=args.extraction_steps
        )

        # === Extract candidates with n_mask=6 (shorter) ===
        cands_6 = extract_candidates_agnostic(
            model, tokenizer, old_ctx, qtext, args.n_candidates,
            n_mask=6, extraction_steps=args.extraction_steps
        )

        # === Expand evidence (standard, using n_mask=12 candidates) ===
        pool_passages_12, pool_new_12 = expand_evidence(
            retriever, qtext, seed_ans, cands_12, initial, args.expand_top_k
        )
        pool_ctx_12 = "\n\n".join(pool_passages_12)

        # === Expand evidence (using n_mask=6 candidates) ===
        pool_passages_6, pool_new_6 = expand_evidence(
            retriever, qtext, seed_ans, cands_6, initial, args.expand_top_k
        )
        pool_ctx_6 = "\n\n".join(pool_passages_6)

        # Hints
        hint_12 = build_hint_v2(cands_12)
        hint_6 = build_hint_v2(cands_6)

        # Hint contexts
        hint_ctx_12 = f"{hint_12}\n\n{pool_ctx_12}" if hint_12 else pool_ctx_12
        hint_ctx_6 = f"{hint_6}\n\n{pool_ctx_6}" if hint_6 else pool_ctx_6

        # === baseline: decode from C0, 32 tokens ===
        baseline_ans = seed_ans

        # === pool: standard DNMR (n_mask=12, n_tokens=32, no hint) ===
        pool_ans = simple_decode(model, tokenizer, pool_ctx_12, qtext,
                                 steps=args.steps, n_tokens=32)

        # === pool_8: short canvas only (n_mask=12, n_tokens=8) ===
        pool_8_ans = simple_decode(model, tokenizer, pool_ctx_12, qtext,
                                   steps=args.steps, n_tokens=8)

        # === pool_8_hint2: short canvas + hint in context (n_mask=12, n_tokens=8) ===
        pool_8_hint2_ans = simple_decode(model, tokenizer, hint_ctx_12, qtext,
                                         steps=args.steps, n_tokens=8)

        # === pool_m6_8_hint2: n_mask=6 extraction + short canvas + hint (n_tokens=8) ===
        pool_m6_8_hint2_ans = simple_decode(model, tokenizer, hint_ctx_6, qtext,
                                            steps=args.steps, n_tokens=8)

        # === pool_m6_8_hint2_yn: above + yes/no detection ===
        yn_tokens = 2 if yn else 8
        pool_m6_8_hint2_yn_ans = simple_decode(model, tokenizer, hint_ctx_6, qtext,
                                               steps=args.steps, n_tokens=yn_tokens)

        elapsed = time.time() - tq

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext, "gold": gold, "elapsed": round(elapsed, 2),
            "is_yesno": yn,
            "cands_12": [c.get("text", "")[:50] for c in cands_12],
            "cands_6": [c.get("text", "")[:50] for c in cands_6],
            "hint_12": hint_12, "hint_6": hint_6,
            "new_passages_12": len(pool_new_12),
            "new_passages_6": len(pool_new_6),
            "n_passages_12": len(pool_passages_12),
            "n_passages_6": len(pool_passages_6),
        }

        answers = {
            "baseline": baseline_ans,
            "pool": pool_ans,
            "pool_8": pool_8_ans,
            "pool_8_hint2": pool_8_hint2_ans,
            "pool_m6_8_hint2": pool_m6_8_hint2_ans,
            "pool_m6_8_hint2_yn": pool_m6_8_hint2_yn_ans,
        }

        for method, ans in answers.items():
            s = score(ans, gold)
            s["answer"] = ans
            row[method] = s
            totals[method]["f1"] += s["f1"]
            totals[method]["em"] += s["em"]
            totals[method]["contain"] += s["contain"]

        results.append(row)

        print(f"\n[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s) {'[Y/N]' if yn else ''}", flush=True)
        print(f"  Gold: {gold}", flush=True)
        print(f"  Cands12: {row['cands_12']}", flush=True)
        print(f"  Cands6:  {row['cands_6']}", flush=True)
        print(f"  Hint6:   {hint_6}", flush=True)
        print(f"  Passages: 12mask={len(pool_passages_12)} 6mask={len(pool_passages_6)}", flush=True)
        for m in methods:
            a = row[m]["answer"][:45]
            f = row[m]["f1"]
            c = "Y" if row[m]["contain"] else "."
            print(f"  {m:22s} {a:45s} F1={f:.3f} C={c}", flush=True)

        # Save incrementally
        if (qi + 1) % 5 == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()}
                       for m in methods}
            with open(args.output, "w") as f:
                json.dump({"summary": summary, "results": results, "config": vars(args),
                           "timing": {"elapsed_sec": round(time.time() - t_start, 1)}}, f, indent=2)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\n{'Method':<24s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 48)
    for m in methods:
        s = summary[m]
        print(f"{m:<24s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")
    print(f"\nTotal: {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, n):.1f}s/q)")


if __name__ == "__main__":
    main()
