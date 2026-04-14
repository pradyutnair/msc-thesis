"""DNMR-BI v2: Tests posterior-ordered evidence + two-phase decode.

Methods:
  baseline        - decode from C0
  pool            - standard DNMR
  pool_poep       - posterior-ordered evidence placement
  pool_hint2      - improved bridge hint (clean entity names only)
  pool_poep_hint  - POEP + improved hint
  pool_2phase     - two-phase decode: commit reasoning before answer

Run: python -u src/daes/dnmr_bi_pilot_v2.py --model dream --dataset musique --n_questions 50
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


STRIP_VERBS = {
    "promote", "find", "search", "identify", "locate", "determine",
    "contribute", "provide", "support", "create", "build", "make",
    "answer", "show", "describe", "explain", "tell", "give",
}


def clean_entity(text):
    """Aggressively extract clean entity name from bridge candidate."""
    text = re.sub(r"^(?:The answer is:?\s*|The\s+|A\s+|An\s+|To\s+|In\s+)", "", text, flags=re.IGNORECASE).strip()
    text = text.split("\n")[0].split(". ")[0].strip()
    words = text.split()
    if words and words[0].lower().rstrip("s") in STRIP_VERBS:
        words = words[1:]
    if len(words) > 3:
        words = words[:3]
    result = " ".join(words).strip().rstrip(",.")
    return result if len(result) > 1 else ""


def build_hint_v2(bridge_cands):
    """Build a clean entity-only hint."""
    entities = []
    seen = set()
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        entity = clean_entity(text)
        if entity and entity.lower() not in seen:
            seen.add(entity.lower())
            entities.append(entity)
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


@torch.inference_mode()
def two_phase_decode(model, tokenizer, context, question,
                     n_reason=16, n_answer=16, reason_steps=16, answer_steps=16):
    """Two-phase decode: commit reasoning tokens before answer tokens.

    Phase 1: denoise only positions [0, n_reason) — captures intermediate facts.
    Phase 2: denoise only positions [n_reason, n_reason+n_answer) — generates answer
             conditioned on committed reasoning tokens.

    Total forward passes = reason_steps + answer_steps (same as single-phase with
    steps = reason_steps + answer_steps).
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    n_total = n_reason + n_answer
    canvas = prefix_ids + [mask_id] * n_total
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    # Phase 1: denoise reasoning region only
    remaining = n_reason
    k_per_step = max(1, math.ceil(n_reason / reason_steps))
    for step in range(reason_steps):
        if remaining <= 0:
            break
        reason_mask = (x[0, n_prefix:n_prefix + n_reason] == mask_id).nonzero(as_tuple=True)[0]
        if len(reason_mask) == 0:
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, reason_mask + n_prefix]
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == reason_steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, reason_mask[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

    # Phase 2: denoise answer region only (reasoning is now committed)
    ans_offset = n_prefix + n_reason
    remaining = n_answer
    k_per_step = max(1, math.ceil(n_answer / answer_steps))
    for step in range(answer_steps):
        if remaining <= 0:
            break
        answer_mask = (x[0, ans_offset:ans_offset + n_answer] == mask_id).nonzero(as_tuple=True)[0]
        if len(answer_mask) == 0:
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, answer_mask + ans_offset]
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == answer_steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, answer_mask[topk] + ans_offset] = x0[topk]
        remaining -= len(topk)

    reasoning = decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_reason])
    answer = decode_answer(tokenizer, x[0, ans_offset:ans_offset + n_answer])
    return answer, reasoning


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


def expand_evidence_poep(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
    """Posterior-Ordered Evidence Placement: reverse bridge order so
    highest-posterior bridge passages end up LAST (closest to answer mask).

    Exploits attention position bias: passages near the end of context
    receive higher attention from the answer mask positions.
    """
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
    print(f"=== DNMR-BI v2 Pilot ===", flush=True)

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

    methods = ["baseline", "pool", "pool_poep", "pool_hint2", "pool_poep_hint", "pool_2phase"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext, steps=args.steps, n_tokens=32)
        seed_ans = baseline_ans

        bridge_cands = extract_candidates_generic(
            model, tokenizer, old_ctx, qtext,
            args.n_candidates, extraction_steps=args.extraction_steps
        )

        # Standard pool evidence
        pool_passages, pool_new = expand_evidence(
            retriever, qtext, seed_ans, bridge_cands, initial, args.expand_top_k
        )
        pool_ctx = "\n\n".join(pool_passages)

        # POEP evidence (reversed bridge order)
        poep_passages, poep_new = expand_evidence_poep(
            retriever, qtext, seed_ans, bridge_cands, initial, args.expand_top_k
        )
        poep_ctx = "\n\n".join(poep_passages)

        # Improved bridge hint
        hint = build_hint_v2(bridge_cands)
        hint_ctx = f"{hint}\n\n{pool_ctx}" if hint else pool_ctx
        poep_hint_ctx = f"{hint}\n\n{poep_ctx}" if hint else poep_ctx

        # pool: standard decode
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext, steps=args.steps, n_tokens=32)
        # pool_poep: posterior-ordered evidence
        pool_poep_ans = simple_decode(model, tokenizer, poep_ctx, qtext, steps=args.steps, n_tokens=32)
        # pool_hint2: improved hint only
        pool_hint2_ans = simple_decode(model, tokenizer, hint_ctx, qtext, steps=args.steps, n_tokens=32)
        # pool_poep_hint: combined
        pool_poep_hint_ans = simple_decode(model, tokenizer, poep_hint_ctx, qtext, steps=args.steps, n_tokens=32)
        # pool_2phase: two-phase decode from pool context
        pool_2phase_ans, pool_2phase_reasoning = two_phase_decode(
            model, tokenizer, pool_ctx, qtext,
            n_reason=16, n_answer=16, reason_steps=16, answer_steps=16
        )

        elapsed = time.time() - tq
        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext, "gold": gold,
            "elapsed": round(elapsed, 2),
            "hint": hint,
            "bridge_cands": [c.get("text", "")[:40] for c in bridge_cands],
        }

        answers = {
            "baseline": baseline_ans,
            "pool": pool_ans,
            "pool_poep": pool_poep_ans,
            "pool_hint2": pool_hint2_ans,
            "pool_poep_hint": pool_poep_hint_ans,
            "pool_2phase": pool_2phase_ans,
        }

        for method, ans in answers.items():
            s = score(ans, gold)
            s["answer"] = ans
            if method == "pool_2phase":
                s["reasoning"] = pool_2phase_reasoning
            row[method] = s
            totals[method]["f1"] += s["f1"]
            totals[method]["em"] += s["em"]
            totals[method]["contain"] += s["contain"]

        results.append(row)

        if (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s) hint={hint[:50]}", flush=True)
            for m in methods:
                a = row[m]["answer"][:35]
                f = row[m]["f1"]
                extra = ""
                if m == "pool_2phase":
                    extra = f" reason={pool_2phase_reasoning[:25]}"
                print(f"  {m:16s} {a:35s} F1={f:.3f}{extra}", flush=True)

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
