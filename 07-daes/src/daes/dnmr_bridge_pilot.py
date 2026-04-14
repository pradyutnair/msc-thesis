"""DNMR Bridge Pilot: Bridge-prompt extraction + POEP + Hint for LLaDA.

Key idea: Instead of extracting from "Short Answer: [MASK]" (peaked posterior),
extract from "Key intermediate entity: [MASK x 6]" (hopefully more diverse).
Combined with posterior-ordered evidence placement and entity hints.

Methods:
  baseline          - decode from C0
  pool              - standard DNMR (current extraction, 12 masks)
  pool_bridge6      - bridge-prompt extraction, 6 masks
  pool_bridge6_poep - + posterior-ordered evidence placement
  pool_bridge6_hint - + entity hint prepended
  pool_bridge6_full - POEP + hint combined

Run: python -u src/daes/dnmr_bridge_pilot.py --model llada --dataset musique --n_questions 10
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
    short_user_prompt,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# Bridge-prompt extraction (NEW)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_bridges_prompt(model, tokenizer, context, question,
                           n_candidates=3, n_positions=3, n_branch=2,
                           n_mask=6, extraction_steps=6):
    """Extract bridge entity candidates using a bridge-specific prompt.

    Instead of "Answer: [MASK x 12]", uses:
    "Key intermediate entity needed: [MASK x 6]"

    Shorter mask region (6 tokens) forces entity-level output.
    Bridge-specific prompt encourages intermediate facts, not final answers.
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)

    # Bridge-specific prompt
    bridge_prompt = (
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Before answering, identify the key intermediate entity: "
    )
    messages = [{"role": "user", "content": bridge_prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    # Step 1: Forward pass on fully masked canvas
    out = model(x, attention_mask=attn)
    logits = prepare_logits(out.logits)

    # Step 2: Select positions by entropy
    answer_logits = logits[0, n_prefix:n_prefix + n_mask]
    probs = torch.softmax(answer_logits / 0.3, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    entropy_positions = torch.topk(entropy, min(n_positions, n_mask)).indices.tolist()
    top_positions = [0]
    for p in entropy_positions:
        if p not in top_positions:
            top_positions.append(p)
        if len(top_positions) >= n_positions + 1:
            break

    # Step 3: Build branch canvases
    branch_canvases = []
    branch_meta = []
    for pos_local in top_positions:
        pos_global = n_prefix + pos_local
        pos_probs = torch.softmax(answer_logits[pos_local] / 0.3, dim=-1)
        top_probs, top_ids = torch.topk(pos_probs, n_branch)
        for i in range(len(top_probs)):
            c = list(canvas)
            c[pos_global] = top_ids[i].item()
            branch_canvases.append(c)
            branch_meta.append((pos_local, top_probs[i].item(), top_ids[i].item()))

    if not branch_canvases:
        return []

    # Step 4: Denoise each branch
    x_all = torch.tensor(branch_canvases, dtype=torch.long, device=device)
    neg_ent = _neg_entropy()

    for bi in range(len(branch_canvases)):
        x_c = x_all[bi:bi+1]
        remaining = n_mask - 1
        for step in range(extraction_steps):
            if remaining <= 0:
                break
            mi = (x_c[0] == mask_id)
            if not mi.any():
                break
            out = model(x_c, attention_mask=attn)
            l2 = prepare_logits(out.logits)
            mp = mi.nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=neg_ent)
            k = min(max(1, remaining // extraction_steps), remaining)
            if step == extraction_steps - 1:
                k = remaining
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            remaining -= len(tk)

    # Step 5: Decode and deduplicate
    candidates = []
    seen = set()
    for bi in range(len(branch_meta)):
        pos_local, prob, tid = branch_meta[bi]
        cand_text = tokenizer.decode(
            x_all[bi, n_prefix:n_prefix + n_mask].tolist(),
            skip_special_tokens=True
        ).strip()
        # Minimal cleaning — just trim obvious junk
        cand_text = cand_text.split("\n")[0].strip().rstrip(".,;:")
        # Strip "The answer is" if it snuck in
        cand_text = re.sub(
            r"^(?:The answer is:?\s*|The\s+key\s+(?:intermediate\s+)?entity\s+(?:is|needed)?\s*:?\s*)",
            "", cand_text, flags=re.IGNORECASE
        ).strip()

        if cand_text and len(cand_text) > 1 and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append({
                "text": cand_text, "init_conf": prob, "position": pos_local
            })
            if len(candidates) >= n_candidates:
                break

    return candidates


# ---------------------------------------------------------------------------
# Simple decode (unchanged)
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
# Evidence expansion variants
# ---------------------------------------------------------------------------
def expand_evidence(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
    """Standard expansion: question + each candidate."""
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
    """POEP: reverse bridge order so highest-posterior passages end up last
    (closest to answer mask in the context)."""
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


def build_hint(bridge_cands):
    """Build entity hint from bridge candidates."""
    entities = []
    seen = set()
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        # Light cleaning only
        text = text.strip().rstrip(".,;:")
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            entities.append(text)
    if not entities:
        return ""
    return "Related entities: " + ", ".join(entities[:3]) + "."


def score(pred, gold):
    import re, string
    from collections import Counter
    def normalize(s):
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(c for c in s if c not in string.punctuation)
        return " ".join(s.split())
    pt = normalize(pred).split()
    gt = normalize(gold).split()
    if not pt or not gt:
        return {"f1": 0, "precision": 0, "recall": 0, "em": 0, "contain": 0}
    common = Counter(pt) & Counter(gt)
    overlap = sum(common.values())
    if overlap == 0:
        return {"f1": 0, "precision": 0, "recall": 0, "em": 0, "contain": 0}
    p = overlap / len(pt)
    r = overlap / len(gt)
    f1 = 2 * p * r / (p + r)
    em = float(normalize(pred) == normalize(gold))
    contain = float(gold.strip().lower() in pred.strip().lower())
    return {"f1": round(f1, 4), "precision": round(p, 4), "recall": round(r, 4),
            "em": em, "contain": contain}


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
    print(f"=== DNMR Bridge Pilot ===", flush=True)
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

    methods = ["baseline", "pool", "pool_bridge6", "pool_bridge6_poep",
               "pool_bridge6_hint", "pool_bridge6_full"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # Shared seed decode
        seed_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        # === baseline ===
        baseline_ans = seed_ans

        # === pool: standard DNMR extraction (12 masks) ===
        std_cands = extract_candidates_generic(
            model, tokenizer, old_ctx, qtext, args.n_candidates,
            extraction_steps=args.extraction_steps
        )
        pool_passages, pool_new = expand_evidence(
            retriever, qtext, seed_ans, std_cands, initial, args.expand_top_k
        )
        pool_ans = simple_decode(
            model, tokenizer, "\n\n".join(pool_passages), qtext,
            steps=args.steps, n_tokens=args.answer_tokens
        )

        # === bridge6: bridge-prompt extraction (6 masks) ===
        bridge_cands = extract_bridges_prompt(
            model, tokenizer, old_ctx, qtext,
            n_candidates=args.n_candidates, n_mask=6, extraction_steps=6
        )

        # pool_bridge6: standard evidence placement
        b6_passages, b6_new = expand_evidence(
            retriever, qtext, seed_ans, bridge_cands, initial, args.expand_top_k
        )
        b6_ctx = "\n\n".join(b6_passages)
        b6_ans = simple_decode(model, tokenizer, b6_ctx, qtext,
                               steps=args.steps, n_tokens=args.answer_tokens)

        # pool_bridge6_poep: posterior-ordered evidence
        b6_poep_passages, _ = expand_evidence_poep(
            retriever, qtext, seed_ans, bridge_cands, initial, args.expand_top_k
        )
        b6_poep_ctx = "\n\n".join(b6_poep_passages)
        b6_poep_ans = simple_decode(model, tokenizer, b6_poep_ctx, qtext,
                                    steps=args.steps, n_tokens=args.answer_tokens)

        # pool_bridge6_hint: entity hint prepended
        hint = build_hint(bridge_cands)
        b6_hint_ctx = f"{hint}\n\n{b6_ctx}" if hint else b6_ctx
        b6_hint_ans = simple_decode(model, tokenizer, b6_hint_ctx, qtext,
                                    steps=args.steps, n_tokens=args.answer_tokens)

        # pool_bridge6_full: POEP + hint
        b6_full_ctx = f"{hint}\n\n{b6_poep_ctx}" if hint else b6_poep_ctx
        b6_full_ans = simple_decode(model, tokenizer, b6_full_ctx, qtext,
                                    steps=args.steps, n_tokens=args.answer_tokens)

        elapsed = time.time() - tq

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext, "gold": gold, "elapsed": round(elapsed, 2),
            "std_candidates": [c.get("text", "")[:50] for c in std_cands],
            "bridge_candidates": [c.get("text", "")[:50] for c in bridge_cands],
            "hint": hint,
            "new_passages_std": len(pool_new),
            "new_passages_bridge": len(b6_new),
        }

        answers = {
            "baseline": baseline_ans,
            "pool": pool_ans,
            "pool_bridge6": b6_ans,
            "pool_bridge6_poep": b6_poep_ans,
            "pool_bridge6_hint": b6_hint_ans,
            "pool_bridge6_full": b6_full_ans,
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
            print(f"  Std cands:    {row['std_candidates']}", flush=True)
            print(f"  Bridge cands: {row['bridge_candidates']}", flush=True)
            print(f"  Hint: {hint}", flush=True)
            for m in methods:
                a = row[m]["answer"][:40]
                f = row[m]["f1"]
                c = "Y" if row[m]["contain"] else "."
                print(f"  {m:22s} {a:40s} F1={f:.3f} C={c}", flush=True)

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()}
                       for m in methods}
            with open(args.output, "w") as f:
                json.dump({"summary": summary, "results": results, "config": vars(args),
                           "timing": {"elapsed_sec": round(time.time() - t_start, 1)}}, f, indent=2)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\n{'Method':<22s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 45)
    for m in methods:
        s = summary[m]
        print(f"{m:<22s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")
    print(f"\nTotal: {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, n):.1f}s/q)")


if __name__ == "__main__":
    main()
