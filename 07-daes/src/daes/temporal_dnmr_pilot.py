"""Temporal DNMR: Extract bridge candidates at intermediate denoising step τ.
Key idea: LLaDA's distribution is peaked at step 0 but diversifies at step ~4
after some tokens commit and create context for the rest.
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy, Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18


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
def temporal_extract(model, tokenizer, context, question, tau=4, n_candidates=3,
                     n_branch=3, n_mask=12, total_steps=32, completion_steps=12):
    """Extract bridge candidates from denoising step τ instead of step 0.

    1. Start fully masked canvas
    2. Denoise for τ steps (committing tokens along the way)
    3. At step τ, read distribution at STILL-MASKED positions
    4. Branch on top-k tokens at highest-entropy masked positions
    5. Denoise each branch to completion
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    neg_ent = _neg_entropy()

    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k_per_step = max(1, math.ceil(n_mask / total_steps))
    remaining = n_mask

    # Phase 1: Denoise for τ steps
    for step in range(tau):
        if remaining <= 0:
            break
        masked_local = (x[0, n_prefix:n_prefix + n_mask] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, masked_local + n_prefix]
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=neg_ent)
        n_commit = min(k_per_step, remaining)
        if step == tau - 1:
            # Don't commit everything at last pre-extraction step
            n_commit = min(n_commit, max(1, remaining - 4))
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

    # Phase 2: Read distribution at still-masked positions
    masked_local = (x[0, n_prefix:n_prefix + n_mask] == mask_id).nonzero(as_tuple=True)[0]
    if len(masked_local) == 0:
        # All committed, fall back to using committed answer as single candidate
        ans_text = decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_mask])
        return [{"text": ans_text, "position": -1, "init_conf": 1.0}]

    out = model(x, attention_mask=attn)
    logits = prepare_logits(out.logits)

    # Find highest-entropy masked positions
    answer_logits = logits[0, n_prefix:n_prefix + n_mask]
    probs = torch.softmax(answer_logits / 0.3, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    # Only consider still-masked positions
    masked_entropy = torch.zeros(n_mask, device=device) - 1
    masked_entropy[masked_local] = entropy[masked_local].float()
    top_positions = torch.topk(masked_entropy, min(3, len(masked_local))).indices.tolist()

    # Phase 3: Branch on top-k tokens at selected positions
    candidates = []
    seen = set()

    for pos_local in top_positions:
        pos_global = n_prefix + pos_local
        pos_probs = torch.softmax(answer_logits[pos_local] / 0.3, dim=-1)
        top_probs, top_ids = torch.topk(pos_probs, n_branch)

        for i in range(len(top_probs)):
            x_c = x.clone()
            x_c[0, pos_global] = top_ids[i].item()

            # Phase 4: Denoise branch to completion
            rem = (x_c[0, n_prefix:n_prefix + n_mask] == mask_id).sum().item()
            for step in range(completion_steps):
                if rem <= 0:
                    break
                mi = (x_c[0] == mask_id)
                if not mi.any():
                    break
                o2 = model(x_c, attention_mask=attn)
                l2 = prepare_logits(o2.logits)
                mp = mi.nonzero(as_tuple=True)[0]
                c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=neg_ent)
                k = min(max(1, rem // completion_steps), rem)
                if step == completion_steps - 1:
                    k = rem
                _, tk = torch.topk(c2, min(k, len(c2)))
                x_c[0, mp[tk]] = x02[tk]
                rem -= len(tk)

            cand_text = decode_answer(tokenizer, x_c[0, n_prefix:n_prefix + n_mask])
            cand_text = cand_text.split("\n")[0].split(". ")[0].strip()

            if cand_text and len(cand_text) > 1 and cand_text.lower() not in seen:
                seen.add(cand_text.lower())
                candidates.append({
                    "text": cand_text,
                    "position": pos_local,
                    "init_conf": round(top_probs[i].item(), 4),
                })
                if len(candidates) >= n_candidates:
                    break
        if len(candidates) >= n_candidates:
            break

    return candidates


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--tau", type=int, default=4, help="Denoising step at which to extract")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== Temporal DNMR (tau={args.tau}) ===", flush=True)

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
    print(f"  Loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    questions = json.load(open(QUESTION_FILES[args.dataset]))[:args.n_questions]
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, 5)
    print(f"  Retrieved in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "temporal_dnmr"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # Baseline
        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens)

        # Temporal DNMR
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                        steps=16, n_tokens=16, temperature=0.0)
        cands = temporal_extract(model, tokenizer, old_ctx, qtext, tau=args.tau)
        expanded, new_p = expand_evidence(retriever, qtext, seed_ans, cands, initial)
        new_ctx = "\n\n".join(expanded)
        tdnmr_ans = simple_decode(model, tokenizer, new_ctx, qtext,
                                  steps=args.steps, n_tokens=args.answer_tokens)

        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold,
               "candidates": [c.get("text", "")[:40] for c in cands],
               "n_new_passages": len(new_p)}

        for method, ans in [("baseline", baseline_ans), ("temporal_dnmr", tdnmr_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        results.append(row)
        elapsed = time.time() - tq
        print(f"[{qi+1}/{len(questions)}] ({elapsed:.1f}s) cands={[c['text'][:20] for c in cands]} new={len(new_p)}", flush=True)
        for m in methods:
            print(f"  {m:16s} {row[m]['answer'][:50]:50s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\nSummary")
    for m in methods:
        print(f"  {m:16s} F1={summary[m]['f1']:.4f} EM={summary[m]['em']:.4f} contain={summary[m]['contain']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args)}, f, indent=2)
    print(f"Saved to {args.output}")
    print(f"Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
