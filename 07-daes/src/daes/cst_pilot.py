"""CST v2: Counterfactual Score Transport — fixed per Codex review.
Fix 1: Per-passage additive contrast: logits(C0 ∪ {d_j}) - logits(C0), not replacement
Fix 2: Compute delta on MASKED state, not committed tokens
Fix 3: Use SKL for position ranking instead of L2 norm
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
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens]), x[0, n_prefix:n_prefix + n_tokens].clone()


@torch.inference_mode()
def cst_decode(model, tokenizer, context_c0, new_passages, question,
               committed_tokens, steps=32, n_tokens=32, beta=0.5, remask_frac=0.3):
    """CST v2: per-passage additive evidence transport on masked state.

    1. Build fully masked canvas under C0 → logits_c0
    2. For each new passage d_j: logits(C0 ∪ {d_j}) on masked state → delta_j
    3. Aggregate delta = Σ delta_j
    4. Rank positions by SKL(p_c0, p_c0+delta) → remask top positions
    5. Remask those on committed answer, re-denoise under C0 with guidance
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    neg_ent = _neg_entropy()

    # Build C0 prompt
    prefix_c0, n_prefix_c0 = build_short_prompt(tokenizer, context_c0, question)

    # Step 1: Forward pass on FULLY MASKED canvas under C0
    canvas_masked = prefix_c0 + [mask_id] * n_tokens
    x_masked = torch.tensor([canvas_masked], dtype=torch.long, device=device)
    attn_c0 = torch.ones((1, len(canvas_masked)), dtype=torch.long, device=device)

    out_c0 = model(x_masked, attention_mask=attn_c0)
    logits_c0 = prepare_logits(out_c0.logits)[0, n_prefix_c0:n_prefix_c0 + n_tokens].float()

    # Step 2: Per-passage additive counterfactual: logits(C0 ∪ {d_j}) - logits(C0)
    delta_sum = torch.zeros_like(logits_c0)
    n_passages_used = 0

    for passage in new_passages:
        # C0 + one new passage (additive, not replacement)
        c0_plus_dj = context_c0 + "\n\n" + passage
        prefix_cj, n_prefix_cj = build_short_prompt(tokenizer, c0_plus_dj, question)
        canvas_cj = prefix_cj + [mask_id] * n_tokens
        x_cj = torch.tensor([canvas_cj], dtype=torch.long, device=device)
        attn_cj = torch.ones((1, len(canvas_cj)), dtype=torch.long, device=device)

        out_cj = model(x_cj, attention_mask=attn_cj)
        logits_cj = prepare_logits(out_cj.logits)[0, n_prefix_cj:n_prefix_cj + n_tokens].float()

        delta_j = logits_cj - logits_c0
        delta_sum += delta_j
        n_passages_used += 1

    if n_passages_used > 0:
        delta_sum /= n_passages_used  # average shift

    # Step 3: Rank positions by SKL between p_c0 and p_shifted
    p_c0 = torch.softmax(logits_c0, dim=-1).clamp(min=1e-10)
    p_shifted = torch.softmax(logits_c0 + delta_sum, dim=-1).clamp(min=1e-10)
    kl_pq = (p_c0 * (p_c0.log() - p_shifted.log())).sum(dim=-1)
    kl_qp = (p_shifted * (p_shifted.log() - p_c0.log())).sum(dim=-1)
    skl = 0.5 * (kl_pq + kl_qp)

    # Step 4: Identify high-sensitivity positions to remask
    n_remask = max(1, int(n_tokens * remask_frac))
    n_remask = min(n_remask, n_tokens)
    _, remask_positions = torch.topk(skl, n_remask)

    # Step 5: Create canvas with committed tokens, then remask high-sensitivity positions
    canvas_redenoise = prefix_c0 + committed_tokens.tolist()
    x_re = torch.tensor([canvas_redenoise], dtype=torch.long, device=device)
    for pos in remask_positions:
        x_re[0, n_prefix_c0 + pos] = mask_id

    # Step 6: Recompute delta on the actual remasked state (Codex: recompute once after remasking)
    out_re_c0 = model(x_re, attention_mask=attn_c0)
    logits_re_c0 = prepare_logits(out_re_c0.logits)[0, n_prefix_c0:n_prefix_c0 + n_tokens].float()

    delta_re = torch.zeros_like(logits_re_c0)
    for passage in new_passages:
        c0_plus_dj = context_c0 + "\n\n" + passage
        prefix_cj, n_prefix_cj = build_short_prompt(tokenizer, c0_plus_dj, question)
        # Use same remasked answer tokens
        canvas_cj_re = prefix_cj + x_re[0, n_prefix_c0:n_prefix_c0 + n_tokens].tolist()
        x_cj_re = torch.tensor([canvas_cj_re], dtype=torch.long, device=device)
        attn_cj_re = torch.ones((1, len(canvas_cj_re)), dtype=torch.long, device=device)

        out_cj_re = model(x_cj_re, attention_mask=attn_cj_re)
        logits_cj_re = prepare_logits(out_cj_re.logits)[0, n_prefix_cj:n_prefix_cj + n_tokens].float()
        delta_re += (logits_cj_re - logits_re_c0)

    if n_passages_used > 0:
        delta_re /= n_passages_used

    # Step 7: Re-denoise remasked positions under C0 with CST guidance
    remaining = n_remask
    k_per_step = max(1, math.ceil(remaining / steps))

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_re[0, n_prefix_c0:n_prefix_c0 + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        out = model(x_re, attention_mask=attn_c0)
        logits = prepare_logits(out.logits)
        base_logits = logits[0, masked_local + n_prefix_c0].float()

        # CST guidance from remasked-state delta
        guidance = delta_re[masked_local]
        guided_logits = base_logits + beta * guidance

        confidence, x0 = sample_tokens(guided_logits.to(logits.dtype), temperature=0.0, neg_entropy=neg_ent)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x_re[0, masked_local[topk] + n_prefix_c0] = x0[topk]
        remaining -= len(topk)

    answer = decode_answer(tokenizer, x_re[0, n_prefix_c0:n_prefix_c0 + n_tokens])
    return answer, {
        "n_remasked": n_remask,
        "n_new_passages": n_passages_used,
        "mean_skl": round(skl.mean().item(), 4),
        "max_skl": round(skl.max().item(), 4),
        "remask_positions": sorted(remask_positions.tolist()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--remask_frac", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== CST v2 (beta={args.beta}, remask={args.remask_frac}) ===", flush=True)

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

    methods = ["baseline", "cst"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        c0_text = "\n\n".join(initial)

        # Baseline decode under C0
        baseline_ans, committed = simple_decode(model, tokenizer, c0_text, qtext,
                                                steps=args.steps, n_tokens=args.answer_tokens)

        # Retrieve new passages using provisional answer (multi-hop retrieval)
        c1_query = f"query: {qtext} {baseline_ans}"
        new_passages = retriever.retrieve(c1_query, top_k=3)
        # Filter out passages already in C0
        initial_set = set(initial)
        new_passages = [p for p in new_passages if p not in initial_set]

        if len(new_passages) == 0:
            # No new evidence — CST = baseline
            cst_ans = baseline_ans
            cst_stats = {"n_remasked": 0, "n_new_passages": 0, "mean_skl": 0, "max_skl": 0, "remask_positions": []}
        else:
            # CST: counterfactual score transport
            cst_ans, cst_stats = cst_decode(
                model, tokenizer, c0_text, new_passages[:3], qtext,
                committed, steps=args.steps, n_tokens=args.answer_tokens,
                beta=args.beta, remask_frac=args.remask_frac,
            )

        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold,
               "baseline_correct": int(baseline_ans.strip().lower() == gold.strip().lower()),
               "cst_stats": cst_stats}

        for method, ans in [("baseline", baseline_ans), ("cst", cst_ans)]:
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
        if (qi + 1) % 5 == 0 or qi == 0:
            print(f"[{qi+1}/{len(questions)}] ({elapsed:.1f}s) new={cst_stats['n_new_passages']} skl={cst_stats['mean_skl']:.3f}", flush=True)
            for m in methods:
                print(f"  {m:12s} {row[m]['answer'][:50]:50s} F1={row[m]['f1']:.3f}", flush=True)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\nSummary")
    for m in methods:
        print(f"  {m:12s} F1={summary[m]['f1']:.4f} EM={summary[m]['em']:.4f} contain={summary[m]['contain']:.4f}")

    # Also report split by baseline correct vs incorrect
    correct_bl = [r for r in results if r["baseline_correct"]]
    wrong_bl = [r for r in results if not r["baseline_correct"]]
    if wrong_bl:
        cst_f1_on_wrong = sum(r["cst"]["f1"] for r in wrong_bl) / len(wrong_bl)
        bl_f1_on_wrong = sum(r["baseline"]["f1"] for r in wrong_bl) / len(wrong_bl)
        print(f"\n  On baseline-wrong ({len(wrong_bl)}q): baseline F1={bl_f1_on_wrong:.4f}, cst F1={cst_f1_on_wrong:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args)}, f, indent=2)
    print(f"Saved to {args.output}")
    print(f"Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
