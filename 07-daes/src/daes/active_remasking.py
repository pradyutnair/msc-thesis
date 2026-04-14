"""
Active Remasking for dLLM Multi-Hop QA.

After initial generation:
1. Track per-token confidence during denoising
2. Identify low-confidence answer tokens (bottom P% by neg_entropy)
3. Use high-confidence tokens to form targeted retrieval query
4. Retrieve additional evidence with E5
5. Remask low-confidence positions → re-denoise with new evidence in context

This is unique to dLLMs — AR models cannot selectively revise specific tokens.
"""

import argparse, json, os, sys, time, re, string, pickle, random
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from dataclasses import dataclass

random.seed(42)

# Import shared components from dgmqr
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")
from dgmqr import Retriever, normalize_answer, compute_f1, extract_candidates, FEW_SHOT_PREFIX


# ---------------------------------------------------------------------------
# Generation with full canvas tracking (needed for remasking)
# ---------------------------------------------------------------------------

def dllm_generate_tracked(model, tokenizer, context, question, steps=128,
                          n_tokens=512, temperature=0.1, few_shot=False):
    """
    Generate answer and return the full canvas + per-token confidences.
    Unlike dgmqr.dllm_generate, this returns the raw canvas tensor for remasking.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id

    if few_shot:
        prompt = FEW_SHOT_PREFIX + f"{context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens
    token_confidences = torch.zeros(n_tokens, device=device)

    for step in range(steps):
        if remaining <= 0:
            break
        mi = (x == mask_id)
        if not mi.any():
            break
        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mp = mi[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mp], temperature=temperature, neg_entropy=True)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(conf, min(n_commit, len(conf)))
        selected = mp[topk]
        x[0, selected] = x0[topk]

        for idx, pos in enumerate(selected):
            local_pos = pos.item() - n_prefix
            if 0 <= local_pos < n_tokens:
                token_confidences[local_pos] = conf[topk[idx]]
        remaining -= len(topk)

    gen_ids = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer, token_confidences, x, n_prefix


def remask_and_redenoise(model, tokenizer, x, n_prefix, token_confidences,
                         new_context, question, remask_pct=0.3, steps=64,
                         n_tokens=512, temperature=0.1, few_shot=False):
    """
    Remask bottom remask_pct% of committed tokens and re-denoise with new context.
    
    Key difference from CGRR: we rebuild the canvas with the NEW context prefix
    but preserve high-confidence tokens at their positions.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Build new prompt with additional evidence
    if few_shot:
        prompt = FEW_SHOT_PREFIX + f"{new_context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"{new_context}\n\nQuestion: {question}\n\nAnswer:"

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    new_prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    new_n_prefix = len(new_prefix_ids)

    # Get the generated tokens from original canvas
    gen_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()

    # Find committed NON-EOS tokens and their confidences
    non_eos_mask = (gen_tokens != eos_id) & (gen_tokens != mask_id)
    if not non_eos_mask.any():
        # No content tokens — just re-generate from scratch
        new_canvas = new_prefix_ids + [mask_id] * n_tokens
        x_new = torch.tensor([new_canvas], dtype=torch.long, device=device)
    else:
        non_eos_positions = non_eos_mask.nonzero(as_tuple=True)[0]
        non_eos_conf = token_confidences[non_eos_positions]

        # Determine how many to remask (bottom P% of content tokens)
        n_content = len(non_eos_positions)
        n_to_remask = max(1, int(n_content * remask_pct))

        # Find the lowest-confidence content tokens
        _, remask_indices = torch.topk(non_eos_conf, n_to_remask, largest=False)
        remask_positions = non_eos_positions[remask_indices]

        # Build new canvas: new prefix + gen tokens with some remasked
        gen_remasked = gen_tokens.clone()
        gen_remasked[remask_positions] = mask_id

        new_canvas = new_prefix_ids + gen_remasked.tolist()
        x_new = torch.tensor([new_canvas], dtype=torch.long, device=device)

    attn = torch.ones((1, x_new.shape[1]), dtype=torch.long, device=device)

    # Re-denoise only the masked positions
    remaining = (x_new[0, new_n_prefix:] == mask_id).sum().item()
    if remaining == 0:
        # Nothing to re-denoise
        gen_ids = x_new[0, new_n_prefix:].tolist()
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip(), 0

    k_per_step = max(1, remaining // steps)

    for step in range(steps):
        if remaining <= 0:
            break
        mi = (x_new[0] == mask_id)
        mi[:new_n_prefix] = False  # don't touch prefix
        if not mi.any():
            break
        mp = mi.nonzero(as_tuple=True)[0]

        with torch.no_grad():
            out = model(x_new, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        conf, x0 = sample_tokens(logits[0, mp], temperature=temperature, neg_entropy=True)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(conf, min(n_commit, len(conf)))
        x_new[0, mp[topk]] = x0[topk]
        remaining -= len(topk)

    gen_ids = x_new[0, new_n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return answer, remaining


# ---------------------------------------------------------------------------
# Full active remasking pipeline
# ---------------------------------------------------------------------------

def run_active_remask(model, tokenizer, retriever, question, n_candidates=3,
                      remask_pct=0.3, remask_steps=64, few_shot=False):
    """
    Full pipeline:
    1. Initial retrieval + generation (with confidence tracking)
    2. Extract high-confidence tokens as retrieval query
    3. Candidate-guided retrieval (same as pool mode)
    4. Remask low-confidence tokens
    5. Re-denoise with expanded evidence
    """
    # Stage 1: Initial retrieval + tracked generation
    initial_passages = retriever.retrieve(question, top_k=5)
    initial_context = "\n\n".join(initial_passages)

    init_answer, token_conf, canvas, n_prefix = dllm_generate_tracked(
        model, tokenizer, initial_context, question, few_shot=few_shot
    )

    # Stage 2: Extract candidates from dLLM distribution
    candidates = extract_candidates(model, tokenizer, initial_context, question, n_candidates)

    # Stage 3: Candidate-guided retrieval → pool passages
    all_passages = list(initial_passages)
    seen_texts = set(p[:100] for p in all_passages)

    # Also use high-confidence tokens from initial answer as additional query
    if init_answer and len(init_answer.strip()) > 2:
        answer_query = f"{question} {init_answer.strip()[:100]}"
        answer_passages = retriever.retrieve(answer_query, top_k=3)
        for p in answer_passages:
            if p[:100] not in seen_texts:
                all_passages.append(p)
                seen_texts.add(p[:100])

    for cand in candidates:
        hop2_query = f"{question} {cand['text']}"
        hop2_passages = retriever.retrieve(hop2_query, top_k=3)
        for p in hop2_passages:
            if p[:100] not in seen_texts:
                all_passages.append(p)
                seen_texts.add(p[:100])

    expanded_context = "\n\n".join(all_passages)

    # Stage 4+5: Remask low-confidence tokens and re-denoise with expanded context
    remasked_answer, remaining = remask_and_redenoise(
        model, tokenizer, canvas, n_prefix, token_conf,
        expanded_context, question,
        remask_pct=remask_pct, steps=remask_steps, few_shot=few_shot,
    )

    # Count content tokens in initial vs remasked
    eos_id = tokenizer.eos_token_id
    init_tokens = canvas[0, n_prefix:].tolist()
    n_content_init = sum(1 for t in init_tokens if t != eos_id and t != tokenizer.mask_token_id)

    stats = {
        "method": "active_remask",
        "init_answer": init_answer[:80],
        "n_candidates": len(candidates),
        "n_passages": len(all_passages),
        "n_content_tokens_init": n_content_init,
        "remask_pct": remask_pct,
        "candidates": [c["text"][:40] for c in candidates],
    }
    return remasked_answer, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Active Remasking for dLLM Multi-Hop QA")
    parser.add_argument("--dataset", default="musique", choices=["musique", "hotpotqa", "2wikimultihop"])
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--remask_pct", type=float, default=0.3, help="Fraction of content tokens to remask")
    parser.add_argument("--remask_steps", type=int, default=64, help="Denoising steps for re-denoising")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.end_idx is not None:
        start, end = args.start_idx, args.end_idx
    else:
        start, end = args.start_idx, args.start_idx + args.n_questions

    retriever = Retriever(args.dataset)

    print("Loading Dream-7B...", flush=True)
    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print("Model loaded.", flush=True)

    all_qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))
    qs = all_qs[start:end]
    print(f"Active Remasking on {args.dataset} [{start}:{end}] ({len(qs)}q, remask={args.remask_pct})", flush=True)

    fs_tag = "_fewshot" if args.few_shot else ""
    rp_tag = f"_rp{int(args.remask_pct*100)}"
    out_path = os.path.join(args.output_dir,
        f"dgmqr_remask{rp_tag}{fs_tag}_{args.dataset}_{start}_{end}.jsonl")

    # Also run baseline + pool for comparison (same questions)
    from dgmqr import run_baseline, run_pool, dllm_generate

    predictions = []
    sum_f1 = {"baseline": 0, "pool": 0, "remask": 0}
    sum_contain = {"baseline": 0, "pool": 0, "remask": 0}

    for i, q in enumerate(qs):
        t0 = time.time()

        # Baseline
        bl_answer, _ = run_baseline(model, tokenizer, retriever, q["question"], few_shot=args.few_shot)
        _, _, bl_f1 = compute_f1(bl_answer, q["answer"])
        bl_contain = q["answer"].lower() in bl_answer.lower()

        # Pool
        pool_answer, _ = run_pool(model, tokenizer, retriever, q["question"],
                                   n_candidates=args.n_candidates, few_shot=args.few_shot)
        _, _, pool_f1 = compute_f1(pool_answer, q["answer"])
        pool_contain = q["answer"].lower() in pool_answer.lower()

        # Active remasking
        remask_answer, stats = run_active_remask(
            model, tokenizer, retriever, q["question"],
            n_candidates=args.n_candidates,
            remask_pct=args.remask_pct,
            remask_steps=args.remask_steps,
            few_shot=args.few_shot,
        )
        _, _, remask_f1 = compute_f1(remask_answer, q["answer"])
        remask_contain = q["answer"].lower() in remask_answer.lower()

        elapsed = time.time() - t0

        sum_f1["baseline"] += bl_f1
        sum_f1["pool"] += pool_f1
        sum_f1["remask"] += remask_f1
        sum_contain["baseline"] += bl_contain
        sum_contain["pool"] += pool_contain
        sum_contain["remask"] += remask_contain

        pred = {
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "baseline_answer": bl_answer,
            "pool_answer": pool_answer,
            "remask_answer": remask_answer,
            "baseline_f1": round(bl_f1, 4),
            "pool_f1": round(pool_f1, 4),
            "remask_f1": round(remask_f1, 4),
            "baseline_contain": bl_contain,
            "pool_contain": pool_contain,
            "remask_contain": remask_contain,
            "time": round(elapsed, 2),
            "stats": stats,
        }
        predictions.append(pred)

        with open(out_path, "a") as fw:
            fw.write(json.dumps(pred) + "\n")

        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) "
              f"BL={bl_f1:.2f} Pool={pool_f1:.2f} Remask={remask_f1:.2f} "
              f"| Gold: {q['answer']} | Remask: {remask_answer[:60]}", flush=True)

    n = len(predictions)
    print(f"\n{'='*60}", flush=True)
    print(f"Active Remasking | {args.dataset} | N={n} | remask_pct={args.remask_pct}", flush=True)
    for mode in ["baseline", "pool", "remask"]:
        print(f"  {mode:10s}: F1={sum_f1[mode]/n*100:.1f}%  Contain={sum_contain[mode]/n*100:.1f}%", flush=True)
    print(f"  Output: {out_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
