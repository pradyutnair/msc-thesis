"""DIR: Denoising-Interleaved Retrieval for Multi-Hop QA.
Retrieval happens INSIDE the denoising loop at checkpoint steps.

Run: python -u src/daes/eamd_dir.py --model dream --dataset musique --n_questions 50
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, build_short_pair, prepare_logits,
    get_mask_id, decode_answer, compute_f1, compute_w_t,
    compute_v2_guidance, short_generate, QUESTION_FILES,
    short_user_prompt,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


def extract_partial_answer(tokenizer, x, n_prefix, n_tokens, mask_id):
    """Extract partial answer preserving positional order.
    Replace mask tokens with a placeholder so the string is coherent."""
    answer_region = x[0, n_prefix:n_prefix + n_tokens].tolist()
    eos_id = tokenizer.eos_token_id
    # Truncate at first EOS
    if eos_id in answer_region:
        answer_region = answer_region[:answer_region.index(eos_id)]
    # Replace masks with empty but keep position order
    clean = [t for t in answer_region if t != mask_id]
    if not clean:
        return ""
    return tokenizer.decode(clean, skip_special_tokens=True).strip()


def extract_bridge_candidates(tokenizer, logits, x, n_prefix, n_tokens, mask_id, top_k=3):
    """Extract top-k bridge entity candidates from the posterior at masked positions."""
    answer_region = x[0, n_prefix:n_prefix + n_tokens]
    masked_positions = (answer_region == mask_id).nonzero(as_tuple=True)[0]
    if len(masked_positions) == 0:
        return []

    # Use the position with highest entropy (most uncertain = most likely bridge position)
    pos_logits = logits[0, masked_positions + n_prefix]
    probs = F.softmax(pos_logits, dim=-1)
    entropies = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    # Pick the highest-entropy position
    bridge_pos = masked_positions[entropies.argmax()]
    bridge_logits = logits[0, bridge_pos + n_prefix]
    bridge_probs = F.softmax(bridge_logits, dim=-1)

    # Get top-k candidates
    topk_probs, topk_ids = torch.topk(bridge_probs, min(top_k, len(bridge_probs)))
    candidates = []
    for prob, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
        text = tokenizer.decode([tid], skip_special_tokens=True).strip()
        if text and len(text) > 1:  # Skip single chars
            candidates.append({"text": text, "prob": prob})
    return candidates


def build_retrieval_queries(question, partial_answer, bridge_candidates):
    """Build retrieval queries from the current denoising state."""
    queries = []
    # Query 1: question + partial answer
    if partial_answer:
        queries.append(f"query: {question} {partial_answer}")
    # Query 2+: question + each bridge candidate
    for cand in bridge_candidates:
        queries.append(f"query: {question} {cand['text']}")
    if not queries:
        queries.append(f"query: {question}")
    return queries


@torch.inference_mode()
def dir_denoise(model, tokenizer, retriever, question, initial_passages,
                steps=32, n_tokens=32, temperature=0.0,
                gamma_mult=0.5, gamma_cap=8.0,
                checkpoint_fracs=(0.25, 0.50, 0.75),
                expand_top_k=3, bridge_top_k=3,
                remask_tau=0.20, remask_prior=0.30, remask_cost=0.0,
                remask_max_m=4):
    """DIR denoising with retrieval checkpoints.

    At each checkpoint:
    1. Extract partial answer + bridge candidates from current state
    2. Retrieve new evidence
    3. Ghost step: evaluate under new evidence without changing canvas
    4. Remask tokens with high KL divergence
    5. Continue denoising with updated evidence
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    eos_id = tokenizer.eos_token_id

    # Compute checkpoint steps
    checkpoint_steps = set(int(frac * steps) for frac in checkpoint_fracs)

    # Current evidence
    current_passages = list(initial_passages)
    current_context = "\n\n".join(current_passages)

    # Build initial sequences
    prompt = short_user_prompt(current_context, question)
    msg = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    full_ids = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))

    # Tracking stats
    stats = {
        "checkpoints_triggered": 0,
        "total_new_passages": 0,
        "total_remasked": 0,
        "recovery_rate": [],
        "checkpoint_details": [],
    }

    for step in range(steps):
        if remaining <= 0:
            break

        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        # === RETRIEVAL CHECKPOINT ===
        if step in checkpoint_steps and step > 0:
            # Get current model predictions for the full canvas
            out_pre = model(x, attention_mask=attn)
            logits_pre = prepare_logits(out_pre.logits)

            # 1. Extract partial answer from committed tokens
            partial_answer = extract_partial_answer(tokenizer, x, n_prefix, n_tokens, mask_id)

            # 2. Extract bridge candidates from uncertain positions
            bridge_candidates = extract_bridge_candidates(
                tokenizer, logits_pre, x, n_prefix, n_tokens, mask_id, top_k=bridge_top_k
            )

            # 3. Build and execute retrieval queries
            queries = build_retrieval_queries(question, partial_answer, bridge_candidates)
            new_results = retriever.retrieve_batch(queries, expand_top_k)

            # Union new passages with existing
            existing_set = set(current_passages)
            new_passages = []
            for result_list in new_results:
                for passage in result_list:
                    if passage not in existing_set:
                        new_passages.append(passage)
                        existing_set.add(passage)

            stats["checkpoints_triggered"] += 1  # Count all checkpoint executions
            if new_passages:
                current_passages.extend(new_passages)
                new_context = "\n\n".join(current_passages)
                stats["total_new_passages"] += len(new_passages)

                # 4. Rebuild sequences with new context
                new_prompt = short_user_prompt(new_context, question)
                new_msg = [{"role": "user", "content": new_prompt}]
                new_text = tokenizer.apply_chat_template(new_msg, tokenize=False, add_generation_prompt=True)
                new_prefix_ids = tokenizer.encode(new_text, add_special_tokens=False)
                new_n_prefix = len(new_prefix_ids)

                # Transfer answer tokens to new sequence
                answer_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()
                new_full_ids = new_prefix_ids + [mask_id] * n_tokens
                x_new = torch.tensor([new_full_ids], dtype=torch.long, device=device)
                x_new[0, new_n_prefix:new_n_prefix + n_tokens] = answer_tokens
                attn_new = torch.ones((1, len(new_full_ids)), dtype=torch.long, device=device)

                # 5. Ghost step: evaluate under new evidence
                out_post = model(x_new, attention_mask=attn_new)
                logits_post = prepare_logits(out_post.logits)

                # 6. Compute KL divergence for committed positions and remask
                committed_positions = (x_new[0, new_n_prefix:new_n_prefix + n_tokens] != mask_id).nonzero(as_tuple=True)[0]

                if len(committed_positions) > 0:
                    # Get old and new logits at committed positions
                    # Old logits from pre-checkpoint pass (need to align positions)
                    # Use the pre-checkpoint logits at the answer positions
                    old_pos = committed_positions + n_prefix
                    new_pos = committed_positions + new_n_prefix

                    log_p_old = F.log_softmax(logits_pre[0, old_pos], dim=-1)
                    log_p_new = F.log_softmax(logits_post[0, new_pos], dim=-1)
                    p_old = log_p_old.exp()

                    # KL divergence: D_KL(p_old || p_new)
                    kl_div = (p_old * (log_p_old - log_p_new)).sum(dim=-1)

                    # Remask top-m positions with highest KL
                    if remask_max_m > 0:
                        # Theorem 7.3: logistic remask rule
                        import math as _math
                        _prior = max(1e-6, min(1-1e-6, remask_prior))
                        logit_prior = _math.log(_prior / (1.0 - _prior))
                        remask_probs = torch.sigmoid(
                            torch.tensor(logit_prior, device=kl_div.device)
                            + (kl_div - remask_cost) / max(remask_tau, 1e-6)
                        )
                        remask_decisions = torch.bernoulli(remask_probs).bool()
                        eligible_indices = remask_decisions.nonzero(as_tuple=True)[0]

                        if len(eligible_indices) > 0:
                            if len(eligible_indices) > remask_max_m:
                                _, top_kl = torch.topk(kl_div[eligible_indices], remask_max_m)
                                eligible_indices = eligible_indices[top_kl]
                            remask_local = committed_positions[eligible_indices]

                            # Remask these positions
                            x_new[0, remask_local + new_n_prefix] = mask_id
                            remaining += len(remask_local)
                            stats["total_remasked"] += len(remask_local)

                            # Log remasked positions for recovery rate
                            remasked_info = [(pos.item(), x_new[0, pos + new_n_prefix].item()) for pos in remask_local]
                            stats.setdefault("remasked_log", []).append({"step": step, "positions": remasked_info})

                            checkpoint_detail = {
                                "step": step,
                                "new_passages": len(new_passages),
                                "remasked": len(remask_local),
                                "mean_kl": kl_div.mean().item(),
                                "max_kl": kl_div.max().item(),
                                "partial_answer": partial_answer[:50],
                                "bridge_candidates": [c["text"] for c in bridge_candidates[:3]],
                            }
                            stats["checkpoint_details"].append(checkpoint_detail)

                # Update state
                x = x_new
                attn = attn_new
                n_prefix = new_n_prefix
                current_context = new_context
            else:
                # Checkpoint executed but no new passages found
                stats["checkpoint_details"].append({
                    "step": step, "new_passages": 0, "remasked": 0,
                    "partial_answer": partial_answer[:50],
                    "bridge_candidates": [c["text"] for c in bridge_candidates[:3]],
                })

        # === STANDARD DENOISING STEP ===
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, masked_local + n_prefix]

        # Sample and commit
        confidence, x0 = sample_tokens(token_logits, temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens]
    answer = decode_answer(tokenizer, answer_tokens)
    stats["n_passages_final"] = len(current_passages)

    return answer, stats


@torch.inference_mode()
def baseline_denoise(model, tokenizer, context, question,
                     steps=32, n_tokens=32, temperature=0.0):
    """Simple baseline: denoise with fixed context, no checkpoints."""
    device = model.device
    mask_id = get_mask_id(tokenizer)

    prompt = short_user_prompt(context, question)
    msg = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    full_ids = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)

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
        confidence, x0 = sample_tokens(token_logits, temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens]
    return decode_answer(tokenizer, answer_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--bridge_top_k", type=int, default=3)
    parser.add_argument("--gamma_mult", type=float, default=0.5)
    parser.add_argument("--gamma_cap", type=float, default=8.0)
    parser.add_argument("--remask_max_m", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== DIR: Denoising-Interleaved Retrieval ===", flush=True)
    print(f"Model: {args.model}, Steps: {args.steps}, Tokens: {args.answer_tokens}", flush=True)

    # Load model
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    import eamd_v2_wiki18
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"  Model loaded in {time.time() - t_start:.1f}s", flush=True)

    # Load retriever + questions
    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    # Batch initial retrieval
    print("Batch initial retrieval...", flush=True)
    t1 = time.time()
    queries = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(queries, args.initial_top_k)
    print(f"  Done in {time.time() - t1:.1f}s", flush=True)

    # Run all methods
    print(f"Running methods: baseline, pool, DIR...", flush=True)
    methods = ["baseline", "pool", "dir"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold}

        # Baseline: decode from C0 only
        baseline_ans = baseline_denoise(model, tokenizer, old_ctx, qtext,
                                         steps=args.steps, n_tokens=args.answer_tokens,
                                         temperature=args.temperature)

        # Pool: pre-expand evidence then decode (same as before)
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                         steps=16, n_tokens=16, temperature=args.temperature)
        # Simple expansion: question + seed answer
        expand_queries = [f"query: {qtext} {seed_ans}"]
        expand_results = retriever.retrieve_batch(expand_queries, args.expand_top_k)
        expanded = list(initial)
        existing = set(initial)
        for passage in expand_results[0]:
            if passage not in existing:
                expanded.append(passage)
                existing.add(passage)
        pool_ctx = "\n\n".join(expanded)
        pool_ans = baseline_denoise(model, tokenizer, pool_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens,
                                     temperature=args.temperature)

        # DIR: denoising with retrieval checkpoints
        dir_ans, dir_stats = dir_denoise(
            model, tokenizer, retriever, qtext, initial,
            steps=args.steps, n_tokens=args.answer_tokens,
            temperature=args.temperature,
            gamma_mult=args.gamma_mult, gamma_cap=args.gamma_cap,
            expand_top_k=args.expand_top_k, bridge_top_k=args.bridge_top_k,
            remask_max_m=args.remask_max_m,
        )

        elapsed = time.time() - tq

        for method, ans in [("baseline", baseline_ans), ("pool", pool_ans), ("dir", dir_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        row["dir_stats"] = dir_stats
        row["elapsed"] = round(elapsed, 2)
        results.append(row)

        log_this = (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1
        if log_this:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s) "
                  f"ckpts={dir_stats['checkpoints_triggered']} "
                  f"new_pass={dir_stats['total_new_passages']} "
                  f"remasked={dir_stats['total_remasked']}", flush=True)
            for m in methods:
                print(f"  {m:12s} {row[m]['answer'][:50]:50s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

    n = len(questions)
    t_total = time.time() - t_start

    print(f"\n{'Method':<12s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 34)
    summary = {}
    for m in methods:
        s = {k: round(v / n, 4) for k, v in totals[m].items()}
        summary[m] = s
        print(f"{m:<12s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")

    # DIR-specific stats
    avg_ckpts = sum(r["dir_stats"]["checkpoints_triggered"] for r in results) / n
    avg_new = sum(r["dir_stats"]["total_new_passages"] for r in results) / n
    avg_remask = sum(r["dir_stats"]["total_remasked"] for r in results) / n
    print(f"\nDIR stats: avg_checkpoints={avg_ckpts:.1f} avg_new_passages={avg_new:.1f} avg_remasked={avg_remask:.1f}")
    print(f"Total: {t_total:.1f}s ({t_total/n:.1f}s/q)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                    "timing": {"total_sec": round(t_total, 1), "sec_per_q": round(t_total / n, 1)},
                    "dir_summary": {"avg_checkpoints": avg_ckpts, "avg_new_passages": avg_new, "avg_remasked": avg_remask}},
                   f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
