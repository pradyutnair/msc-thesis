"""EAMD v2 Fast - Unified batched denoising. One forward pass per step for all 4 methods.
50q @ 128 steps / 128 tokens in <10 min on a single H100.

Run: python -u src/daes/eamd_v2_fast.py --model dream --dataset musique --n_questions 50 --output results/fast_pilot.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever,
    build_short_pair, build_short_cond_and_prior, prepare_logits,
    get_mask_id, decode_answer, compute_f1,
    compute_v2_guidance, compute_signal_and_scale, compute_w_t,
    short_generate, expand_evidence, QUESTION_FILES,
    MODEL_REF, TOKENIZER_REF, MODEL_TYPE_REF,
)
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
from dllm.pipelines.dream.sampler import sample_tokens


def pad_and_batch(sequences, pad_id, device):
    """Pad list of 1D int lists to same length, return [B, L] tensor + attention mask."""
    max_len = max(len(s) for s in sequences)
    padded = []
    masks = []
    for s in sequences:
        pad_len = max_len - len(s)
        padded.append(s + [pad_id] * pad_len)
        masks.append([1] * len(s) + [0] * pad_len)
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


@torch.inference_mode()
def fast_unified_denoise(model, tokenizer, question, old_context, new_context,
                         steps=128, n_tokens=128, temperature=0.1,
                         gamma_cap=8.0, lambda_max=1.0, beta=1.0):
    """Process ONE question: all 4 methods in a single batched denoising loop.

    6 rows per question:
      0: baseline   (C0 context)
      1: aram_cond  (C0 context)
      2: aram_prior (no-context / prior)
      3: pool       (C1 context)
      4: eamd_full  (C1 context)
      5: eamd_base  (C0 context)

    ONE model forward pass per step (batch_size=6).
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Build token sequences for each row
    c1_ids, c0_ids, n_pf = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)
    _, prior_ids, n_pf_prior = build_short_cond_and_prior(tokenizer, old_context, question, n_tokens)

    seqs = [
        list(c0_ids),       # 0: baseline
        list(c0_ids),       # 1: aram_cond
        list(prior_ids),    # 2: aram_prior
        list(c1_ids),       # 3: pool
        list(c1_ids),       # 4: eamd_full
        list(c0_ids),       # 5: eamd_base
    ]
    prefixes = [n_pf, n_pf, n_pf_prior, n_pf, n_pf, n_pf]

    x, attn = pad_and_batch(seqs, pad_id, device)
    # x shape: [6, max_seq_len]

    k_per_step = max(1, math.ceil(n_tokens / steps))
    remaining = [n_tokens, n_tokens, n_tokens, n_tokens]  # baseline, aram, pool, eamd

    # Method definitions: (name, logit_rows, commit_rows)
    # logit_rows: which rows to read logits from
    # commit_rows: which rows to write committed tokens to
    METHODS = [
        ("baseline",       [0],    [0]),
        ("aram",           [1, 2], [1, 2]),
        ("pool",           [3],    [3]),
        ("eamd_v2_regen",  [4, 5], [4, 5]),
    ]

    stats = {name: {"conf": [], "gamma": []} for name, _, _ in METHODS}

    for step in range(steps):
        # === ONE forward pass for all 6 rows ===
        out = model(x, attention_mask=attn)
        all_logits = prepare_logits(out.logits)

        for mi, (name, logit_rows, commit_rows) in enumerate(METHODS):
            if remaining[mi] <= 0:
                continue

            pf = prefixes[logit_rows[0]]
            canvas = x[logit_rows[0], pf:pf + n_tokens]
            masked_local = (canvas == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_local) == 0:
                remaining[mi] = 0
                continue

            # Compute guided logits based on method type
            if name == "baseline":
                logits = all_logits[0, masked_local + pf]
                guided = logits

            elif name == "aram":
                logits_cond = all_logits[1, masked_local + prefixes[1]]
                logits_prior = all_logits[2, masked_local + prefixes[2]]
                _, w_t = compute_w_t(len(masked_local), n_tokens)
                signal, noise, extra, gs = compute_signal_and_scale(
                    logits_cond, logits_prior,
                    lambda_max=lambda_max, beta=beta, schedule=w_t, eps=1e-6,
                )
                guided = logits_prior + extra.unsqueeze(-1) * (logits_cond - logits_prior)
                stats[name]["gamma"].append(extra.mean().item())

            elif name == "pool":
                logits = all_logits[3, masked_local + pf]
                guided = logits

            elif name == "eamd_v2_regen":
                logits_full = all_logits[4, masked_local + prefixes[4]]
                logits_base = all_logits[5, masked_local + prefixes[5]]
                _, w_t = compute_w_t(len(masked_local), n_tokens)
                ig, var, gamma, gs = compute_v2_guidance(
                    logits_full, logits_base, w_t=w_t, gamma_cap=gamma_cap,
                )
                guided = logits_full + gamma.unsqueeze(-1) * (logits_full - logits_base)
                stats[name]["gamma"].append(gamma.mean().item())

            # Sample and commit
            confidence, x0 = sample_tokens(guided, temperature=temperature, neg_entropy=True)
            n_commit = min(k_per_step, remaining[mi])
            if step == steps - 1:
                n_commit = remaining[mi]
            n_commit = min(n_commit, len(confidence))
            _, topk = torch.topk(confidence, n_commit)
            chosen_local = masked_local[topk]
            chosen_tokens = x0[topk]

            # Write committed tokens to ALL rows for this method
            for r in commit_rows:
                x[r, chosen_local + prefixes[r]] = chosen_tokens

            stats[name]["conf"].extend(confidence[topk].tolist())
            remaining[mi] -= len(topk)

    # Extract answers
    answers = {}
    for name, logit_rows, _ in METHODS:
        pf = prefixes[logit_rows[0]]
        tokens = x[logit_rows[0], pf:pf + n_tokens]
        answers[name] = decode_answer(tokenizer, tokens)

    return answers, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--answer_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--gamma_cap", type=float, default=8.0)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    print(f"=== EAMD v2 Fast Pilot ===", flush=True)
    print(f"Model: {args.model}, Steps: {args.steps}, Tokens: {args.answer_tokens}", flush=True)
    t_start = time.time()

    print(f"Loading {args.model}...", flush=True)
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

    print("Loading retriever...", flush=True)
    retriever = Wiki18Retriever()

    questions_file = QUESTION_FILES[args.dataset]
    all_questions = json.load(open(questions_file))
    end_idx = args.start_idx + args.n_questions
    questions = all_questions[args.start_idx:end_idx]
    print(f"Loaded {len(questions)} questions", flush=True)

    # Phase 1: Batch initial retrieval
    print("Phase 1: Batch retrieval...", flush=True)
    t1 = time.time()
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"  Done in {time.time() - t1:.1f}s", flush=True)

    # Phase 2: Evidence expansion (includes quick seed generation)
    print("Phase 2: Evidence expansion...", flush=True)
    t2 = time.time()
    questions_data = []
    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # Quick seed for expansion (16 steps, 16 tokens — just for retrieval queries)
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                         steps=16, n_tokens=16, temperature=args.temperature)

        expanded, cands = expand_evidence(retriever, qtext, old_ctx, initial,
                                           [seed_ans], args.n_candidates, args.expand_top_k)
        new_ctx = "\n\n".join(expanded)

        questions_data.append({
            "question": qtext, "gold": gold, "old_context": old_ctx, "new_context": new_ctx,
            "qid": q.get("qid") or q.get("id", f"dev_{qi}"),
            "n_old": len(initial), "n_new": len(expanded),
        })
        if (qi + 1) % 10 == 0:
            print(f"  Expanded {qi+1}/{len(questions)}", flush=True)
    print(f"  Done in {time.time() - t2:.1f}s", flush=True)

    # Phase 3: Unified batched denoising
    print(f"Phase 3: Unified denoising (batch=6 per question, {args.steps} steps)...", flush=True)
    t3 = time.time()

    methods = ["baseline", "aram", "pool", "eamd_v2_regen"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    for qi, qd in enumerate(questions_data):
        tq = time.time()
        answers, stats = fast_unified_denoise(
            model, tokenizer, qd["question"], qd["old_context"], qd["new_context"],
            steps=args.steps, n_tokens=args.answer_tokens, temperature=args.temperature,
            gamma_cap=args.gamma_cap, lambda_max=args.lambda_max, beta=args.beta,
        )
        eq = time.time() - tq

        row = {"id": qd["qid"], "question": qd["question"], "gold": qd["gold"], "elapsed": round(eq, 2)}
        log_this = (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions_data) - 1
        if log_this:
            print(f"[{qi+1}/{len(questions_data)}] {qd['qid']} ({eq:.1f}s)", flush=True)

        for m in methods:
            ans = answers[m]
            f1_result = compute_f1(ans, qd["gold"])
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == qd["gold"].strip().lower())
            contain = float(qd["gold"].strip().lower() in ans.strip().lower())
            totals[m]["f1"] += f1
            totals[m]["em"] += em
            totals[m]["contain"] += contain
            row[m] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}
            if log_this:
                print(f"  {m:20s} {ans[:50]:50s} F1={f1:.3f} EM={em:.0f}", flush=True)

        results.append(row)

    n = len(questions_data)
    t_denoise = time.time() - t3
    t_total = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Denoising: {t_denoise:.1f}s total, {t_denoise/n:.1f}s/question")
    print(f"Total wall time: {t_total:.1f}s")
    print(f"\nSummary ({n} questions):")
    summary = {}
    for m in methods:
        s = {k: round(v / n, 4) for k, v in totals[m].items()}
        summary[m] = s
        print(f"  {m:20s} F1={s['f1']:.3f} EM={s['em']:.3f} contain={s['contain']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                    "timing": {"denoise_sec": t_denoise, "total_sec": t_total,
                               "sec_per_q": t_denoise / n}}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
