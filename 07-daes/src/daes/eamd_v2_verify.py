"""EAMD-Verify: Best-of-K with evidence-marginal reranking.
Tests whether answer-level evidence scoring beats Pool.

Run: python -u src/daes/eamd_v2_verify.py --model dream --dataset musique --n_questions 50 --K 8
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, build_short_pair, prepare_logits,
    get_mask_id, decode_answer, compute_f1,
    short_generate, expand_evidence, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import random


@torch.inference_mode()
def generate_k_candidates(model, tokenizer, context, question, K=8,
                          steps=32, n_tokens=32, temperature=0.3):
    """Generate K diverse candidate answers from Pool(C1) using different random seeds."""
    device = model.device
    mask_id = get_mask_id(tokenizer)

    from eamd_v2_wiki18 import short_user_prompt
    prompt = short_user_prompt(context, question)
    msg = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    # Build K copies of the same input, all starting fully masked
    full_ids = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([full_ids] * K, dtype=torch.long, device=device)  # [K, seq_len]
    attn = torch.ones((K, len(full_ids)), dtype=torch.long, device=device)

    remaining = [n_tokens] * K
    k_per_step = max(1, math.ceil(n_tokens / steps))

    for step in range(steps):
        # Find masked positions (same structure across candidates but diverge after first commit)
        # Process all K candidates in one batched forward pass
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)

        for ki in range(K):
            if remaining[ki] <= 0:
                continue
            masked_local = (x[ki, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_local) == 0:
                remaining[ki] = 0
                continue

            pos = masked_local + n_prefix
            token_logits = logits[ki, pos]

            confidence, x0 = sample_tokens(token_logits, temperature=temperature, neg_entropy=True)
            n_commit = min(k_per_step, remaining[ki])
            if step == steps - 1:
                n_commit = remaining[ki]
            n_commit = min(n_commit, len(confidence))
            # DIVERSITY: randomly select positions to unmask (not confidence-based)
            # This ensures K candidates diverge in unmasking order
            perm = torch.randperm(len(confidence), device=confidence.device)
            topk = perm[:n_commit]
            x[ki, masked_local[topk] + n_prefix] = x0[topk]
            remaining[ki] -= len(topk)

    # Extract answers
    candidates = []
    for ki in range(K):
        tokens = x[ki, n_prefix:n_prefix + n_tokens]
        answer = decode_answer(tokenizer, tokens)
        candidates.append({"answer": answer, "tokens": tokens.clone()})

    return candidates, n_prefix


@torch.inference_mode()
def score_candidates_pseudo_likelihood(model, tokenizer, candidates, n_prefix,
                                        old_context, new_context, question, n_tokens):
    """Score each candidate using masked pseudo-likelihood under C0 and C1.

    For each candidate, for each answer token position i:
      - mask token i
      - compute log p(a_i | a_{-i}, C1) and log p(a_i | a_{-i}, C0)
      - evidence score contribution = log p(a_i | a_{-i}, C1) - log p(a_i | a_{-i}, C0)

    This gives per-candidate:
      - log_pl_c1: pseudo-log-likelihood under C1
      - log_pl_c0: pseudo-log-likelihood under C0
      - evidence_score: sum of per-token evidence contributions
      - mean_confidence: average token confidence under C1
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    eos_id = tokenizer.eos_token_id

    # Build C1 and C0 prompt prefixes
    c1_ids, c0_ids, n_pf = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)

    scores = []
    for cand in candidates:
        tokens = cand["tokens"]

        # Find content positions (before EOS)
        token_list = tokens.tolist()
        if eos_id in token_list:
            content_len = token_list.index(eos_id)
        else:
            content_len = len([t for t in token_list if t != mask_id])
        content_len = max(1, min(content_len, n_tokens))

        # Build full sequences with this candidate's answer
        x_c1 = torch.tensor([c1_ids], dtype=torch.long, device=device)
        x_c0 = torch.tensor([c0_ids], dtype=torch.long, device=device)
        x_c1[0, n_pf:n_pf + n_tokens] = tokens
        x_c0[0, n_pf:n_pf + n_tokens] = tokens

        log_pl_c1 = 0.0
        log_pl_c0 = 0.0

        # For each content position, mask it and score
        # Batch all masked variants for efficiency
        batch_c1 = x_c1.repeat(content_len, 1)  # [content_len, seq_len]
        batch_c0 = x_c0.repeat(content_len, 1)

        for i in range(content_len):
            batch_c1[i, n_pf + i] = mask_id
            batch_c0[i, n_pf + i] = mask_id

        attn_c1 = torch.ones_like(batch_c1)
        attn_c0 = torch.ones_like(batch_c0)

        # Batched forward passes
        out_c1 = model(batch_c1, attention_mask=attn_c1)
        out_c0 = model(batch_c0, attention_mask=attn_c0)

        logits_c1 = prepare_logits(out_c1.logits)
        logits_c0 = prepare_logits(out_c0.logits)

        for i in range(content_len):
            true_token = tokens[i].item()
            pos = n_pf + i

            lp_c1 = F.log_softmax(logits_c1[i, pos], dim=-1)[true_token].item()
            lp_c0 = F.log_softmax(logits_c0[i, pos], dim=-1)[true_token].item()

            log_pl_c1 += lp_c1
            log_pl_c0 += lp_c0

        evidence_score = log_pl_c1 - log_pl_c0

        # Also compute mean confidence from C1 (no masking, just softmax confidence)
        out_full = model(x_c1, attention_mask=torch.ones_like(x_c1))
        full_logits = prepare_logits(out_full.logits)
        probs = F.softmax(full_logits[0, n_pf:n_pf + content_len], dim=-1)
        conf = probs[range(content_len), tokens[:content_len].tolist()].mean().item()

        scores.append({
            "log_pl_c1": log_pl_c1,
            "log_pl_c0": log_pl_c0,
            "evidence_score": evidence_score,
            "confidence": conf,
            "content_len": content_len,
            "combined_score": log_pl_c1 + 0.5 * evidence_score,
        })

    return scores


def select_by_strategy(candidates, scores, strategy):
    """Select best candidate by strategy."""
    if strategy == "random":
        idx = random.randint(0, len(candidates) - 1)
    elif strategy == "confidence":
        idx = max(range(len(scores)), key=lambda i: scores[i]["confidence"])
    elif strategy == "log_pl_c1":
        idx = max(range(len(scores)), key=lambda i: scores[i]["log_pl_c1"])
    elif strategy == "evidence_score":
        idx = max(range(len(scores)), key=lambda i: scores[i]["evidence_score"])
    elif strategy == "combined":
        idx = max(range(len(scores)), key=lambda i: scores[i]["combined_score"])
    else:
        idx = 0
    return idx, candidates[idx]["answer"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--gen_temperature", type=float, default=0.3)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== EAMD-Verify: Best-of-{args.K} with Evidence Reranking ===", flush=True)
    print(f"Model: {args.model}, Steps: {args.steps}, Tokens: {args.answer_tokens}, K: {args.K}", flush=True)

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

    # Batch retrieval
    print("Phase 1: Batch retrieval...", flush=True)
    t1 = time.time()
    queries = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(queries, args.initial_top_k)
    print(f"  Done in {time.time() - t1:.1f}s", flush=True)

    # Evidence expansion
    print("Phase 2: Evidence expansion...", flush=True)
    t2 = time.time()
    qdata = []
    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                         steps=16, n_tokens=16, temperature=0.0)
        expanded, _ = expand_evidence(retriever, qtext, old_ctx, initial,
                                       [seed_ans], args.n_candidates, args.expand_top_k)
        new_ctx = "\n\n".join(expanded)
        qdata.append({"question": qtext, "gold": gold, "old_context": old_ctx,
                       "new_context": new_ctx, "qid": q.get("qid") or q.get("id", f"dev_{qi}")})
        if (qi + 1) % 10 == 0:
            print(f"  Expanded {qi+1}/{len(questions)}", flush=True)
    print(f"  Done in {time.time() - t2:.1f}s", flush=True)

    # Phase 3: Generate + Score + Rerank
    print(f"Phase 3: Generate K={args.K} candidates + score + rerank...", flush=True)
    t3 = time.time()

    strategies = ["pool_top1", "oracle_K", "random_K", "confidence", "log_pl_c1", "evidence_score", "combined"]
    totals = {s: {"f1": 0.0, "em": 0.0, "contain": 0.0} for s in strategies}
    results = []

    for qi, qd in enumerate(qdata):
        tq = time.time()

        # Generate K candidates
        candidates, n_prefix = generate_k_candidates(
            model, tokenizer, qd["new_context"], qd["question"],
            K=args.K, steps=args.steps, n_tokens=args.answer_tokens,
            temperature=args.gen_temperature,
        )

        # Score candidates
        scores = score_candidates_pseudo_likelihood(
            model, tokenizer, candidates, n_prefix,
            qd["old_context"], qd["new_context"], qd["question"], args.answer_tokens,
        )

        # Evaluate each strategy
        row = {"id": qd["qid"], "question": qd["question"], "gold": qd["gold"],
               "elapsed": round(time.time() - tq, 2), "n_unique": len(set(c["answer"] for c in candidates))}

        # Pool top-1 (first candidate, deterministic seed would give this)
        pool_ans = candidates[0]["answer"]
        f1_result = compute_f1(pool_ans, qd["gold"])
        f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
        em = float(pool_ans.strip().lower() == qd["gold"].strip().lower())
        contain = float(qd["gold"].strip().lower() in pool_ans.strip().lower())
        totals["pool_top1"]["f1"] += f1
        totals["pool_top1"]["em"] += em
        totals["pool_top1"]["contain"] += contain
        row["pool_top1"] = {"answer": pool_ans, "f1": round(f1, 4), "em": em}

        # Oracle (best possible among K)
        best_f1 = 0.0
        best_idx = 0
        for ci, cand in enumerate(candidates):
            cf1_result = compute_f1(cand["answer"], qd["gold"])
            cf1 = cf1_result if isinstance(cf1_result, (int, float)) else cf1_result[0]
            if cf1 > best_f1:
                best_f1 = cf1
                best_idx = ci
        oracle_ans = candidates[best_idx]["answer"]
        oracle_em = float(oracle_ans.strip().lower() == qd["gold"].strip().lower())
        oracle_contain = float(qd["gold"].strip().lower() in oracle_ans.strip().lower())
        totals["oracle_K"]["f1"] += best_f1
        totals["oracle_K"]["em"] += oracle_em
        totals["oracle_K"]["contain"] += oracle_contain
        row["oracle_K"] = {"answer": oracle_ans, "f1": round(best_f1, 4), "em": oracle_em}

        # Other strategies
        for strategy in ["random_K", "confidence", "log_pl_c1", "evidence_score", "combined"]:
            s_name = strategy.replace("_K", "")
            idx, ans = select_by_strategy(candidates, scores, s_name if strategy != "random_K" else "random")
            sf1_result = compute_f1(ans, qd["gold"])
            sf1 = sf1_result if isinstance(sf1_result, (int, float)) else sf1_result[0]
            sem = float(ans.strip().lower() == qd["gold"].strip().lower())
            scontain = float(qd["gold"].strip().lower() in ans.strip().lower())
            totals[strategy]["f1"] += sf1
            totals[strategy]["em"] += sem
            totals[strategy]["contain"] += scontain
            row[strategy] = {"answer": ans, "f1": round(sf1, 4), "em": sem, "selected_idx": idx}

        # Store candidate details
        row["candidates"] = [{"answer": c["answer"], **s} for c, s in zip(candidates, scores)]
        results.append(row)

        if (qi + 1) % 5 == 0 or qi == 0 or qi == len(qdata) - 1:
            elapsed = time.time() - tq
            print(f"  [{qi+1}/{len(qdata)}] {qd['qid']} ({elapsed:.1f}s) unique={row['n_unique']}", flush=True)
            print(f"    pool_top1  F1={row['pool_top1']['f1']:.3f}  oracle  F1={row['oracle_K']['f1']:.3f}  evidence  F1={row.get('evidence_score',{}).get('f1',0):.3f}", flush=True)

    n = len(qdata)
    t_total = time.time() - t_start

    print(f"\n{'Strategy':<20s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 42)
    summary = {}
    for s in strategies:
        sv = {k: round(v / n, 4) for k, v in totals[s].items()}
        summary[s] = sv
        marker = " <-- BEST" if sv["f1"] == max(summary[ss]["f1"] for ss in summary) else ""
        print(f"{s:<20s} {sv['f1']:>6.3f} {sv['em']:>6.3f} {sv['contain']:>8.3f}{marker}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                    "timing": {"total_sec": round(t_total, 1)}}, f, indent=2)
    print(f"\nTotal: {t_total:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
