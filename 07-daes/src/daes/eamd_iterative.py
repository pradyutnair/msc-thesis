"""Iterative EAMD (iEAMD): Multi-round bridge-candidate-driven evidence expansion
with attenuated guidance. The final method.

Each round: extract bridge candidates from token distribution -> expand retrieval ->
guided decode with gamma_0.5. Stops when no new passages or answer stabilizes.

Run: python -u src/daes/eamd_iterative.py --model dream --dataset musique --n_questions 50
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy,
    Wiki18Retriever, build_short_pair, prepare_logits,
    get_mask_id, decode_answer, compute_f1, compute_w_t,
    compute_v2_guidance, short_generate, QUESTION_FILES,
    short_user_prompt,
)
from dgmqr import extract_candidates
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


@torch.inference_mode()
def decode_with_guidance(model, tokenizer, old_context, new_context, question,
                         steps=32, n_tokens=32, temperature=0.0,
                         gamma_mult=0.5, gamma_cap=8.0):
    """Decode with EAMD v2 attenuated guidance between old and new evidence."""
    device = model.device
    mask_id = get_mask_id(tokenizer)

    c1_ids, c0_ids, n_prefix = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)
    x_full = torch.tensor([c1_ids], dtype=torch.long, device=device)
    x_base = torch.tensor([c0_ids], dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(c1_ids)), dtype=torch.long, device=device)
    attn_base = torch.ones((1, len(c0_ids)), dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    gammas = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + n_prefix

        # Batched forward pass
        x_pair = torch.cat([x_base, x_full], dim=0)
        attn_pair = torch.cat([attn_base, attn_full], dim=0)
        out = model(x_pair, attention_mask=attn_pair)
        logits_pair = prepare_logits(out.logits)
        logits_base = logits_pair[0, full_pos]
        logits_full = logits_pair[1, full_pos]

        if gamma_mult > 0:
            _, w_t = compute_w_t(len(masked_local), n_tokens)
            ig, var, gamma, _ = compute_v2_guidance(logits_full, logits_base, w_t=w_t, gamma_cap=gamma_cap)
            gamma = gamma * gamma_mult
            guided = logits_full + gamma.unsqueeze(-1) * (logits_full - logits_base)
            gammas.append(gamma.mean().item())
        else:
            guided = logits_full

        confidence, x0 = sample_tokens(guided, temperature=temperature, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        chosen = masked_local[topk]
        x_full[0, chosen + n_prefix] = x0[topk]
        x_base[0, chosen + n_prefix] = x0[topk]
        remaining -= len(topk)

    answer_tokens = x_full[0, n_prefix:n_prefix + n_tokens]
    answer = decode_answer(tokenizer, answer_tokens)
    return answer, answer_tokens, {"mean_gamma": sum(gammas) / len(gammas) if gammas else 0.0}


@torch.inference_mode()
def simple_decode(model, tokenizer, context, question, steps=32, n_tokens=32, temperature=0.0):
    """Simple decode from context with no guidance."""
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
        confidence, x0 = sample_tokens(token_logits, temperature=temperature, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens]
    return decode_answer(tokenizer, answer_tokens), answer_tokens


def expand_evidence_round(retriever, question, current_answer, bridge_candidates,
                          current_passages, expand_top_k=3):
    """One round of evidence expansion using answer + bridge candidates."""
    queries = [f"query: {question} {current_answer}"]
    for cand in bridge_candidates:
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

    expanded = list(current_passages) + new_passages
    return expanded, new_passages, queries



@torch.inference_mode()
def extract_bridges_from_answer(model, tokenizer, context, question, answer_text, n_candidates=3):
    """Extract bridge entity candidates from an existing answer by finding tokens
    where the model is least certain or predicts alternatives.
    
    This is answer-conditioned: it uses the actual decoded answer, not a fresh canvas.
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    
    # Build prompt with the answer filled in
    prompt = short_user_prompt(context, question)
    msg = [{"role": "user", "content": prompt}]
    text_tmpl = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(text_tmpl, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    
    if not answer_ids:
        return []
    
    n_prefix = len(prefix_ids)
    full_ids = prefix_ids + answer_ids
    x = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(x)
    
    # Forward pass with full answer visible
    out = model(x, attention_mask=attn)
    logits = prepare_logits(out.logits)
    
    # For each answer position, check: what does the model predict vs what's committed?
    # High entropy + different top prediction = likely bridge entity
    candidates = []
    for i, tid in enumerate(answer_ids):
        pos = n_prefix + i
        pos_logits = logits[0, pos]
        pos_probs = F.softmax(pos_logits, dim=-1)
        
        # Entropy at this position
        entropy = -(pos_probs * torch.log(pos_probs + 1e-10)).sum().item()
        
        # Top prediction
        top_id = pos_probs.argmax().item()
        top_prob = pos_probs[top_id].item()
        committed_prob = pos_probs[tid].item()
        
        # Score: high entropy AND model disagrees with committed token
        disagreement = 1.0 - committed_prob
        score = entropy * disagreement
        
        # Get the top-k alternative tokens at this position
        topk_probs, topk_ids = torch.topk(pos_probs, min(5, len(pos_probs)))
        alternatives = []
        for p, t in zip(topk_probs.tolist(), topk_ids.tolist()):
            alt_text = tokenizer.decode([t], skip_special_tokens=True).strip()
            if alt_text and len(alt_text) > 1 and t != tid:
                alternatives.append({"text": alt_text, "prob": p})
        
        candidates.append({
            "position": i,
            "committed": tokenizer.decode([tid], skip_special_tokens=True).strip(),
            "entropy": entropy,
            "score": score,
            "alternatives": alternatives[:3],
        })
    
    # Sort by score (most uncertain + most disagreement)
    candidates.sort(key=lambda c: -c["score"])
    
    # Extract SPAN-LEVEL bridge candidates by expanding high-score positions
    # into full entity spans from the answer text
    def expand_to_span(pos, answer_ids, tokenizer, window=4):
        """Expand a single position into a multi-token span."""
        start = max(0, pos - window)
        end = min(len(answer_ids), pos + window + 1)
        span_ids = answer_ids[start:end]
        span_text = tokenizer.decode(span_ids, skip_special_tokens=True).strip()
        # Also get a narrower span centered on the position
        narrow_start = max(0, pos - 1)
        narrow_end = min(len(answer_ids), pos + 2)
        narrow_ids = answer_ids[narrow_start:narrow_end]
        narrow_text = tokenizer.decode(narrow_ids, skip_special_tokens=True).strip()
        return span_text, narrow_text
    
    result = []
    seen = set()
    # First: use the full answer text as one candidate (strongest query)
    if answer_text and answer_text.lower() not in seen:
        result.append({"text": answer_text, "prob": 1.0, "position": -1, "type": "full_answer"})
        seen.add(answer_text.lower())
    
    for c in candidates[:n_candidates * 3]:
        pos = c["position"]
        # Expand to span
        wide_span, narrow_span = expand_to_span(pos, answer_ids, tokenizer)
        
        # Use the narrow span (3 tokens around the uncertain position)
        if narrow_span and len(narrow_span) > 2 and narrow_span.lower() not in seen:
            result.append({"text": narrow_span, "prob": c["score"], "position": pos, "type": "bridge_span"})
            seen.add(narrow_span.lower())
        
        # Also use alternatives at this position as standalone queries
        for alt in c["alternatives"][:1]:  # Top-1 alternative only
            alt_span = alt["text"]
            if alt_span and len(alt_span) > 2 and alt_span.lower() not in seen:
                result.append({"text": alt_span, "prob": alt["prob"], "position": pos, "type": "alternative"})
                seen.add(alt_span.lower())
        
        if len(result) >= n_candidates + 1:  # +1 for the full answer
            break
    
    return result[:n_candidates + 1]


def iterative_method(model, tokenizer, retriever, question, initial_passages,
                     method="ieamd", max_rounds=3, steps=32, n_tokens=32,
                     temperature=0.0, gamma_mult=0.5, gamma_cap=8.0,
                     n_candidates=3, expand_top_k=3):
    """Run iterative EAMD or iterative Pool.

    Each round:
    1. Decode current answer (with guidance for ieamd, without for ipool)
    2. Extract bridge candidates from token distribution
    3. Expand evidence using answer + bridges
    4. Stop if no new passages or answer unchanged
    """
    current_passages = list(initial_passages)
    old_context = "\n\n".join(current_passages)

    # Round 0: quick seed
    seed_answer, seed_tokens, _ = short_generate(
        model, tokenizer, old_context, question,
        steps=16, n_tokens=16, temperature=temperature
    )

    # Extract initial bridge candidates
    bridge_cands = extract_candidates(model, tokenizer, old_context + '\n\nPreliminary answer: ' + seed_answer, question, n_candidates)

    # Initial expansion
    current_passages, new_p, _ = expand_evidence_round(
        retriever, question, seed_answer, bridge_cands,
        current_passages, expand_top_k
    )
    new_context = "\n\n".join(current_passages)

    round_log = []
    prev_answer = seed_answer

    for r in range(max_rounds):
        # Decode
        if method == "ieamd" and gamma_mult > 0:
            answer, answer_tokens, stats = decode_with_guidance(
                model, tokenizer, old_context, new_context, question,
                steps=steps, n_tokens=n_tokens, temperature=temperature,
                gamma_mult=gamma_mult, gamma_cap=gamma_cap,
            )
        else:
            answer, answer_tokens = simple_decode(
                model, tokenizer, new_context, question,
                steps=steps, n_tokens=n_tokens, temperature=temperature,
            )
            stats = {}

        # Check stopping: answer unchanged
        answer_stable = (answer.strip().lower() == prev_answer.strip().lower())

        # Extract new bridge candidates from this round's token distribution
        bridge_cands = extract_bridges_from_answer(model, tokenizer, new_context, question, answer, n_candidates)

        # Expand evidence
        old_context = new_context
        current_passages, new_p, queries = expand_evidence_round(
            retriever, question, answer, bridge_cands,
            current_passages, expand_top_k
        )
        new_context = "\n\n".join(current_passages)

        round_log.append({
            "round": r,
            "answer": answer,
            "new_passages": len(new_p),
            "total_passages": len(current_passages),
            "n_queries": len(queries),
            "answer_stable": answer_stable,
            "stats": stats,
        })

        # Stop if no new evidence AND answer stable
        if len(new_p) == 0 and answer_stable:
            break
        # Note: we only stop when BOTH no new passages AND answer stable
        # (removed unconditional stop on no new passages alone)

        prev_answer = answer

    return answer, {
        "rounds": len(round_log),
        "total_passages": len(current_passages),
        "round_log": round_log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--gamma_mult", type=float, default=0.5)
    parser.add_argument("--gamma_cap", type=float, default=8.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== Iterative EAMD (iEAMD) ===", flush=True)
    print(f"Model: {args.model}, Rounds: {args.max_rounds}, Steps: {args.steps}, Tokens: {args.answer_tokens}", flush=True)

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

    # Methods to compare
    methods = ["baseline", "pool", "ipool", "ieamd"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    print(f"Running: {methods}", flush=True)
    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold}

        # Baseline: single-shot decode from C0
        baseline_ans, _ = simple_decode(model, tokenizer, old_ctx, qtext,
                                         steps=args.steps, n_tokens=args.answer_tokens,
                                         temperature=args.temperature)

        # Pool: single-round expansion then decode (no guidance)
        pool_ans, pool_stats = iterative_method(
            model, tokenizer, retriever, qtext, initial,
            method="ipool", max_rounds=1, steps=args.steps, n_tokens=args.answer_tokens,
            temperature=args.temperature, gamma_mult=0.0,
            n_candidates=args.n_candidates, expand_top_k=args.expand_top_k,
        )

        # iPool: multi-round expansion, no guidance
        ipool_ans, ipool_stats = iterative_method(
            model, tokenizer, retriever, qtext, initial,
            method="ipool", max_rounds=args.max_rounds, steps=args.steps, n_tokens=args.answer_tokens,
            temperature=args.temperature, gamma_mult=0.0,
            n_candidates=args.n_candidates, expand_top_k=args.expand_top_k,
        )

        # iEAMD: multi-round expansion WITH guidance
        ieamd_ans, ieamd_stats = iterative_method(
            model, tokenizer, retriever, qtext, initial,
            method="ieamd", max_rounds=args.max_rounds, steps=args.steps, n_tokens=args.answer_tokens,
            temperature=args.temperature, gamma_mult=args.gamma_mult, gamma_cap=args.gamma_cap,
            n_candidates=args.n_candidates, expand_top_k=args.expand_top_k,
        )

        elapsed = time.time() - tq

        for method, ans in [("baseline", baseline_ans), ("pool", pool_ans),
                            ("ipool", ipool_ans), ("ieamd", ieamd_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        row["ipool_stats"] = ipool_stats
        row["ieamd_stats"] = ieamd_stats
        row["elapsed"] = round(elapsed, 2)
        results.append(row)

        log_this = (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1
        if log_this:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s) "
                  f"ipool_rounds={ipool_stats['rounds']} ieamd_rounds={ieamd_stats['rounds']}", flush=True)
            for m in methods:
                print(f"  {m:12s} {row[m]['answer'][:45]:45s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

    n = len(questions)
    t_total = time.time() - t_start

    print(f"\n{'Method':<12s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 34)
    summary = {}
    for m in methods:
        s = {k: round(v / n, 4) for k, v in totals[m].items()}
        summary[m] = s
        print(f"{m:<12s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")

    avg_ipool_rounds = sum(r["ipool_stats"]["rounds"] for r in results) / n
    avg_ieamd_rounds = sum(r["ieamd_stats"]["rounds"] for r in results) / n
    print(f"\nAvg rounds: iPool={avg_ipool_rounds:.1f} iEAMD={avg_ieamd_rounds:.1f}")
    print(f"Total: {t_total:.1f}s ({t_total/n:.1f}s/q)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                    "timing": {"total_sec": round(t_total, 1)},
                    "avg_rounds": {"ipool": avg_ipool_rounds, "ieamd": avg_ieamd_rounds}},
                   f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
