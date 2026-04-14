"""EAMD v4: Trajectory-Consistent Contradiction-Aware Guidance.
Multi-round iterative retrieval with persistent trajectory posterior.

Run: python -u src/daes/eamd_v4_trajectory.py --model dream --dataset musique --n_questions 50
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    spread_generate_shared, aram_generate_shared,
    Wiki18Retriever, build_short_pair, prepare_logits,
    get_mask_id, decode_answer, compute_f1, compute_w_t,
    compute_signal_and_scale, short_generate, short_user_prompt,
    QUESTION_FILES,
)
from dgmqr import extract_candidates
from eamd_iterative import extract_bridges_from_answer
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


# ---------------------------------------------------------------------------
# V4: trajectory-consistent contradiction-aware guidance
# ---------------------------------------------------------------------------

def compute_contradiction_score(innovation, trajectory_dir, eps=1e-8):
    """Compute contradiction score: negative cosine alignment between
    innovation vector and accumulated trajectory direction.

    c = max(0, -cos(d, g))

    Returns 0 when aligned (consistent), positive when opposing (contradictory).
    """
    if trajectory_dir.norm() < eps or innovation.norm() < eps:
        return torch.zeros(innovation.shape[0], device=innovation.device)

    # Per-position cosine similarity
    cos_sim = F.cosine_similarity(innovation, trajectory_dir, dim=-1)
    contradiction = torch.clamp(-cos_sim, min=0.0)
    return contradiction


def compute_v4_guidance(logits_new, logits_old, agg_logits, logits_base,
                        w_t, kappa=1.0, eps=1e-6):
    """Compute v4 trajectory-consistent contradiction-aware guidance scale.

    alpha = w_t * S / (N + kappa * c + eps)

    Where:
    - S = KL(p_new || p_old) = evidence signal
    - N = Var_{p_new}(log p_new/p_old) = evidence noise
    - c = contradiction score from trajectory
    """
    # Innovation vector
    innovation = logits_new - logits_old  # [n_positions, vocab]

    # Trajectory direction
    trajectory_dir = agg_logits - logits_base  # [n_positions, vocab]

    # Contradiction score
    contradiction = compute_contradiction_score(innovation, trajectory_dir, eps)

    # Signal: KL(p_new || p_old)
    log_p_new = F.log_softmax(logits_new, dim=-1)
    log_p_old = F.log_softmax(logits_old, dim=-1)
    p_new = log_p_new.exp()
    r = log_p_new - log_p_old
    signal = (p_new * r).sum(dim=-1)  # KL divergence per position

    # Noise: Var_{p_new}(r)
    r_mean = signal.unsqueeze(-1)  # E[r] = KL
    noise = (p_new * (r - r_mean).pow(2)).sum(dim=-1)

    # Guidance scale
    alpha = w_t * signal / (noise + kappa * contradiction + eps)
    alpha = torch.clamp(alpha, min=0.0, max=8.0)

    return alpha, signal, noise, contradiction


def compute_aram_guidance(logits_new, logits_old, w_t, eps=1e-6):
    """Compute raw ARAM-style SNR guidance (no trajectory memory).

    alpha = w_t * S / (N + eps)

    Same signal/noise as v4 but without contradiction penalty.
    This is the fair ablation baseline: iterative retrieval + per-round SNR
    guidance but no persistent trajectory posterior.
    """
    # Signal: KL(p_new || p_old)
    log_p_new = F.log_softmax(logits_new, dim=-1)
    log_p_old = F.log_softmax(logits_old, dim=-1)
    p_new = log_p_new.exp()
    r = log_p_new - log_p_old
    signal = (p_new * r).sum(dim=-1)

    # Noise: Var_{p_new}(r)
    r_mean = signal.unsqueeze(-1)
    noise = (p_new * (r - r_mean).pow(2)).sum(dim=-1)

    alpha = w_t * signal / (noise + eps)
    alpha = torch.clamp(alpha, min=0.0, max=8.0)

    return alpha, signal, noise


# ---------------------------------------------------------------------------
# Decode routines
# ---------------------------------------------------------------------------

@torch.inference_mode()
def v4_trajectory_decode(model, tokenizer, old_context, new_context, question,
                         agg_logits_prev, logits_base,
                         steps=32, n_tokens=32, temperature=0.0,
                         kappa=1.0):
    """One round of v4 trajectory-consistent decode.

    Uses aggregated logits from previous rounds as trajectory memory.
    Returns: answer, new aggregated logits, round logits, stats
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)

    c1_ids, c0_ids, n_prefix = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)
    x_new = torch.tensor([c1_ids], dtype=torch.long, device=device)
    x_old = torch.tensor([c0_ids], dtype=torch.long, device=device)
    attn_new = torch.ones((1, len(c1_ids)), dtype=torch.long, device=device)
    attn_old = torch.ones((1, len(c0_ids)), dtype=torch.long, device=device)

    # Initialize aggregated logits for this round
    current_agg = agg_logits_prev.clone() if agg_logits_prev is not None else None
    current_base = logits_base.clone() if logits_base is not None else None

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    alphas = []
    contradictions = []
    signals = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_new[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + n_prefix

        # Batched forward pass: old + new evidence
        x_pair = torch.cat([x_old, x_new], dim=0)
        attn_pair = torch.cat([attn_old, attn_new], dim=0)
        out = model(x_pair, attention_mask=attn_pair)
        logits_pair = prepare_logits(out.logits)
        l_old = logits_pair[0, full_pos]  # [n_masked, vocab]
        l_new = logits_pair[1, full_pos]  # [n_masked, vocab]

        _, w_t = compute_w_t(len(masked_local), n_tokens)

        if current_agg is not None and current_base is not None:
            # V4 trajectory guidance
            agg_at_pos = current_agg[masked_local]  # [n_masked, vocab]
            base_at_pos = current_base[masked_local]  # [n_masked, vocab]

            alpha, sig, noi, contra = compute_v4_guidance(
                l_new, l_old, agg_at_pos, base_at_pos,
                w_t=w_t, kappa=kappa,
            )

            # Update aggregated logits: bar_ell_r = bar_ell_{r-1} + alpha * (ell_r - ell_{r-1})
            innovation = l_new - l_old
            updated_agg = agg_at_pos + alpha.unsqueeze(-1) * innovation

            # Decode from aggregated logits
            guided = updated_agg

            alphas.append(alpha.mean().item())
            contradictions.append(contra.mean().item())
            signals.append(sig.mean().item())
        else:
            # First round: no trajectory memory, just use new logits
            guided = l_new

        confidence, x0 = sample_tokens(guided, temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        chosen = masked_local[topk]
        x_new[0, chosen + n_prefix] = x0[topk]
        x_old[0, chosen + n_prefix] = x0[topk]

        # Update aggregated logits at committed positions
        if current_agg is not None:
            current_agg[chosen] = guided[topk].to(current_agg.dtype)

        remaining -= len(topk)

    answer_tokens = x_new[0, n_prefix:n_prefix + n_tokens]
    answer = decode_answer(tokenizer, answer_tokens)

    # Compute final aggregated logits for next round
    # Do one more forward pass with the full answer to get clean logits
    out_final = model(x_new, attention_mask=attn_new)
    final_logits = prepare_logits(out_final.logits)[0, n_prefix:n_prefix + n_tokens]

    if current_agg is not None:
        # Carry forward the updated aggregation
        new_agg = current_agg.clone()
    else:
        new_agg = final_logits.clone().to(logits_base.dtype if logits_base is not None else final_logits.dtype)

    stats = {
        "mean_alpha": sum(alphas) / len(alphas) if alphas else 0.0,
        "mean_contradiction": sum(contradictions) / len(contradictions) if contradictions else 0.0,
        "mean_signal": sum(signals) / len(signals) if signals else 0.0,
    }

    return answer, new_agg, final_logits, stats


@torch.inference_mode()
def iterative_aram_decode(model, tokenizer, old_context, new_context, question,
                          steps=32, n_tokens=32, temperature=0.0):
    """One round of iterative ARAM decode: raw SNR guidance, NO trajectory memory.

    Each round independently computes alpha = w_t * S / (N + eps) and applies
    guidance as: guided = l_new + alpha * (l_new - l_old).
    No information persists across rounds -- this is the fair ablation baseline.
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)

    c1_ids, c0_ids, n_prefix = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)
    x_new = torch.tensor([c1_ids], dtype=torch.long, device=device)
    x_old = torch.tensor([c0_ids], dtype=torch.long, device=device)
    attn_new = torch.ones((1, len(c1_ids)), dtype=torch.long, device=device)
    attn_old = torch.ones((1, len(c0_ids)), dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    alphas = []
    signals = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_new[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + n_prefix

        # Batched forward pass: old + new evidence
        x_pair = torch.cat([x_old, x_new], dim=0)
        attn_pair = torch.cat([attn_old, attn_new], dim=0)
        out = model(x_pair, attention_mask=attn_pair)
        logits_pair = prepare_logits(out.logits)
        l_old = logits_pair[0, full_pos]
        l_new = logits_pair[1, full_pos]

        _, w_t = compute_w_t(len(masked_local), n_tokens)
        alpha, sig, noi = compute_aram_guidance(l_new, l_old, w_t=w_t)

        # Standard ARAM guidance: l_new + alpha * (l_new - l_old)
        guided = l_new + alpha.unsqueeze(-1) * (l_new - l_old)

        alphas.append(alpha.mean().item())
        signals.append(sig.mean().item())

        confidence, x0 = sample_tokens(guided, temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        chosen = masked_local[topk]
        x_new[0, chosen + n_prefix] = x0[topk]
        x_old[0, chosen + n_prefix] = x0[topk]
        remaining -= len(topk)

    answer_tokens = x_new[0, n_prefix:n_prefix + n_tokens]
    answer = decode_answer(tokenizer, answer_tokens)

    stats = {
        "mean_alpha": sum(alphas) / len(alphas) if alphas else 0.0,
        "mean_signal": sum(signals) / len(signals) if signals else 0.0,
    }
    return answer, stats


@torch.inference_mode()
def simple_decode(model, tokenizer, context, question, steps=32, n_tokens=32, temperature=0.0):
    """Simple decode from context (no guidance)."""
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
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


# ---------------------------------------------------------------------------
# Evidence expansion
# ---------------------------------------------------------------------------

def expand_evidence(retriever, question, answer, bridge_cands, current_passages, expand_top_k=3):
    """Expand evidence using answer + bridge candidates."""
    queries = [f"query: {question} {answer}"]
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
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== EAMD v4: Trajectory-Consistent Contradiction-Aware ===", flush=True)
    print(f"Model: {args.model}, Rounds: {args.max_rounds}, Steps: {args.steps}, kappa: {args.kappa}", flush=True)

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
    queries = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(queries, args.initial_top_k)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool", "ipool", "ispread", "iaram", "v4_trajectory"]
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

        # --- Baseline: single-shot decode from C0, no guidance ---
        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)

        # --- Pool: single-round expansion then decode (no guidance) ---
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                         steps=16, n_tokens=16, temperature=0.0)
        bridge_cands = extract_candidates(model, tokenizer, old_ctx, qtext, args.n_candidates)
        pool_passages, _ = expand_evidence(retriever, qtext, seed_ans, bridge_cands,
                                            initial, args.expand_top_k)
        pool_ctx = "\n\n".join(pool_passages)
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext,
                                  steps=args.steps, n_tokens=args.answer_tokens)

        # --- iPool: iterative expansion, no guidance ---
        ipool_passages = list(initial)
        ipool_ctx = old_ctx
        prev_ans = seed_ans
        for r in range(args.max_rounds):
            bc = extract_bridges_from_answer(
                model, tokenizer, ipool_ctx, qtext, prev_ans, args.n_candidates
            )
            ipool_passages, new_p = expand_evidence(retriever, qtext, prev_ans, bc,
                                                     ipool_passages, args.expand_top_k)
            ipool_ctx = "\n\n".join(ipool_passages)
            ipool_ans = simple_decode(model, tokenizer, ipool_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)
            if len(new_p) == 0 and ipool_ans.strip().lower() == prev_ans.strip().lower():
                break
            prev_ans = ipool_ans

        # iSPREAD (iterative expansion + SPREAD denoising each round)
        ispread_passages = list(initial)
        ispread_ctx = old_ctx
        prev_ispread_ans = seed_ans
        for r in range(args.max_rounds):
            bc = extract_bridges_from_answer(
                model, tokenizer, ispread_ctx, qtext, prev_ispread_ans, args.n_candidates
            )
            ispread_passages, new_p = expand_evidence(retriever, qtext, prev_ispread_ans, bc,
                                                       ispread_passages, args.expand_top_k)
            ispread_ctx = "\n\n".join(ispread_passages)
            ispread_ans, _ = spread_generate_shared(
                model, tokenizer, ispread_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens, temperature=args.temperature,
            )
            if len(new_p) == 0 and ispread_ans.strip().lower() == prev_ispread_ans.strip().lower():
                break
            prev_ispread_ans = ispread_ans
        
        # iARAM (iterative expansion + proper ARAM denoising: context vs prior)
        iaram_passages = list(initial)
        iaram_ctx = old_ctx
        prev_iaram_ans = seed_ans
        for r in range(args.max_rounds):
            bc = extract_bridges_from_answer(
                model, tokenizer, iaram_ctx, qtext, prev_iaram_ans, args.n_candidates
            )
            iaram_passages, new_p = expand_evidence(retriever, qtext, prev_iaram_ans, bc,
                                                     iaram_passages, args.expand_top_k)
            iaram_ctx = "\n\n".join(iaram_passages)
            iaram_ans, _, _ = aram_generate_shared(
                model, tokenizer, iaram_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens, temperature=args.temperature,
            )
            if len(new_p) == 0 and iaram_ans.strip().lower() == prev_iaram_ans.strip().lower():
                break
            prev_iaram_ans = iaram_ans
        
        v4_passages = list(initial)
        v4_prev_ctx = old_ctx
        # Initialize base logits from C0 (before any expansion)
        # This is the trajectory anchor — contradiction is measured against this
        _c0_prompt = short_user_prompt(old_ctx, qtext)
        _c0_msg = [{"role": "user", "content": _c0_prompt}]
        _c0_text = tokenizer.apply_chat_template(_c0_msg, tokenize=False, add_generation_prompt=True)
        _c0_ids = tokenizer.encode(_c0_text, add_special_tokens=False)
        _c0_full = _c0_ids + [get_mask_id(tokenizer)] * args.answer_tokens
        _c0_x = torch.tensor([_c0_full], dtype=torch.long, device=model.device)
        _c0_attn = torch.ones_like(_c0_x)
        _c0_out = model(_c0_x, attention_mask=_c0_attn)
        _c0_logits = prepare_logits(_c0_out.logits)[0, len(_c0_ids):len(_c0_ids) + args.answer_tokens]
        v4_agg_logits = _c0_logits.clone()  # Start trajectory from C0
        v4_base_logits = _c0_logits.clone()  # Anchor for contradiction detection
        prev_v4_ans = seed_ans
        v4_round_stats = []

        for r in range(args.max_rounds):
            bc = extract_bridges_from_answer(
                model, tokenizer, v4_prev_ctx, qtext, prev_v4_ans, args.n_candidates
            )
            v4_passages, new_p = expand_evidence(retriever, qtext, prev_v4_ans, bc,
                                                  v4_passages, args.expand_top_k)
            v4_ctx = "\n\n".join(v4_passages)

            v4_ans, v4_agg_logits, v4_round_logits, stats = v4_trajectory_decode(
                model, tokenizer, v4_prev_ctx, v4_ctx, qtext,
                agg_logits_prev=v4_agg_logits,
                logits_base=v4_base_logits,
                steps=args.steps, n_tokens=args.answer_tokens,
                temperature=args.temperature, kappa=args.kappa,
            )

            if v4_base_logits is None:
                v4_base_logits = v4_round_logits.clone()

            stats["round"] = r
            stats["new_passages"] = len(new_p)
            v4_round_stats.append(stats)

            if len(new_p) == 0 and v4_ans.strip().lower() == prev_v4_ans.strip().lower():
                break
            v4_prev_ctx = v4_ctx
            prev_v4_ans = v4_ans

        elapsed = time.time() - tq

        for method, ans in [("baseline", baseline_ans), ("pool", pool_ans),
                            ("ipool", ipool_ans), ("ispread", ispread_ans),
                            ("iaram", iaram_ans),
                            ("v4_trajectory", v4_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        # iaram stats tracked inside aram_generate_shared
        row["v4_stats"] = v4_round_stats
        row["elapsed"] = round(elapsed, 2)
        results.append(row)

        log_this = (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1
        if log_this:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s) "
                  f"iaram_done v4_rounds={len(v4_round_stats)}", flush=True)
            for m in methods:
                print(f"  {m:16s} {row[m]['answer'][:40]:40s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

    n = len(questions)
    t_total = time.time() - t_start

    print(f"\n{'Method':<16s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 38)
    summary = {}
    for m in methods:
        s = {k: round(v / n, 4) for k, v in totals[m].items()}
        summary[m] = s
        print(f"{m:<16s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")

    print(f"\nTotal: {t_total:.1f}s ({t_total/n:.1f}s/q)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                    "timing": {"total_sec": round(t_total, 1)}}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
