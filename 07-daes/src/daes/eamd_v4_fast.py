"""EAMD v4 Fast: batched per-round decoding for 6 methods.

Goal: preserve method logic while removing serial per-method denoising.
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever,
    build_short_prompt,
    build_short_pair,
    build_short_cond_and_prior,
    prepare_logits,
    get_mask_id,
    decode_answer,
    compute_f1,
    compute_w_t,
    compute_signal_and_scale,
    short_generate,
    spread_generate_shared,
    aram_generate_shared,
    short_user_prompt,
    QUESTION_FILES,
)
from dgmqr import extract_candidates
from eamd_v4_trajectory import v4_trajectory_decode
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


def pad_and_batch(sequences, pad_id, device):
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


def compute_contradiction_score(innovation, trajectory_dir, eps=1e-8):
    if trajectory_dir.norm() < eps or innovation.norm() < eps:
        return torch.zeros(innovation.shape[0], device=innovation.device)
    cos_sim = F.cosine_similarity(innovation, trajectory_dir, dim=-1)
    return torch.clamp(-cos_sim, min=0.0)


def compute_v4_guidance(logits_new, logits_old, agg_logits, logits_base, w_t, kappa=1.0, eps=1e-6):
    innovation = logits_new - logits_old
    trajectory_dir = agg_logits - logits_base
    contradiction = compute_contradiction_score(innovation, trajectory_dir, eps)

    log_p_new = F.log_softmax(logits_new, dim=-1)
    log_p_old = F.log_softmax(logits_old, dim=-1)
    p_new = log_p_new.exp()
    r = log_p_new - log_p_old
    signal = (p_new * r).sum(dim=-1)
    noise = (p_new * (r - signal.unsqueeze(-1)).pow(2)).sum(dim=-1)

    alpha = w_t * signal / (noise + kappa * contradiction + eps)
    alpha = torch.clamp(alpha, min=0.0, max=8.0)
    return alpha, signal, noise, contradiction


@torch.inference_mode()
def extract_bridges_from_answer_batch(model, tokenizer, items, n_candidates=3):
    """Batch version of extract_bridges_from_answer.

    items: list of (context, question, answer_text)
    """
    device = model.device
    outputs = [[] for _ in items]
    if not items:
        return outputs

    valid = []
    seqs = []
    answer_id_lists = []
    prefix_lens = []
    for idx, (context, question, answer_text) in enumerate(items):
        if not answer_text:
            continue
        prompt = short_user_prompt(context, question)
        msg = [{"role": "user", "content": prompt}]
        text_tmpl = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer.encode(text_tmpl, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        if not answer_ids:
            continue
        valid.append((idx, answer_text))
        seqs.append(prefix_ids + answer_ids)
        answer_id_lists.append(answer_ids)
        prefix_lens.append(len(prefix_ids))

    if not seqs:
        return outputs

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    x, attn = pad_and_batch(seqs, pad_id, device)
    out = model(x, attention_mask=attn)
    logits = prepare_logits(out.logits)

    def expand_to_span(pos, answer_ids, tokenizer, window=4):
        wide_start = max(0, pos - window)
        wide_end = min(len(answer_ids), pos + window + 1)
        wide = tokenizer.decode(answer_ids[wide_start:wide_end], skip_special_tokens=True).strip()
        narrow_start = max(0, pos - 1)
        narrow_end = min(len(answer_ids), pos + 2)
        narrow = tokenizer.decode(answer_ids[narrow_start:narrow_end], skip_special_tokens=True).strip()
        return wide, narrow

    for batch_idx, ((orig_idx, answer_text), answer_ids, n_prefix) in enumerate(zip(valid, answer_id_lists, prefix_lens)):
        candidates = []
        for i, tid in enumerate(answer_ids):
            pos = n_prefix + i
            pos_logits = logits[batch_idx, pos]
            pos_probs = F.softmax(pos_logits, dim=-1)
            entropy = -(pos_probs * torch.log(pos_probs + 1e-10)).sum().item()
            committed_prob = pos_probs[tid].item()
            score = entropy * (1.0 - committed_prob)

            topk_probs, topk_ids = torch.topk(pos_probs, min(5, len(pos_probs)))
            alternatives = []
            for p, t in zip(topk_probs.tolist(), topk_ids.tolist()):
                if t == tid:
                    continue
                alt_text = tokenizer.decode([t], skip_special_tokens=True).strip()
                if alt_text and len(alt_text) > 1:
                    alternatives.append({"text": alt_text, "prob": p})

            candidates.append(
                {
                    "position": i,
                    "score": score,
                    "alternatives": alternatives[:3],
                }
            )

        candidates.sort(key=lambda c: -c["score"])
        result = []
        seen = set()

        if answer_text and answer_text.lower() not in seen:
            result.append({"text": answer_text, "prob": 1.0, "position": -1, "type": "full_answer"})
            seen.add(answer_text.lower())

        for c in candidates[: n_candidates * 3]:
            pos = c["position"]
            _, narrow_span = expand_to_span(pos, answer_ids, tokenizer)
            if narrow_span and len(narrow_span) > 2 and narrow_span.lower() not in seen:
                result.append({"text": narrow_span, "prob": c["score"], "position": pos, "type": "bridge_span"})
                seen.add(narrow_span.lower())

            for alt in c["alternatives"][:1]:
                alt_text = alt["text"]
                if alt_text and len(alt_text) > 2 and alt_text.lower() not in seen:
                    result.append({"text": alt_text, "prob": alt["prob"], "position": pos, "type": "alternative"})
                    seen.add(alt_text.lower())

            if len(result) >= n_candidates + 1:
                break

        outputs[orig_idx] = result[: n_candidates + 1]

    return outputs


def expand_evidence(retriever, question, answer, bridge_cands, current_passages, expand_top_k=3):
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


@torch.inference_mode()
def short_generate_batch(model, tokenizer, contexts, questions, steps=16, n_tokens=16, temperature=0.0):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    seqs = []
    prefixes = []
    for context, question in zip(contexts, questions):
        prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
        seqs.append(prefix_ids + [mask_id] * n_tokens)
        prefixes.append(n_prefix)

    x, attn = pad_and_batch(seqs, pad_id, device)
    remaining = [n_tokens] * len(seqs)
    k_per_step = max(1, math.ceil(n_tokens / steps))

    for step in range(steps):
        if all(r <= 0 for r in remaining):
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)

        for i in range(len(seqs)):
            if remaining[i] <= 0:
                continue
            pf = prefixes[i]
            masked_local = (x[i, pf:pf + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_local) == 0:
                remaining[i] = 0
                continue
            full_pos = masked_local + pf
            token_logits = logits[i, full_pos]
            confidence, x0 = sample_tokens(token_logits, temperature=temperature, neg_entropy=True)
            n_commit = min(k_per_step, remaining[i])
            if step == steps - 1:
                n_commit = remaining[i]
            n_commit = min(n_commit, len(confidence))
            _, topk = torch.topk(confidence, n_commit)
            chosen = masked_local[topk]
            x[i, chosen + pf] = x0[topk]
            remaining[i] -= len(topk)

    answers = []
    for i in range(len(seqs)):
        pf = prefixes[i]
        answers.append(decode_answer(tokenizer, x[i, pf:pf + n_tokens]))
    return answers


@torch.inference_mode()
def unified_round_decode(
    model,
    tokenizer,
    question,
    baseline_context,
    pool_context,
    ipool_context,
    iaram_context,
    steps=32,
    n_tokens=32,
    temperature=0.0,
):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    rows = []
    prefixes = []
    names = []

    # 0 baseline
    prefix_ids, n_prefix = build_short_prompt(tokenizer, baseline_context, question)
    rows.append(prefix_ids + [mask_id] * n_tokens)
    prefixes.append(n_prefix)
    names.append("baseline")

    # 1 pool
    prefix_ids, n_prefix = build_short_prompt(tokenizer, pool_context, question)
    rows.append(prefix_ids + [mask_id] * n_tokens)
    prefixes.append(n_prefix)
    names.append("pool")

    # 2 ipool
    prefix_ids, n_prefix = build_short_prompt(tokenizer, ipool_context, question)
    rows.append(prefix_ids + [mask_id] * n_tokens)
    prefixes.append(n_prefix)
    names.append("ipool")

    # 3/4 iaram cond/prior
    cond_ids, prior_ids, n_prefix = build_short_cond_and_prior(tokenizer, iaram_context, question, n_tokens)
    rows.append(cond_ids)
    prefixes.append(n_prefix)
    names.append("iaram_cond")
    rows.append(prior_ids)
    prefixes.append(n_prefix)
    names.append("iaram_prior")

    x, attn = pad_and_batch(rows, pad_id, device)

    k_per_step = max(1, math.ceil(n_tokens / steps))
    remaining = {name: n_tokens for name in ["baseline", "pool", "ipool", "iaram"]}
    iaram_stats = {"lambda": []}

    row_ix = {name: idx for idx, name in enumerate(names)}
    all_rows = list(range(len(names)))

    for step in range(steps):
        if all(v <= 0 for v in remaining.values()):
            break

        # Keep prior answer canvas synced with the conditional branch.
        cond_i = row_ix["iaram_cond"]
        prior_i = row_ix["iaram_prior"]
        pf_iaram = prefixes[cond_i]
        x[prior_i, pf_iaram:pf_iaram + n_tokens] = x[cond_i, pf_iaram:pf_iaram + n_tokens]

        out_all = model(x[all_rows], attention_mask=attn[all_rows])
        logits_all = prepare_logits(out_all.logits)

        # Plain rows: baseline, pool, ipool
        for method in ["baseline", "pool", "ipool"]:
            if remaining[method] <= 0:
                continue
            i = row_ix[method]
            pf = prefixes[i]
            masked_local = (x[i, pf:pf + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_local) == 0:
                remaining[method] = 0
                continue
            full_pos = masked_local + pf
            guided = logits_all[i, full_pos]
            confidence, x0 = sample_tokens(guided, temperature=temperature, neg_entropy=True)
            n_commit = min(k_per_step, remaining[method])
            if step == steps - 1:
                n_commit = remaining[method]
            n_commit = min(n_commit, len(confidence))
            _, topk = torch.topk(confidence, n_commit)
            chosen = masked_local[topk]
            x[i, chosen + pf] = x0[topk]
            remaining[method] -= len(topk)

        # iARAM
        if remaining["iaram"] > 0:
            pf = prefixes[cond_i]
            masked_local = (x[cond_i, pf:pf + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_local) == 0:
                remaining["iaram"] = 0
            else:
                full_pos = masked_local + pf
                logits_cond = logits_all[cond_i, full_pos]
                logits_prior = logits_all[prior_i, full_pos]
                _, w_t = compute_w_t(len(masked_local), n_tokens)
                _, _, lam, _ = compute_signal_and_scale(
                    logits_cond,
                    logits_prior,
                    lambda_max=1.0,
                    beta=0.5,
                    eps=1e-6,
                    schedule=w_t,
                )
                guided = logits_prior + lam.unsqueeze(-1) * (logits_cond - logits_prior)
                iaram_stats["lambda"].append(lam.mean().item())
                confidence, x0 = sample_tokens(guided, temperature=temperature, neg_entropy=True)
                n_commit = min(k_per_step, remaining["iaram"])
                if step == steps - 1:
                    n_commit = remaining["iaram"]
                n_commit = min(n_commit, len(confidence))
                _, topk = torch.topk(confidence, n_commit)
                chosen = masked_local[topk]
                tokens = x0[topk]
                x[cond_i, chosen + pf] = tokens
                x[prior_i, chosen + pf] = tokens
                remaining["iaram"] -= len(topk)

    answers = {}
    for method in ["baseline", "pool", "ipool"]:
        i = row_ix[method]
        pf = prefixes[i]
        answers[method] = decode_answer(tokenizer, x[i, pf:pf + n_tokens])
    answers["iaram"] = decode_answer(tokenizer, x[row_ix["iaram_cond"], pf_iaram:pf_iaram + n_tokens])

    stats = {
        "iaram": {"mean_lambda": sum(iaram_stats["lambda"]) / len(iaram_stats["lambda"]) if iaram_stats["lambda"] else 0.0},
    }
    return answers, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print("=== EAMD v4 Fast ===", flush=True)
    print(f"Model: {args.model}, Rounds: {args.max_rounds}, Steps: {args.steps}", flush=True)

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

    print("Batch initial retrieval...", flush=True)
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    print("Batch seed generation...", flush=True)
    seed_answers = short_generate_batch(
        model,
        tokenizer,
        ["\n\n".join(p) for p in all_initial],
        [q["question"] for q in questions],
        steps=16,
        n_tokens=16,
        temperature=args.temperature,
    )
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool", "ipool", "ispread", "iaram", "v4_trajectory"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)
        seed_ans = seed_answers[qi]

        # pool: keep the original one-shot generic bridge extractor
        pool_bridge_cands = extract_candidates(model, tokenizer, old_ctx, qtext, args.n_candidates)
        pool_passages, _ = expand_evidence(retriever, qtext, seed_ans, pool_bridge_cands, initial, args.expand_top_k)
        pool_ctx = "\n\n".join(pool_passages)

        # iterative methods share round-0 answer-conditioned expansion from the seed answer
        seed_bridge = extract_bridges_from_answer_batch(
            model, tokenizer, [(old_ctx, qtext, seed_ans)], n_candidates=args.n_candidates
        )[0]
        shared_round0_passages, shared_new = expand_evidence(
            retriever, qtext, seed_ans, seed_bridge, initial, args.expand_top_k
        )
        round0_ctx = "\n\n".join(shared_round0_passages)

        # v4 C0 anchor
        mask_id = get_mask_id(tokenizer)
        c0_prefix_ids, c0_pf = build_short_prompt(tokenizer, old_ctx, qtext)
        c0_full = c0_prefix_ids + [mask_id] * args.answer_tokens
        c0_x = torch.tensor([c0_full], dtype=torch.long, device=model.device)
        c0_attn = torch.ones_like(c0_x)
        c0_logits = prepare_logits(model(c0_x, attention_mask=c0_attn).logits)[0, c0_pf:c0_pf + args.answer_tokens]

        answers0, round0_basic_stats = unified_round_decode(
            model,
            tokenizer,
            question=qtext,
            baseline_context=old_ctx,
            pool_context=pool_ctx,
            ipool_context=round0_ctx,
            iaram_context=round0_ctx,
            steps=args.steps,
            n_tokens=args.answer_tokens,
            temperature=args.temperature,
        )

        ispread_ans, round0_ispread_stats = spread_generate_shared(
            model,
            tokenizer,
            round0_ctx,
            qtext,
            steps=args.steps,
            n_tokens=args.answer_tokens,
            temperature=args.temperature,
        )
        v4_ans, v4_agg, _, round0_v4_stats = v4_trajectory_decode(
            model,
            tokenizer,
            old_ctx,
            round0_ctx,
            qtext,
            agg_logits_prev=c0_logits,
            logits_base=c0_logits,
            steps=args.steps,
            n_tokens=args.answer_tokens,
            temperature=args.temperature,
            kappa=args.kappa,
        )

        baseline_ans = answers0["baseline"]
        pool_ans = answers0["pool"]
        ipool_ans = answers0["ipool"]
        iaram_ans = answers0["iaram"]

        ipool_passages = list(shared_round0_passages)
        ispread_passages = list(shared_round0_passages)
        iaram_passages = list(shared_round0_passages)
        v4_passages = list(shared_round0_passages)

        ipool_ctx = round0_ctx
        ispread_ctx = round0_ctx
        iaram_ctx = round0_ctx
        v4_prev_ctx = round0_ctx

        prev_ipool_ans = ipool_ans
        prev_ispread_ans = ispread_ans
        prev_iaram_ans = iaram_ans
        prev_v4_ans = v4_ans

        v4_round_stats = [{"round": 0, "new_passages": len(shared_new), **round0_v4_stats}]

        for r in range(1, args.max_rounds):
            batch_bridge = extract_bridges_from_answer_batch(
                model,
                tokenizer,
                [
                    (ipool_ctx, qtext, prev_ipool_ans),
                    (ispread_ctx, qtext, prev_ispread_ans),
                    (iaram_ctx, qtext, prev_iaram_ans),
                    (v4_prev_ctx, qtext, prev_v4_ans),
                ],
                n_candidates=args.n_candidates,
            )
            ipool_bc, ispread_bc, iaram_bc, v4_bc = batch_bridge

            ipool_passages, ipool_new = expand_evidence(retriever, qtext, prev_ipool_ans, ipool_bc, ipool_passages, args.expand_top_k)
            ispread_passages, ispread_new = expand_evidence(retriever, qtext, prev_ispread_ans, ispread_bc, ispread_passages, args.expand_top_k)
            iaram_passages, iaram_new = expand_evidence(retriever, qtext, prev_iaram_ans, iaram_bc, iaram_passages, args.expand_top_k)
            v4_passages, v4_new = expand_evidence(retriever, qtext, prev_v4_ans, v4_bc, v4_passages, args.expand_top_k)

            new_answers, round_basic_stats = unified_round_decode(
                model,
                tokenizer,
                question=qtext,
                baseline_context=old_ctx,
                pool_context=pool_ctx,
                ipool_context="\n\n".join(ipool_passages),
                iaram_context="\n\n".join(iaram_passages),
                steps=args.steps,
                n_tokens=args.answer_tokens,
                temperature=args.temperature,
            )
            ispread_ans, round_ispread_stats = spread_generate_shared(
                model,
                tokenizer,
                "\n\n".join(ispread_passages),
                qtext,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                temperature=args.temperature,
            )
            v4_ans, v4_agg, _, round_v4_stats = v4_trajectory_decode(
                model,
                tokenizer,
                v4_prev_ctx,
                "\n\n".join(v4_passages),
                qtext,
                agg_logits_prev=v4_agg,
                logits_base=c0_logits,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                temperature=args.temperature,
                kappa=args.kappa,
            )

            ipool_ans = new_answers["ipool"]
            iaram_ans = new_answers["iaram"]
            v4_round_stats.append({"round": r, "new_passages": len(v4_new), **round_v4_stats})

            ipool_stop = len(ipool_new) == 0 and ipool_ans.strip().lower() == prev_ipool_ans.strip().lower()
            ispread_stop = len(ispread_new) == 0 and ispread_ans.strip().lower() == prev_ispread_ans.strip().lower()
            iaram_stop = len(iaram_new) == 0 and iaram_ans.strip().lower() == prev_iaram_ans.strip().lower()
            v4_stop = len(v4_new) == 0 and v4_ans.strip().lower() == prev_v4_ans.strip().lower()

            ipool_ctx = "\n\n".join(ipool_passages)
            ispread_ctx = "\n\n".join(ispread_passages)
            iaram_ctx = "\n\n".join(iaram_passages)
            v4_prev_ctx = "\n\n".join(v4_passages)
            prev_ipool_ans = ipool_ans
            prev_ispread_ans = ispread_ans
            prev_iaram_ans = iaram_ans
            prev_v4_ans = v4_ans

            if ipool_stop and ispread_stop and iaram_stop and v4_stop:
                break

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext,
            "gold": gold,
        }

        for method, ans in [
            ("baseline", baseline_ans),
            ("pool", pool_ans),
            ("ipool", ipool_ans),
            ("ispread", ispread_ans),
            ("iaram", iaram_ans),
            ("v4_trajectory", v4_ans),
        ]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        row["v4_stats"] = v4_round_stats
        row["elapsed"] = round(time.time() - tq, 2)
        results.append(row)

        if (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({row['elapsed']:.1f}s)", flush=True)
            for m in methods:
                print(f"  {m:16s} {row[m]['answer'][:40]:40s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()} for m in methods}
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "results": results,
                        "config": vars(args),
                        "timing": {"elapsed_sec": round(time.time() - t_start, 1)},
                    },
                    f,
                    indent=2,
                )

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\n{'Method':<16s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 38)
    for m in methods:
        s = summary[m]
        print(f"{m:<16s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")
    print(f"\nTotal: {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, n):.1f}s/q)")


if __name__ == "__main__":
    main()
