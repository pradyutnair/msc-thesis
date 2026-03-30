"""iDNMR-Filtered: Posterior-weighted passage selection under fixed context budget.

Key change from idnmr_pilot.py: instead of naive union of all retrieved passages,
score new passages by posterior-weighted relevance and keep only top-B per round.

Methods compared:
  baseline      - single decode from C0
  pool          - single-round: extract_candidates -> expand -> decode
  idnmr         - iterative with naive union (original)
  idnmr_filtered - iterative with posterior-weighted passage selection

Run: python -u src/daes/idnmr_filtered.py --model dream --dataset musique --n_questions 50
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate,
    short_user_prompt, extract_candidates_generic, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
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
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


def expand_evidence_naive(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
    """Original naive union expansion."""
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


def expand_evidence_filtered(retriever, e5_model, question, seed_answer, bridge_cands,
                             current_passages, expand_top_k=3, max_new=5, lambda_ans=0.3):
    """Posterior-weighted passage selection.
    
    For each query q_i with weight w_i, score each retrieved passage d by:
      s(d) += w_i * cos(e(q_i), e(d)) / (rank_i + 1)
    Sum across ALL queries that retrieved d (not max).
    Keep only top-max_new new passages per round.
    """
    # Build query family with posterior weights
    queries = []
    weights = []
    
    # Answer query (weight = lambda_ans)
    queries.append(f"query: {question} {seed_answer}")
    weights.append(lambda_ans)
    
    # Bridge queries (weight = normalized candidate confidence)
    total_conf = sum(c.get("init_conf", 1.0) for c in bridge_cands if isinstance(c, dict))
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        conf = cand.get("init_conf", 1.0) if isinstance(cand, dict) else 1.0
        if text and len(text) > 1:
            queries.append(f"query: {question} {text}")
            weights.append((1.0 - lambda_ans) * conf / max(total_conf, 1e-6))
    
    if not queries:
        return list(current_passages), []
    
    results = retriever.retrieve_batch(queries, expand_top_k)
    existing = set(current_passages)
    
    # Encode all queries and collect candidate passages
    q_embs = e5_model.encode(queries, normalize_embeddings=True, batch_size=32)
    
    # Collect all unique new passages with per-query rank info
    passage_query_info = {}  # passage -> [(query_idx, rank)]
    for qi, passages in enumerate(results):
        for rank, passage in enumerate(passages):
            if passage in existing:
                continue
            if passage not in passage_query_info:
                passage_query_info[passage] = []
            passage_query_info[passage].append((qi, rank))
    
    if not passage_query_info:
        return list(current_passages), []
    
    # Encode candidate passages
    candidate_passages = list(passage_query_info.keys())
    p_embs = e5_model.encode(
        [f"passage: {p[:512]}" for p in candidate_passages],
        normalize_embeddings=True,
        batch_size=32,
    )
    
    # Score: s(d) = sum over queries that retrieved d of w_i * cos(q_i, d) / (rank_i + 1)
    import numpy as np
    scores = np.zeros(len(candidate_passages))
    for pi, passage in enumerate(candidate_passages):
        for qi, rank in passage_query_info[passage]:
            cos_sim = float(np.dot(q_embs[qi], p_embs[pi]))
            scores[pi] += weights[qi] * cos_sim / (rank + 1)
    
    # Take top-max_new
    if len(candidate_passages) > max_new:
        top_indices = np.argsort(scores)[::-1][:max_new]
        new_passages = [candidate_passages[i] for i in top_indices]
    else:
        new_passages = candidate_passages
    
    return list(current_passages) + new_passages, new_passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--max_new_per_round", type=int, default=5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== iDNMR-Filtered Pilot ===", flush=True)
    print(f"Model: {args.model}, Rounds: {args.max_rounds}, max_new={args.max_new_per_round}", flush=True)

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
    print(f"  Model loaded in {time.time() - t_start:.1f}s", flush=True)

    # E5 model for passage relevance scoring (already loaded by retriever, reuse)
    e5_model = SentenceTransformer("intfloat/e5-base-v2", device="cuda:0")
    
    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    print("Batch initial retrieval...", flush=True)
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool", "idnmr", "idnmr_filtered"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # === baseline ===
        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens)

        # === pool: single-round ===
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                        steps=16, n_tokens=16, temperature=0.0)
        pool_cands = extract_candidates_generic(model, tokenizer, old_ctx, qtext, args.n_candidates)
        pool_passages, _ = expand_evidence_naive(retriever, qtext, seed_ans, pool_cands,
                                                 initial, args.expand_top_k)
        pool_ctx = "\n\n".join(pool_passages)
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        # === idnmr: naive union (original) ===
        idnmr_passages = list(initial)
        idnmr_ctx = old_ctx
        prev_idnmr_ans = seed_ans
        for r in range(args.max_rounds):
            bc = extract_candidates_generic(model, tokenizer, idnmr_ctx, qtext, args.n_candidates)
            idnmr_passages, new_p = expand_evidence_naive(retriever, qtext, prev_idnmr_ans, bc,
                                                          idnmr_passages, args.expand_top_k)
            idnmr_ctx = "\n\n".join(idnmr_passages)
            idnmr_ans = simple_decode(model, tokenizer, idnmr_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)
            if len(new_p) == 0 and idnmr_ans.strip().lower() == prev_idnmr_ans.strip().lower():
                break
            prev_idnmr_ans = idnmr_ans

        # === idnmr_filtered: posterior-weighted selection ===
        filt_passages = list(initial)
        filt_ctx = old_ctx
        prev_filt_ans = seed_ans
        filt_stats = []
        for r in range(args.max_rounds):
            bc = extract_candidates_generic(model, tokenizer, filt_ctx, qtext, args.n_candidates)
            filt_passages, new_p = expand_evidence_filtered(
                retriever, e5_model, qtext, prev_filt_ans, bc,
                filt_passages, args.expand_top_k, max_new=args.max_new_per_round
            )
            filt_ctx = "\n\n".join(filt_passages)
            filt_ans = simple_decode(model, tokenizer, filt_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens)
            filt_stats.append({"round": r, "new_passages": len(new_p), "total": len(filt_passages)})
            if len(new_p) == 0 and filt_ans.strip().lower() == prev_filt_ans.strip().lower():
                break
            prev_filt_ans = filt_ans

        elapsed = time.time() - tq
        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold, "elapsed": round(elapsed, 2),
               "pool_n_passages": len(pool_passages),
               "idnmr_n_passages": len(idnmr_passages),
               "filtered_n_passages": len(filt_passages),
               "filtered_stats": filt_stats}

        for method, ans in [("baseline", baseline_ans), ("pool", pool_ans),
                            ("idnmr", idnmr_ans), ("idnmr_filtered", filt_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        results.append(row)

        log_this = (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1
        if log_this:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s) passages: pool={len(pool_passages)} idnmr={len(idnmr_passages)} filt={len(filt_passages)}", flush=True)
            for m in methods:
                print(f"  {m:16s} {row[m]['answer'][:40]:40s} F1={row[m]['f1']:.3f}", flush=True)

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()} for m in methods}
            with open(args.output, "w") as f:
                json.dump({"summary": summary, "results": results, "config": vars(args),
                           "timing": {"elapsed_sec": round(time.time() - t_start, 1)}}, f, indent=2)

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
