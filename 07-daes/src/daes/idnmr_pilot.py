"""iDNMR Pilot: Iterative Diffusion-Native Multi-Query Retrieval.

Key difference from iPool/iSPREAD/iARAM: uses distribution-based bridge
extraction (extract_candidates) at EVERY round, not just round 0.

Methods compared:
  baseline     - single decode from C0 (no expansion)
  pool         - single-round: extract_candidates -> expand -> decode
  ipool        - iterative: extract_bridges_from_answer each round -> expand -> decode
  idnmr        - iterative: extract_candidates each round -> expand -> decode
  idnmr_2round - same as idnmr but capped at 2 rounds

Run: python -u src/daes/idnmr_pilot.py --model dream --dataset musique --n_questions 50
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/Fast-dLLM/llada")

from eamd_v2_wiki18 import (
    _neg_entropy,
    Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate,
    short_user_prompt, extract_candidates_generic, QUESTION_FILES,
)
from eamd_iterative import extract_bridges_from_answer
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18


# ---------------------------------------------------------------------------
# Simple decode (no guidance)
# ---------------------------------------------------------------------------
# Global: set by main() when --decode_backend fast-dllm
_FAST_DLLM_GEN_FN = None
_FAST_DLLM_MASK_ID = None


@torch.inference_mode()
def simple_decode(model, tokenizer, context, question, steps=32, n_tokens=32):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)

    # Fast-dLLM path for LLaDA
    if _FAST_DLLM_GEN_FN is not None:
        prompt = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        output, _ = _FAST_DLLM_GEN_FN(
            model, prompt,
            steps=steps, gen_length=n_tokens, block_length=n_tokens,
            temperature=0.0, remasking="low_confidence",
            mask_id=_FAST_DLLM_MASK_ID or mask_id,
        )
        return decode_answer(tokenizer, output[0, n_prefix:n_prefix + n_tokens])

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


# ---------------------------------------------------------------------------
# Evidence expansion
# ---------------------------------------------------------------------------
def expand_evidence(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
    """Expand evidence using seed answer + bridge candidates as retrieval queries."""
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
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12, help="Denoising steps per branch rollout (12=original, 4=fast)")
    parser.add_argument("--decode_backend", default="vanilla", choices=["vanilla", "fast-dllm"],
                        help="Use fast-dllm prefix cache for LLaDA decode (3-4x speedup)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== iDNMR Pilot ===", flush=True)
    print(f"Model: {args.model}, Rounds: {args.max_rounds}, Steps: {args.steps}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    global _FAST_DLLM_GEN_FN, _FAST_DLLM_MASK_ID

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    elif args.decode_backend == "fast-dllm":
        from model.modeling_llada import LLaDAModelLM  # Fast-dLLM/llada/model/
        from generate import generate_with_prefix_cache  # Fast-dLLM/llada/generate.py
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = LLaDAModelLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda().eval()
        _FAST_DLLM_GEN_FN = generate_with_prefix_cache
        _FAST_DLLM_MASK_ID = tokenizer.convert_tokens_to_ids("[MASK]")
        print(f"  Fast-dLLM prefix cache enabled for LLaDA", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

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
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool", "ipool", "idnmr", "idnmr_2round"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # === baseline: decode from C0 ===
        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens)

        # === pool: single-round, distribution-based extraction ===
        seed_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens)
        pool_cands = extract_candidates_generic(model, tokenizer, old_ctx, qtext, args.n_candidates, extraction_steps=args.extraction_steps)
        pool_passages, _ = expand_evidence(retriever, qtext, seed_ans, pool_cands,
                                           initial, args.expand_top_k)
        pool_ctx = "\n\n".join(pool_passages)
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        # === ipool: iterative with answer-conditioned bridges ===
        ipool_passages = list(initial)
        ipool_ctx = old_ctx
        prev_ipool_ans = seed_ans
        for r in range(args.max_rounds):
            bc = extract_bridges_from_answer(
                model, tokenizer, ipool_ctx, qtext, prev_ipool_ans, args.n_candidates
            )
            ipool_passages, new_p = expand_evidence(retriever, qtext, prev_ipool_ans, bc,
                                                    ipool_passages, args.expand_top_k)
            ipool_ctx = "\n\n".join(ipool_passages)
            ipool_ans = simple_decode(model, tokenizer, ipool_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)
            if len(new_p) == 0 and ipool_ans.strip().lower() == prev_ipool_ans.strip().lower():
                break
            prev_ipool_ans = ipool_ans

        # === idnmr: iterative with distribution-based extraction each round ===
        idnmr_passages = list(initial)
        idnmr_ctx = old_ctx
        prev_idnmr_ans = seed_ans
        idnmr_round_stats = []
        for r in range(args.max_rounds):
            # KEY DIFFERENCE: use extract_candidates_generic (token distribution)
            # not extract_bridges_from_answer (committed tokens)
            bc = extract_candidates_generic(model, tokenizer, idnmr_ctx, qtext, args.n_candidates, extraction_steps=args.extraction_steps)
            idnmr_passages, new_p = expand_evidence(retriever, qtext, prev_idnmr_ans, bc,
                                                    idnmr_passages, args.expand_top_k)
            idnmr_ctx = "\n\n".join(idnmr_passages)
            idnmr_ans = simple_decode(model, tokenizer, idnmr_ctx, qtext,
                                      steps=args.steps, n_tokens=args.answer_tokens)
            idnmr_round_stats.append({
                "round": r, "new_passages": len(new_p),
                "n_candidates": len(bc),
                "candidates": [c.get("text", "")[:40] for c in bc],
            })
            if len(new_p) == 0 and idnmr_ans.strip().lower() == prev_idnmr_ans.strip().lower():
                break
            prev_idnmr_ans = idnmr_ans

        # === idnmr_2round: same but capped at 2 rounds ===
        idnmr2_passages = list(initial)
        idnmr2_ctx = old_ctx
        prev_idnmr2_ans = seed_ans
        for r in range(min(2, args.max_rounds)):
            bc = extract_candidates_generic(model, tokenizer, idnmr2_ctx, qtext, args.n_candidates, extraction_steps=args.extraction_steps)
            idnmr2_passages, new_p = expand_evidence(retriever, qtext, prev_idnmr2_ans, bc,
                                                     idnmr2_passages, args.expand_top_k)
            idnmr2_ctx = "\n\n".join(idnmr2_passages)
            idnmr2_ans = simple_decode(model, tokenizer, idnmr2_ctx, qtext,
                                       steps=args.steps, n_tokens=args.answer_tokens)
            if len(new_p) == 0 and idnmr2_ans.strip().lower() == prev_idnmr2_ans.strip().lower():
                break
            prev_idnmr2_ans = idnmr2_ans

        elapsed = time.time() - tq

        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold, "elapsed": round(elapsed, 2),
               "idnmr_stats": idnmr_round_stats}

        for method, ans in [("baseline", baseline_ans), ("pool", pool_ans),
                            ("ipool", ipool_ans), ("idnmr", idnmr_ans),
                            ("idnmr_2round", idnmr2_ans)]:
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
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s)", flush=True)
            for m in methods:
                print(f"  {m:16s} {row[m]['answer'][:40]:40s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

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
