"""DNMR-Replace pilot: Use bridge extraction to REPLACE initial passages, not append.
Key idea: retrieve with bridge queries, then re-retrieve top-K from union using original question.
LLaDA can't handle more context, but can benefit from BETTER context.
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy, Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate,
    extract_candidates_generic, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18
import numpy as np


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
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


def expand_replace(retriever, question, seed_answer, bridge_cands, initial_passages, budget=5, expand_top_k=3):
    """DNMR-Replace: retrieve with bridges, then re-rank union by question relevance, keep top-budget."""
    # Get new passages from bridge queries
    queries = [f"query: {question} {seed_answer}"]
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if text and len(text) > 1:
            queries.append(f"query: {question} {text}")
    results = retriever.retrieve_batch(queries, expand_top_k)
    
    # Collect all unique passages
    all_passages = list(initial_passages)
    seen = set(initial_passages)
    new_count = 0
    for result_list in results:
        for passage in result_list:
            if passage not in seen:
                all_passages.append(passage)
                seen.add(passage)
                new_count += 1
    
    # Re-rank ALL passages by relevance to original question and keep top-budget
    if len(all_passages) <= budget:
        return all_passages, new_count
    
    q_vec = retriever.model.encode([f"query: {question}"], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    p_vecs = retriever.model.encode(all_passages, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores = (q_vec @ p_vecs.T)[0]
    top_indices = np.argsort(-scores)[:budget]
    selected = [all_passages[i] for i in sorted(top_indices)]
    
    # Count how many new passages made it into the selection
    initial_set = set(initial_passages)
    new_in_selection = sum(1 for p in selected if p not in initial_set)
    
    return selected, new_in_selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--budget", type=int, default=5, help="Max passages after replacement")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== DNMR-Replace Pilot ===", flush=True)
    print(f"Model: {args.model}, Budget: {args.budget}", flush=True)

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
    print(f"Loaded {len(questions)} questions", flush=True)

    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.budget)
    print(f"  Retrieved in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "dnmr_replace"]
    totals = {m: {"f1": 0.0, "em": 0.0} for m in methods}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # baseline
        baseline_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens)

        # dnmr_replace: extract bridges, retrieve, replace (not append)
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                        steps=16, n_tokens=16, temperature=0.0)
        cands = extract_candidates_generic(model, tokenizer, old_ctx, qtext, 3)
        replaced_passages, n_new = expand_replace(
            retriever, qtext, seed_ans, cands, initial, budget=args.budget
        )
        new_ctx = "\n\n".join(replaced_passages)
        replace_ans = simple_decode(model, tokenizer, new_ctx, qtext,
                                    steps=args.steps, n_tokens=args.answer_tokens)

        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold, "n_new_in_budget": n_new,
               "candidates": [c.get("text", "")[:40] for c in cands]}

        for method, ans in [("baseline", baseline_ans), ("dnmr_replace", replace_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em}

        results.append(row)
        elapsed = time.time() - tq
        print(f"[{qi+1}/{len(questions)}] ({elapsed:.1f}s) new_in_budget={n_new}", flush=True)
        for m in methods:
            print(f"  {m:16s} {row[m]['answer'][:50]:50s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\nSummary")
    for m in methods:
        print(f"  {m:16s} F1={summary[m]['f1']:.4f} EM={summary[m]['em']:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Saved to {args.output}")
    print(f"Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
