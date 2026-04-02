"""Passage-Anchored Query Co-Denoising (PAQCD) Pilot.

For each question:
1. Retrieve initial top-5 passages (C0)
2. For each passage p_i, generate a search query via masked denoising:
   prompt = "Passage: {p_i}\nQuestion: {Q}\nSearch query to find more information:"
   + [MASK]^m
3. Retrieve with each generated query
4. Pool all passages (initial + new)
5. Decode final answer from pooled passages
6. Compare to baseline (decode from C0 only)
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy, Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate, QUESTION_FILES,
    short_user_prompt,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
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
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


@torch.inference_mode()
def generate_query_for_passage(model, tokenizer, passage, question, steps=16, n_tokens=16):
    """Generate a search query conditioned on a single passage + question."""
    device = model.device
    mask_id = get_mask_id(tokenizer)

    # Truncate passage to ~300 chars to keep prompt short
    passage_short = passage[:300].rsplit(" ", 1)[0]

    prompt = (
        f"Given this passage and question, write a short search query "
        f"to find additional useful information.\n\n"
        f"Passage: {passage_short}\n\n"
        f"Question: {question}\n\n"
        f"Search query:"
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

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

    query_text = tokenizer.decode(x[0, n_prefix:n_prefix + n_tokens].tolist(), skip_special_tokens=True).strip()
    # Clean up: take first line, remove quotes
    query_text = query_text.split(chr(10))[0].strip()
    return query_text


def run_paqcd(model, tokenizer, retriever, question, initial_passages,
              query_steps=16, query_tokens=16, decode_steps=32, decode_tokens=32,
              expand_top_k=3):
    """Passage-Anchored Query Co-Denoising."""

    # Stage 1: Generate one query per passage
    queries = []
    for p in initial_passages:
        q = generate_query_for_passage(model, tokenizer, p, question,
                                       steps=query_steps, n_tokens=query_tokens)
        if q and len(q) > 3:
            queries.append(q)

    # Deduplicate queries
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q)

    # Stage 2: Retrieve with each query
    all_passages = list(initial_passages)
    seen_texts = set(p[:100] for p in all_passages)

    if unique_queries:
        retrieval_queries = [f"query: {question} {q}" for q in unique_queries]
        batch_results = retriever.retrieve_batch(retrieval_queries, expand_top_k)
        for result_list in batch_results:
            for p in result_list:
                if p[:100] not in seen_texts:
                    all_passages.append(p)
                    seen_texts.add(p[:100])

    # Stage 3: Decode from pooled passages
    pooled_context = "\n\n".join(all_passages)
    answer = simple_decode(model, tokenizer, pooled_context, question,
                           steps=decode_steps, n_tokens=decode_tokens)

    return answer, {
        "n_queries": len(unique_queries),
        "queries": unique_queries[:5],
        "n_passages_total": len(all_passages),
        "n_new_passages": len(all_passages) - len(initial_passages),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== PAQCD Pilot ({args.model}) ===", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == "dream":
        ma = SimpleNamespace(model_name_or_path="Dream-org/Dream-v0-Instruct-7B")
        model = dllm.utils.get_model(model_args=ma).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=ma)
    else:
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).cuda().eval()
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    query_texts = ["query: " + q["question"] for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, 5)
    print(f"Loaded {len(questions)} questions in {time.time() - t0:.1f}s", flush=True)

    totals = {"baseline": {"f1": 0, "em": 0, "contain": 0},
              "paqcd": {"f1": 0, "em": 0, "contain": 0}}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        c0_text = "\n\n".join(initial)

        # Baseline
        bl_answer = simple_decode(model, tokenizer, c0_text, qtext)

        # PAQCD
        paqcd_answer, paqcd_stats = run_paqcd(
            model, tokenizer, retriever, qtext, initial)

        elapsed = time.time() - tq

        row = {"id": q.get("id", f"dev_{qi}"), "question": qtext, "gold": gold, "elapsed": round(elapsed, 1)}
        for method, ans in [("baseline", bl_answer), ("paqcd", paqcd_answer)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result[2] if isinstance(f1_result, tuple) else f1_result
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans[:80], "f1": round(f1, 4), "em": em, "contain": contain}
        row["paqcd_stats"] = paqcd_stats
        results.append(row)

        if (qi + 1) % 5 == 0 or qi == 0 or qi == len(questions) - 1:
            n_done = qi + 1
            bl_f1 = totals["baseline"]["f1"] / n_done
            pq_f1 = totals["paqcd"]["f1"] / n_done
            qs = paqcd_stats["queries"][:2]
            print(f"[{n_done}/{len(questions)}] ({elapsed:.1f}s) BL={bl_f1:.3f} PAQCD={pq_f1:.3f} queries={qs}", flush=True)
            print(f"  BL:   {bl_answer[:60]}", flush=True)
            print(f"  PAQCD:{paqcd_answer[:60]}", flush=True)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in totals}
    print(chr(10) + chr(61) * 60, flush=True)
    print(f"PAQCD Pilot | {args.model} | {args.dataset} | N={n}", flush=True)
    for m in ["baseline", "paqcd"]:
        s = summary[m]
        print(f"  {m:12s} F1={s[f1]:.4f} EM={s[em]:.4f} contain={s[contain]:.4f}", flush=True)
    delta = summary["paqcd"]["f1"] - summary["baseline"]["f1"]
    print(f"  Delta F1: {delta:+.4f} ({delta*100:+.1f}pp)", flush=True)
    print(f"  Total: {time.time() - t0:.1f}s ({(time.time() - t0) / max(1, n):.1f}s/q)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                   "timing": {"total_sec": round(time.time() - t0, 1)}}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
