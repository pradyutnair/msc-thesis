"""2x2 Seed Ablation: query_prefix (on/off) x seed_length (16/32).
Runs pool method from idnmr_pilot logic with controlled variations.
Same 50q, same retriever, same model, same extraction."""
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


def expand_pool(retriever, question, seed_ans, bridge_cands, initial_passages,
                expand_top_k=3, use_prefix=True):
    prefix = "query: " if use_prefix else ""
    queries = [prefix + question + " " + seed_ans]
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if text and len(text) > 1:
            queries.append(prefix + question + " " + text)
    results = retriever.retrieve_batch(queries, expand_top_k)
    existing = set(initial_passages)
    all_p = list(initial_passages)
    for result_list in results:
        for p in result_list:
            if p not in existing:
                all_p.append(p)
                existing.add(p)
    return all_p, queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.time()
    print("=== Seed Ablation (%s) ===" % args.model, flush=True)
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
    print("Model loaded in %.1fs" % (time.time() - t0), flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    query_texts = ["query: " + q["question"] for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, 5)
    print("Loaded %d questions in %.1fs" % (len(questions), time.time() - t0), flush=True)

    cells = [
        ("prefix_on_seed16", True, 16),
        ("prefix_on_seed32", True, 32),
        ("prefix_off_seed16", False, 16),
        ("prefix_off_seed32", False, 32),
    ]
    totals = {"baseline": {"f1": 0.0}}
    for name, _, _ in cells:
        totals[name] = {"f1": 0.0}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        c0 = "\n\n".join(initial)

        # Baseline
        bl_ans = simple_decode(model, tokenizer, c0, qtext, steps=32, n_tokens=32)
        bl_f1 = compute_f1(bl_ans, gold)
        bl_f1 = bl_f1[2] if isinstance(bl_f1, tuple) else bl_f1
        totals["baseline"]["f1"] += bl_f1

        # Extract bridges once (same for all cells)
        cands = extract_candidates_generic(model, tokenizer, c0, qtext, 3, extraction_steps=12)

        row = {"id": q.get("id", "dev_%d" % qi), "question": qtext, "gold": gold,
               "baseline_f1": round(bl_f1, 4), "baseline_ans": bl_ans[:60],
               "candidates": [c.get("text", "")[:40] for c in cands]}

        for name, use_prefix, seed_len in cells:
            # Generate seed
            if seed_len == 16:
                seed, _, _ = short_generate(model, tokenizer, c0, qtext,
                                            steps=16, n_tokens=16, temperature=0.0)
            else:
                seed = simple_decode(model, tokenizer, c0, qtext, steps=32, n_tokens=32)

            # Expand
            pool_passages, queries_used = expand_pool(
                retriever, qtext, seed, cands, initial, expand_top_k=3, use_prefix=use_prefix)
            pool_ctx = "\n\n".join(pool_passages)

            # Decode
            pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext, steps=32, n_tokens=32)
            f1 = compute_f1(pool_ans, gold)
            f1 = f1[2] if isinstance(f1, tuple) else f1
            totals[name]["f1"] += f1

            row[name] = {"f1": round(f1, 4), "ans": pool_ans[:60],
                         "n_passages": len(pool_passages), "seed": seed[:40],
                         "n_queries": len(queries_used)}

        results.append(row)
        elapsed = time.time() - tq

        if (qi + 1) % 10 == 0 or qi == 0 or qi == len(questions) - 1:
            n_done = qi + 1
            parts = ["bl=%.3f" % (totals["baseline"]["f1"] / n_done)]
            for name, _, _ in cells:
                parts.append("%s=%.3f" % (name[:10], totals[name]["f1"] / n_done))
            print("[%d/%d] (%.1fs) %s" % (n_done, len(questions), elapsed, " | ".join(parts)), flush=True)

    n = len(results)
    summary = {}
    summary["baseline"] = round(totals["baseline"]["f1"] / n, 4)
    for name, _, _ in cells:
        summary[name] = round(totals[name]["f1"] / n, 4)

    print(chr(10) + "=" * 60, flush=True)
    print("Seed Ablation | %s | %s | N=%d" % (args.model, args.dataset, n), flush=True)
    for k, v in summary.items():
        delta = v - summary["baseline"]
        print("  %-20s F1=%.4f  delta=%+.4f" % (k, v, delta), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args)}, f, indent=2)
    print("Saved to %s" % args.output)
    print("Total: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
