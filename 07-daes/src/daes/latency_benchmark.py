"""Latency & memory benchmark: dLLM extraction vs AR extraction.

Measures per-phase timing and GPU memory for the DNMR pipeline.
Phases: model load, candidate extraction, retrieval, pool decode.
Compares dLLM posterior extraction (free from same model) vs AR extraction
(requires separate model + separate inference).

Run: python -u src/daes/latency_benchmark.py --model dream --dataset musique --n_questions 20
     python -u src/daes/latency_benchmark.py --model dream --dataset musique --n_questions 20 --ar_model Qwen/Qwen2.5-7B-Instruct
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy,
    Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate,
    short_user_prompt, extract_candidates_generic, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import eamd_v2_wiki18


# ---------------------------------------------------------------------------
# Simple decode (same as idnmr_pilot.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# AR candidate extraction
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_candidates_ar(ar_model, ar_tokenizer, context, question, n_candidates=3):
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer briefly:"
    messages = [{"role": "user", "content": prompt}]
    input_text = ar_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = ar_tokenizer.encode(input_text, return_tensors="pt").to(ar_model.device)

    candidates = []
    seen = set()
    for _ in range(n_candidates * 3):
        output = ar_model.generate(
            input_ids, max_new_tokens=30, temperature=0.7,
            do_sample=True, top_p=0.9,
            pad_token_id=ar_tokenizer.eos_token_id,
        )
        text = ar_tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append({"text": text, "init_conf": 1.0 / (len(candidates) + 1)})
            if len(candidates) >= n_candidates:
                break
    return candidates


# ---------------------------------------------------------------------------
# Evidence expansion
# ---------------------------------------------------------------------------
def expand_evidence(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
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
# Memory helpers
# ---------------------------------------------------------------------------
def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0

def gpu_mem_current_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--ar_model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="AR model for candidate extraction comparison")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/latency_benchmark/{args.model}_{args.dataset}_{args.n_questions}q.json"

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"=== Latency Benchmark ===", flush=True)
    print(f"Reader: {args.model}, AR: {args.ar_model}, Dataset: {args.dataset}", flush=True)
    print(f"N questions: {args.n_questions}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Phase: Load dLLM reader ---
    reset_peak_mem()
    t0 = time.time()
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

    dllm_load_sec = time.time() - t0
    dllm_mem_mb = gpu_mem_current_mb()
    print(f"dLLM loaded: {dllm_load_sec:.1f}s, GPU mem: {dllm_mem_mb:.0f} MB", flush=True)

    # --- Phase: Load AR model ---
    t0 = time.time()
    ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_model, trust_remote_code=True)
    ar_model = AutoModelForCausalLM.from_pretrained(
        args.ar_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    ).eval()
    ar_load_sec = time.time() - t0
    dual_mem_mb = gpu_mem_current_mb()
    ar_additional_mem_mb = dual_mem_mb - dllm_mem_mb
    print(f"AR loaded: {ar_load_sec:.1f}s, dual GPU: {dual_mem_mb:.0f} MB (+{ar_additional_mem_mb:.0f} MB)", flush=True)

    # --- Load retriever + questions ---
    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)

    # --- Per-question timing ---
    timings = []  # per-question breakdown

    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)
        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"), "question": qtext}

        # 1. Seed decode (shared, needed for both pipelines)
        torch.cuda.synchronize()
        t0 = time.time()
        seed_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)
        torch.cuda.synchronize()
        row["seed_decode_sec"] = time.time() - t0

        # 2. dLLM candidate extraction
        torch.cuda.synchronize()
        t0 = time.time()
        dllm_cands = extract_candidates_generic(
            model, tokenizer, old_ctx, qtext, args.n_candidates,
            extraction_steps=args.extraction_steps
        )
        torch.cuda.synchronize()
        row["dllm_extract_sec"] = time.time() - t0
        row["dllm_n_cands"] = len(dllm_cands)

        # 3. AR candidate extraction
        torch.cuda.synchronize()
        t0 = time.time()
        ar_cands = extract_candidates_ar(
            ar_model, ar_tokenizer, old_ctx, qtext, args.n_candidates
        )
        torch.cuda.synchronize()
        row["ar_extract_sec"] = time.time() - t0
        row["ar_n_cands"] = len(ar_cands)

        # 4. Retrieval (dLLM candidates)
        t0 = time.time()
        dllm_passages, dllm_new = expand_evidence(
            retriever, qtext, seed_ans, dllm_cands, initial, args.expand_top_k
        )
        row["dllm_retrieval_sec"] = time.time() - t0
        row["dllm_n_new_passages"] = len(dllm_new)

        # 5. Retrieval (AR candidates)
        t0 = time.time()
        ar_passages, ar_new = expand_evidence(
            retriever, qtext, seed_ans, ar_cands, initial, args.expand_top_k
        )
        row["ar_retrieval_sec"] = time.time() - t0
        row["ar_n_new_passages"] = len(ar_new)

        # 6. Pool decode (dLLM candidates)
        torch.cuda.synchronize()
        t0 = time.time()
        dllm_pool_ans = simple_decode(
            model, tokenizer, "\n\n".join(dllm_passages), qtext,
            steps=args.steps, n_tokens=args.answer_tokens
        )
        torch.cuda.synchronize()
        row["dllm_pool_decode_sec"] = time.time() - t0

        # 7. Pool decode (AR candidates)
        torch.cuda.synchronize()
        t0 = time.time()
        ar_pool_ans = simple_decode(
            model, tokenizer, "\n\n".join(ar_passages), qtext,
            steps=args.steps, n_tokens=args.answer_tokens
        )
        torch.cuda.synchronize()
        row["ar_pool_decode_sec"] = time.time() - t0

        # Compute totals
        row["dllm_total_sec"] = row["seed_decode_sec"] + row["dllm_extract_sec"] + row["dllm_retrieval_sec"] + row["dllm_pool_decode_sec"]
        row["ar_total_sec"] = row["seed_decode_sec"] + row["ar_extract_sec"] + row["ar_retrieval_sec"] + row["ar_pool_decode_sec"]

        # F1 for sanity
        dllm_f1 = compute_f1(dllm_pool_ans, gold)
        row["dllm_f1"] = round(dllm_f1 if isinstance(dllm_f1, (int, float)) else dllm_f1[0], 4)
        ar_f1 = compute_f1(ar_pool_ans, gold)
        row["ar_f1"] = round(ar_f1 if isinstance(ar_f1, (int, float)) else ar_f1[0], 4)

        timings.append(row)

        print(f"[{qi+1}/{len(questions)}] {row['id']}", flush=True)
        print(f"  seed_decode: {row['seed_decode_sec']:.2f}s", flush=True)
        print(f"  dllm_extract: {row['dllm_extract_sec']:.2f}s  ar_extract: {row['ar_extract_sec']:.2f}s", flush=True)
        print(f"  dllm_retr: {row['dllm_retrieval_sec']:.2f}s  ar_retr: {row['ar_retrieval_sec']:.2f}s", flush=True)
        print(f"  dllm_decode: {row['dllm_pool_decode_sec']:.2f}s  ar_decode: {row['ar_pool_decode_sec']:.2f}s", flush=True)
        print(f"  dllm_total: {row['dllm_total_sec']:.2f}s  ar_total: {row['ar_total_sec']:.2f}s", flush=True)

    # --- Summary ---
    n = len(timings)
    summary = {
        "n_questions": n,
        "reader_model": model_name,
        "ar_model": args.ar_model,
        "dllm_gpu_mem_mb": round(dllm_mem_mb),
        "dual_model_gpu_mem_mb": round(dual_mem_mb),
        "ar_additional_mem_mb": round(ar_additional_mem_mb),
    }

    for key in ["seed_decode_sec", "dllm_extract_sec", "ar_extract_sec",
                "dllm_retrieval_sec", "ar_retrieval_sec", "dllm_pool_decode_sec", "ar_pool_decode_sec",
                "dllm_total_sec", "ar_total_sec"]:
        vals = [t[key] for t in timings]
        summary[f"avg_{key}"] = round(sum(vals) / max(1, n), 3)

    # Marginal cost: extraction only
    summary["dllm_marginal_extract_pct"] = round(
        100 * summary["avg_dllm_extract_sec"] / max(0.001, summary["avg_dllm_total_sec"]), 1
    )
    summary["ar_marginal_extract_pct"] = round(
        100 * summary["avg_ar_extract_sec"] / max(0.001, summary["avg_ar_total_sec"]), 1
    )

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY ({n} questions)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"GPU memory:", flush=True)
    print(f"  dLLM only:     {summary['dllm_gpu_mem_mb']} MB", flush=True)
    print(f"  dLLM + AR:     {summary['dual_model_gpu_mem_mb']} MB", flush=True)
    print(f"  AR additional: {summary['ar_additional_mem_mb']} MB", flush=True)
    print(f"\nPer-question averages:", flush=True)
    print(f"  Seed decode:     {summary['avg_seed_decode_sec']:.3f}s", flush=True)
    print(f"  dLLM extraction: {summary['avg_dllm_extract_sec']:.3f}s ({summary['dllm_marginal_extract_pct']}% of total)", flush=True)
    print(f"  AR extraction:   {summary['avg_ar_extract_sec']:.3f}s ({summary['ar_marginal_extract_pct']}% of total)", flush=True)
    print(f"  dLLM retrieval:  {summary['avg_dllm_retrieval_sec']:.3f}s", flush=True)
    print(f"  AR retrieval:    {summary['avg_ar_retrieval_sec']:.3f}s", flush=True)
    print(f"  dLLM pool decode: {summary['avg_dllm_pool_decode_sec']:.3f}s", flush=True)
    print(f"  AR pool decode:   {summary['avg_ar_pool_decode_sec']:.3f}s", flush=True)
    print(f"\nTotal per question:", flush=True)
    print(f"  dLLM pipeline: {summary['avg_dllm_total_sec']:.3f}s (single model)", flush=True)
    print(f"  AR pipeline:   {summary['avg_ar_total_sec']:.3f}s (dual model)", flush=True)

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "timings": timings, "config": vars(args)}, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
