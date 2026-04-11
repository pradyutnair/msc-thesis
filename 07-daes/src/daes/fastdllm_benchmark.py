"""Fast-dLLM efficiency benchmark: vanilla vs prefix-cached dLLM vs AR pipeline.

Measures wall-clock latency, FLOPs (analytical), and GPU memory for the full
DNMR pipeline under three configurations:
  1. vanilla:   standard dLLM denoising (no KV cache)
  2. fast-dllm: prefix-cached dLLM denoising (reuses prompt KV)
  3. ar:        AR model candidate extraction (requires second model)

FLOPs estimation: 2 * n_params * seq_len per forward pass (standard transformer
approximation). We count forward passes per phase and report total TFLOPs/question.

Supports both Dream-7B (FastdLLMDreamSampler) and LLaDA-8B (generate_with_prefix_cache).

Run:
  python -u src/daes/fastdllm_benchmark.py --model dream --dataset musique --n_questions 20
  python -u src/daes/fastdllm_benchmark.py --model llada --dataset musique --n_questions 20
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/Fast-dLLM/llada")

from eamd_v2_wiki18 import (
    _neg_entropy,
    Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1,
    short_user_prompt, extract_candidates_generic, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
import eamd_v2_wiki18
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Model parameter counting
# ---------------------------------------------------------------------------
def count_params(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def estimate_flops_per_fwd(n_params, seq_len):
    """Approximate FLOPs for one dense transformer forward pass.
    Standard estimate: ~2 * params * seq_len (matmul-dominated).
    """
    return 2 * n_params * seq_len


def estimate_ar_flops(n_params, prefill_len, n_generated):
    """Approximate FLOPs for AR generation with KV cache.
    Prefill: 2 * params * prefill_len (full forward pass)
    Each decode step: 2 * params * 1 (single-token forward, attends to cached KV)
    Total: 2 * params * (prefill_len + n_generated)
    """
    return 2 * n_params * (prefill_len + n_generated)


# ---------------------------------------------------------------------------
# Vanilla dLLM decode (counts forward passes)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def vanilla_decode(model, tokenizer, context, question, steps=32, n_tokens=32):
    """Standard dLLM decode. Returns (answer_text, n_forward_passes, avg_seq_len)."""
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)
    seq_len = len(canvas)
    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    n_fwd = 0

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break
        out = model(x, attention_mask=attn)
        n_fwd += 1
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

    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens]), n_fwd, seq_len


# ---------------------------------------------------------------------------
# Fast-dLLM decode (Dream: FastdLLMDreamSampler, LLaDA: generate_with_prefix_cache)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def fastdllm_decode_dream(sampler, tokenizer, context, question, steps=32, n_tokens=32,
                           threshold=0.9, block_size=None):
    """Fast-dLLM decode for Dream using FastdLLMDreamSampler with prefix cache."""
    from dllm.pipelines.fastdllm.dream import FastdLLMDreamSamplerConfig

    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)

    if block_size is None:
        # Pick largest divisor of n_tokens that also divides steps
        for bs in [n_tokens, 16, 8, 4, 2, 1]:
            if n_tokens % bs == 0 and steps % (n_tokens // bs) == 0:
                block_size = bs
                break
        else:
            block_size = n_tokens

    config = FastdLLMDreamSamplerConfig(
        steps=steps,
        max_new_tokens=n_tokens,
        temperature=0.0,
        alg="confidence_threshold",
        alg_temp=0.0,
        threshold=threshold,
        use_cache="prefix",
        block_size=block_size,
    )

    output = sampler.sample([prefix_ids], config, return_dict=True)
    n_fwd = len(output.histories) - 1 if output.histories else steps
    seq_len = n_prefix + n_tokens
    ans = decode_answer(tokenizer, output.sequences[0, -n_tokens:])
    return ans, n_fwd, seq_len


@torch.inference_mode()
def fastdllm_decode_llada(model, tokenizer, context, question, steps=32, n_tokens=32,
                           block_size=32, threshold=None):
    """Fast-dLLM decode for LLaDA using generate_with_prefix_cache."""
    from generate import generate_with_prefix_cache as gen_prefix  # type: ignore

    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)

    # Ensure divisibility
    if n_tokens % block_size != 0:
        block_size = n_tokens

    prompt = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
    output, _ = gen_prefix(
        model, prompt,
        steps=steps,
        gen_length=n_tokens,
        block_length=block_size,
        temperature=0.0,
        remasking="low_confidence",
        mask_id=mask_id,
        threshold=threshold,
    )
    ans = decode_answer(tokenizer, output[0, n_prefix:n_prefix + n_tokens])
    n_fwd = steps  # prefix cache still does ~steps NFEs, just faster per-NFE
    seq_len = n_prefix + n_tokens
    return ans, n_fwd, seq_len


# ---------------------------------------------------------------------------
# AR candidate extraction (same as latency_benchmark.py)
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
    n_fwd = 0
    candidates = []
    seen = set()
    for _ in range(n_candidates * 3):
        output = ar_model.generate(
            input_ids, max_new_tokens=30, temperature=0.7,
            do_sample=True, top_p=0.9,
            pad_token_id=ar_tokenizer.eos_token_id,
        )
        n_fwd += (output.shape[1] - input_ids.shape[1])  # autoregressive: 1 fwd per token
        text = ar_tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append({"text": text, "init_conf": 1.0 / (len(candidates) + 1)})
            if len(candidates) >= n_candidates:
                break
    return candidates, n_fwd, input_ids.shape[1]


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
def gpu_mem_current_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


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
    parser.add_argument("--ar_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--fastdllm_threshold", type=float, default=0.9,
                        help="Confidence threshold for fast-dLLM (Dream only)")
    parser.add_argument("--fastdllm_block_size", type=int, default=32,
                        help="Block size for fast-dLLM prefix cache")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/fastdllm_benchmark/{args.model}_{args.dataset}_{args.n_questions}q.json"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"=== Fast-dLLM Efficiency Benchmark ===", flush=True)
    print(f"Reader: {args.model}, AR: {args.ar_model}, Dataset: {args.dataset}", flush=True)
    print(f"N questions: {args.n_questions}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Load dLLM model ---
    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    print(f"Loading dLLM: {model_name}...", flush=True)

    t0 = time.time()
    if args.model == "dream":
        # For fast-dLLM Dream, need FastdLLMDreamConfig
        from dllm.pipelines.fastdllm.dream import (
            FastdLLMDreamConfig, FastdLLMDreamSampler,
        )
        model_args = SimpleNamespace(model_name_or_path=model_name)
        fastdllm_config = FastdLLMDreamConfig.from_pretrained(model_name)
        model = dllm.utils.get_model(model_args=model_args, config=fastdllm_config).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
        sampler = FastdLLMDreamSampler(model=model, tokenizer=tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda().eval()
        sampler = None
    dllm_load_sec = time.time() - t0

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model

    dllm_n_params = count_params(model)
    dllm_mem_mb = gpu_mem_current_mb()
    print(f"dLLM loaded: {dllm_load_sec:.1f}s, {dllm_n_params/1e9:.2f}B params, "
          f"GPU: {dllm_mem_mb:.0f} MB", flush=True)

    # --- Load AR model ---
    print(f"Loading AR: {args.ar_model}...", flush=True)
    t0 = time.time()
    ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_model, trust_remote_code=True)
    ar_model = AutoModelForCausalLM.from_pretrained(
        args.ar_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    ).eval()
    ar_load_sec = time.time() - t0

    ar_n_params = count_params(ar_model)
    dual_mem_mb = gpu_mem_current_mb()
    ar_mem_mb = dual_mem_mb - dllm_mem_mb
    print(f"AR loaded: {ar_load_sec:.1f}s, {ar_n_params/1e9:.2f}B params, "
          f"dual GPU: {dual_mem_mb:.0f} MB (+{ar_mem_mb:.0f} MB)", flush=True)

    # --- Load retriever + questions ---
    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions\n", flush=True)

    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)

    # --- Per-question benchmark ---
    timings = []

    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)
        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"), "question": qtext}

        # ===== PIPELINE A: Vanilla dLLM (no cache) =====
        torch.cuda.synchronize()
        t0 = time.time()
        seed_ans_v, seed_fwd_v, seed_seqlen = vanilla_decode(
            model, tokenizer, old_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens,
        )
        torch.cuda.synchronize()
        row["vanilla_seed_sec"] = time.time() - t0
        row["vanilla_seed_nfwd"] = seed_fwd_v

        # dLLM extraction (shared for vanilla & fast-dllm, since extraction reads logits from seed pass)
        t0 = time.time()
        dllm_cands = extract_candidates_generic(
            model, tokenizer, old_ctx, qtext, args.n_candidates,
            extraction_steps=args.extraction_steps,
        )
        torch.cuda.synchronize()
        row["dllm_extract_sec"] = time.time() - t0
        row["dllm_n_cands"] = len(dllm_cands)
        # Extraction does n_candidates * extraction_steps forward passes (branching)
        extract_fwd = args.n_candidates * args.extraction_steps
        row["dllm_extract_nfwd"] = extract_fwd

        # Retrieval (same for all pipelines)
        t0 = time.time()
        dllm_passages, dllm_new = expand_evidence(
            retriever, qtext, seed_ans_v, dllm_cands, initial, args.expand_top_k
        )
        row["retrieval_sec"] = time.time() - t0
        row["n_new_passages"] = len(dllm_new)
        pool_ctx = "\n\n".join(dllm_passages)

        # Vanilla pool decode
        torch.cuda.synchronize()
        t0 = time.time()
        pool_ans_v, pool_fwd_v, pool_seqlen = vanilla_decode(
            model, tokenizer, pool_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens,
        )
        torch.cuda.synchronize()
        row["vanilla_pool_sec"] = time.time() - t0
        row["vanilla_pool_nfwd"] = pool_fwd_v
        row["vanilla_total_sec"] = (row["vanilla_seed_sec"] + row["dllm_extract_sec"]
                                    + row["retrieval_sec"] + row["vanilla_pool_sec"])

        # ===== PIPELINE B: Fast-dLLM (prefix cache) =====
        torch.cuda.synchronize()
        t0 = time.time()
        if args.model == "dream":
            seed_ans_f, seed_fwd_f, _ = fastdllm_decode_dream(
                sampler, tokenizer, old_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens,
                threshold=args.fastdllm_threshold,
                block_size=args.fastdllm_block_size,
            )
        else:
            seed_ans_f, seed_fwd_f, _ = fastdllm_decode_llada(
                model, tokenizer, old_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens,
                block_size=args.fastdllm_block_size,
            )
        torch.cuda.synchronize()
        row["fast_seed_sec"] = time.time() - t0
        row["fast_seed_nfwd"] = seed_fwd_f

        # Fast-dLLM pool decode
        torch.cuda.synchronize()
        t0 = time.time()
        if args.model == "dream":
            pool_ans_f, pool_fwd_f, _ = fastdllm_decode_dream(
                sampler, tokenizer, pool_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens,
                threshold=args.fastdllm_threshold,
                block_size=args.fastdllm_block_size,
            )
        else:
            pool_ans_f, pool_fwd_f, _ = fastdllm_decode_llada(
                model, tokenizer, pool_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens,
                block_size=args.fastdllm_block_size,
            )
        torch.cuda.synchronize()
        row["fast_pool_sec"] = time.time() - t0
        row["fast_pool_nfwd"] = pool_fwd_f
        row["fast_total_sec"] = (row["fast_seed_sec"] + row["dllm_extract_sec"]
                                 + row["retrieval_sec"] + row["fast_pool_sec"])

        # ===== PIPELINE C: AR extraction pipeline =====
        torch.cuda.synchronize()
        t0 = time.time()
        ar_cands, ar_extract_fwd, ar_input_len = extract_candidates_ar(
            ar_model, ar_tokenizer, old_ctx, qtext, args.n_candidates
        )
        torch.cuda.synchronize()
        row["ar_extract_sec"] = time.time() - t0
        row["ar_extract_nfwd"] = ar_extract_fwd
        row["ar_n_cands"] = len(ar_cands)

        # AR retrieval
        t0 = time.time()
        ar_passages, ar_new = expand_evidence(
            retriever, qtext, seed_ans_v, ar_cands, initial, args.expand_top_k
        )
        row["ar_retrieval_sec"] = time.time() - t0
        ar_pool_ctx = "\n\n".join(ar_passages)

        # AR pipeline still uses dLLM for seed decode + pool decode (reader model)
        # but with AR-extracted candidates. Use fast-dLLM for decode.
        torch.cuda.synchronize()
        t0 = time.time()
        if args.model == "dream":
            ar_pool_ans, ar_pool_fwd, _ = fastdllm_decode_dream(
                sampler, tokenizer, ar_pool_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens,
                threshold=args.fastdllm_threshold,
                block_size=args.fastdllm_block_size,
            )
        else:
            ar_pool_ans, ar_pool_fwd, _ = fastdllm_decode_llada(
                model, tokenizer, ar_pool_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens,
                block_size=args.fastdllm_block_size,
            )
        torch.cuda.synchronize()
        row["ar_pool_sec"] = time.time() - t0
        # AR total includes fast seed decode + AR extraction + AR retrieval + fast pool decode
        row["ar_total_sec"] = (row["fast_seed_sec"] + row["ar_extract_sec"]
                               + row["ar_retrieval_sec"] + row["ar_pool_sec"])

        # ===== FLOPs estimation =====
        # dLLM FLOPs: count forward passes * flops_per_fwd
        dllm_flops_seed = row["vanilla_seed_nfwd"] * estimate_flops_per_fwd(dllm_n_params, seed_seqlen)
        dllm_flops_extract = extract_fwd * estimate_flops_per_fwd(dllm_n_params, seed_seqlen)
        dllm_flops_pool = row["vanilla_pool_nfwd"] * estimate_flops_per_fwd(dllm_n_params, pool_seqlen)
        row["vanilla_tflops"] = round((dllm_flops_seed + dllm_flops_extract + dllm_flops_pool) / 1e12, 2)

        fast_flops_seed = row["fast_seed_nfwd"] * estimate_flops_per_fwd(dllm_n_params, seed_seqlen)
        fast_flops_pool = row["fast_pool_nfwd"] * estimate_flops_per_fwd(dllm_n_params, pool_seqlen)
        row["fast_tflops"] = round((fast_flops_seed + dllm_flops_extract + fast_flops_pool) / 1e12, 2)

        # AR pipeline FLOPs: dLLM seed + AR extraction + dLLM pool decode
        # AR extraction: prefill prompt + generate n tokens per candidate attempt
        ar_flops_extract = estimate_ar_flops(ar_n_params, ar_input_len, row["ar_extract_nfwd"])
        # AR pool decode uses dLLM; tokenize to get proper seq_len
        ar_pool_prefix_ids, _ = build_short_prompt(tokenizer, ar_pool_ctx, qtext)
        ar_pool_seqlen = len(ar_pool_prefix_ids) + args.answer_tokens
        ar_flops_pool = ar_pool_fwd * estimate_flops_per_fwd(dllm_n_params, ar_pool_seqlen)
        row["ar_tflops"] = round((fast_flops_seed + ar_flops_extract + ar_flops_pool) / 1e12, 2)

        # Sanity F1
        row["vanilla_f1"] = round(compute_f1(pool_ans_v, gold), 4) if isinstance(compute_f1(pool_ans_v, gold), (int, float)) else round(compute_f1(pool_ans_v, gold)[0], 4)
        row["fast_f1"] = round(compute_f1(pool_ans_f, gold), 4) if isinstance(compute_f1(pool_ans_f, gold), (int, float)) else round(compute_f1(pool_ans_f, gold)[0], 4)

        timings.append(row)

        print(f"[{qi+1}/{len(questions)}] {row['id']}", flush=True)
        print(f"  vanilla: {row['vanilla_total_sec']:.2f}s  fast: {row['fast_total_sec']:.2f}s  "
              f"ar: {row['ar_total_sec']:.2f}s", flush=True)
        print(f"  TFLOPs: vanilla={row['vanilla_tflops']}  fast={row['fast_tflops']}  "
              f"ar={row['ar_tflops']}", flush=True)
        print(f"  speedup: {row['vanilla_total_sec']/max(0.001, row['fast_total_sec']):.2f}x "
              f"(fast vs vanilla)", flush=True)

    # --- Summary ---
    n = len(timings)
    summary = {
        "n_questions": n,
        "reader_model": model_name,
        "ar_model": args.ar_model,
        "dllm_params_B": round(dllm_n_params / 1e9, 2),
        "ar_params_B": round(ar_n_params / 1e9, 2),
        "dllm_gpu_mem_mb": round(dllm_mem_mb),
        "dual_model_gpu_mem_mb": round(dual_mem_mb),
        "ar_additional_mem_mb": round(ar_mem_mb),
    }

    for key in ["vanilla_seed_sec", "fast_seed_sec", "dllm_extract_sec", "ar_extract_sec",
                "retrieval_sec", "vanilla_pool_sec", "fast_pool_sec", "ar_pool_sec",
                "vanilla_total_sec", "fast_total_sec", "ar_total_sec",
                "vanilla_tflops", "fast_tflops", "ar_tflops"]:
        vals = [t[key] for t in timings]
        summary[f"avg_{key}"] = round(sum(vals) / max(1, n), 3)

    # Speedups
    vt = summary["avg_vanilla_total_sec"]
    ft = summary["avg_fast_total_sec"]
    at = summary["avg_ar_total_sec"]
    summary["speedup_fast_vs_vanilla"] = round(vt / max(0.001, ft), 2)
    summary["speedup_fast_vs_ar"] = round(at / max(0.001, ft), 2)
    summary["flop_ratio_fast_vs_ar"] = round(
        summary["avg_fast_tflops"] / max(0.001, summary["avg_ar_tflops"]), 2
    )

    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY ({n} questions)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"GPU memory:", flush=True)
    print(f"  dLLM only:     {summary['dllm_gpu_mem_mb']} MB ({summary['dllm_params_B']}B params)", flush=True)
    print(f"  dLLM + AR:     {summary['dual_model_gpu_mem_mb']} MB (+{summary['ar_additional_mem_mb']} MB for AR)", flush=True)
    print(f"\nWall-clock (avg per question):", flush=True)
    print(f"  Vanilla dLLM:  {summary['avg_vanilla_total_sec']:.3f}s", flush=True)
    print(f"  Fast-dLLM:     {summary['avg_fast_total_sec']:.3f}s  "
          f"({summary['speedup_fast_vs_vanilla']}x speedup)", flush=True)
    print(f"  AR pipeline:   {summary['avg_ar_total_sec']:.3f}s  "
          f"(fast-dLLM is {summary['speedup_fast_vs_ar']}x {'faster' if ft < at else 'slower'})", flush=True)
    print(f"\nTFLOPs (avg per question):", flush=True)
    print(f"  Vanilla dLLM:  {summary['avg_vanilla_tflops']}", flush=True)
    print(f"  Fast-dLLM:     {summary['avg_fast_tflops']}", flush=True)
    print(f"  AR pipeline:   {summary['avg_ar_tflops']}  "
          f"(ratio fast/ar: {summary['flop_ratio_fast_vs_ar']})", flush=True)
    print(f"\nPhase breakdown (fast-dLLM pipeline):", flush=True)
    print(f"  Seed decode:   {summary['avg_fast_seed_sec']:.3f}s", flush=True)
    print(f"  Extraction:    {summary['avg_dllm_extract_sec']:.3f}s", flush=True)
    print(f"  Retrieval:     {summary['avg_retrieval_sec']:.3f}s", flush=True)
    print(f"  Pool decode:   {summary['avg_fast_pool_sec']:.3f}s", flush=True)

    result = {"summary": summary, "timings": timings, "config": vars(args)}
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
