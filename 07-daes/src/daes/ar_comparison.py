"""Matched AR Comparison: dLLM posterior candidates vs AR diverse candidates.

Uses IDENTICAL pipeline to DNMR Pool (idnmr_pilot.py) — same retriever, same
decode settings, same multi-query pooling. Only the candidate SOURCE differs:
  pool_dllm    : candidates from dLLM posterior (extract_candidates_generic)
  pool_ar      : candidates from AR model (Qwen3-8B diverse sampling)
  pool_random  : random vocabulary words as candidates (control)
  baseline     : no expansion (decode from C0)

All methods use Dream-7B (or LLaDA) for final answer generation.
AR model (Qwen3-8B) is used ONLY for candidate extraction in pool_ar.

Run: python -u src/daes/ar_comparison.py --model dream --dataset musique --n_questions 50
"""
import argparse, json, math, os, sys, time, random
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

random.seed(42)

# Random candidate words (common nouns/entities for control)
RANDOM_WORDS = [
    "mountain", "river", "country", "film", "city", "president", "queen",
    "capital", "university", "battle", "treaty", "island", "company",
    "author", "director", "album", "stadium", "museum", "bridge", "temple",
    "scientist", "politician", "actor", "singer", "painter", "novel",
    "ocean", "desert", "forest", "valley", "kingdom", "empire", "church",
]


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
# Evidence expansion (same as idnmr_pilot.py — pools ALL candidates)
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
# AR candidate extraction (Qwen3-8B diverse sampling)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def extract_candidates_ar(ar_model, ar_tokenizer, context, question, n_candidates=3, disable_thinking=True):
    """Generate n diverse candidate answers using AR model (Qwen3-8B)."""
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer briefly:"
    messages = [{"role": "user", "content": prompt}]
    input_text = ar_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": False} if disable_thinking else {})
    )
    input_ids = ar_tokenizer.encode(input_text, return_tensors="pt").to(ar_model.device)

    candidates = []
    seen = set()
    for _ in range(n_candidates * 3):  # oversample for dedup
        output = ar_model.generate(
            input_ids, max_new_tokens=30, temperature=0.7,
            do_sample=True, top_p=0.9,
            pad_token_id=ar_tokenizer.eos_token_id,
        )
        text = ar_tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        # Clean: take first phrase
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append({"text": text, "init_conf": 1.0 / (len(candidates) + 1)})
            if len(candidates) >= n_candidates:
                break
    return candidates


# ---------------------------------------------------------------------------
# Random candidate extraction (control)
# ---------------------------------------------------------------------------
def extract_candidates_random(n_candidates=3):
    words = random.sample(RANDOM_WORDS, min(n_candidates, len(RANDOM_WORDS)))
    return [{"text": w, "init_conf": 0.0} for w in words]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--ar_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== AR Comparison ===", flush=True)
    print(f"Reader: {args.model}, AR: {args.ar_model}, Dataset: {args.dataset}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load reader model (Dream or LLaDA)
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
    print(f"  Reader loaded in {time.time() - t_start:.1f}s", flush=True)

    # Load AR model (Qwen3-8B) — on CPU or second GPU if available
    print(f"Loading AR model: {args.ar_model}...", flush=True)
    ar_tokenizer = AutoTokenizer.from_pretrained(args.ar_model, trust_remote_code=True)
    ar_model = AutoModelForCausalLM.from_pretrained(
        args.ar_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    ).eval()
    print(f"  AR model loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    # Batch initial retrieval
    print("Batch initial retrieval...", flush=True)
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool_dllm", "pool_ar", "pool_random"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # Shared seed decode
        seed_ans = simple_decode(model, tokenizer, old_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        # === baseline: decode from C0 ===
        baseline_ans = seed_ans

        # === pool_dllm: dLLM posterior candidates (DNMR) ===
        dllm_cands = extract_candidates_generic(
            model, tokenizer, old_ctx, qtext, args.n_candidates,
            extraction_steps=args.extraction_steps
        )
        dllm_passages, dllm_new = expand_evidence(
            retriever, qtext, seed_ans, dllm_cands, initial, args.expand_top_k
        )
        pool_dllm_ans = simple_decode(
            model, tokenizer, "\n\n".join(dllm_passages), qtext,
            steps=args.steps, n_tokens=args.answer_tokens
        )

        # === pool_ar: AR diverse candidates ===
        ar_cands = extract_candidates_ar(
            ar_model, ar_tokenizer, old_ctx, qtext, args.n_candidates
        )
        ar_passages, ar_new = expand_evidence(
            retriever, qtext, seed_ans, ar_cands, initial, args.expand_top_k
        )
        pool_ar_ans = simple_decode(
            model, tokenizer, "\n\n".join(ar_passages), qtext,
            steps=args.steps, n_tokens=args.answer_tokens
        )

        # === pool_random: random word candidates (control) ===
        random_cands = extract_candidates_random(args.n_candidates)
        random_passages, random_new = expand_evidence(
            retriever, qtext, seed_ans, random_cands, initial, args.expand_top_k
        )
        pool_random_ans = simple_decode(
            model, tokenizer, "\n\n".join(random_passages), qtext,
            steps=args.steps, n_tokens=args.answer_tokens
        )

        elapsed = time.time() - tq

        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext, "gold": gold, "elapsed": round(elapsed, 2),
            "dllm_candidates": [c.get("text", "")[:60] for c in dllm_cands],
            "ar_candidates": [c.get("text", "")[:60] for c in ar_cands],
            "random_candidates": [c.get("text", "")[:60] for c in random_cands],
            "n_new_dllm": len(dllm_new),
            "n_new_ar": len(ar_new),
            "n_new_random": len(random_new),
        }

        for method, ans in [("baseline", baseline_ans), ("pool_dllm", pool_dllm_ans),
                            ("pool_ar", pool_ar_ans), ("pool_random", pool_random_ans)]:
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
                print(f"  {m:16s} {row[m]['answer'][:40]:40s} F1={row[m]['f1']:.3f}", flush=True)
            print(f"  dLLM cands: {row['dllm_candidates']}", flush=True)
            print(f"  AR cands:   {row['ar_candidates']}", flush=True)

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
