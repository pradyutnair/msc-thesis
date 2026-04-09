"""DNMR Pool with KL×entropy extraction + _clean_bridge_candidate + n_tokens=32.

Minimal change from standard pool: only the extraction function and candidate cleaning differ.
Everything else (retrieval, decoding, n_tokens=32) is identical to standard pool.

Run:
  DAES_FAISS_GPU=1 DAES_FAISS_GPU_BACKEND=torch \
  python -u dnmr_pool_kl32.py --model llada --dataset musique \
    --n_questions 50 --output results/pool_kl32/llada_musique_50q.json
"""
import argparse, json, math, os, re, string, sys, time
from collections import Counter
from types import SimpleNamespace

import torch

sys.path.insert(0, os.environ.get("DLLM_PATH", "dllm"))
sys.path.insert(0, os.environ.get("DAES_PATH", "src/daes"))

import eamd_v2_wiki18
from eamd_v2_wiki18 import (
    QUESTION_FILES, Wiki18Retriever,
    build_short_prompt, get_mask_id, prepare_logits, decode_answer,
    extract_candidates_mixed_posterior, _clean_bridge_candidate,
    _neg_entropy,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from transformers import AutoModel, AutoTokenizer


# ── helpers ──────────────────────────────────────────────────────────
def normalize_answer(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())

def score(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return {"f1": 0, "em": 0, "contain": 0}
    common = Counter(pt) & Counter(gt)
    ov = sum(common.values())
    if ov == 0:
        return {"f1": 0, "em": 0, "contain": 0}
    p = ov / len(pt)
    r = ov / len(gt)
    f1 = 2 * p * r / (p + r)
    em = float(normalize_answer(pred) == normalize_answer(gold))
    contain = float(gold.strip().lower() in pred.strip().lower())
    return {"f1": round(f1, 4), "em": em, "contain": contain}

@torch.inference_mode()
def simple_decode(model, tokenizer, context, question, steps=32, n_tokens=32):
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=model.device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=model.device)
    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    neg_ent = _neg_entropy()
    for step in range(steps):
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, masked_local + n_prefix]
        conf, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=neg_ent)
        k = min(max(1, remaining // max(1, steps - step)), remaining, len(conf))
        _, topk = torch.topk(conf, k)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= k
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])

def expand_evidence(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
    queries = [f"query: {question} {seed_answer}"]
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if text:
            queries.append(f"query: {question} {text}")
    all_passages = list(current_passages)
    seen = set(p[:200] for p in all_passages)
    new_passages = []
    results = retriever.retrieve_batch(queries, expand_top_k)
    for batch in results:
        for p in batch:
            key = p[:200]
            if key not in seen:
                seen.add(key)
                all_passages.append(p)
                new_passages.append(p)
    return all_passages, new_passages


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada")
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    # KL extraction params
    parser.add_argument("--bridge_n_mask", type=int, default=6)
    parser.add_argument("--bridge_steps", type=int, default=12)
    parser.add_argument("--bridge_n_branch", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== DNMR Pool KL32 ===")
    print(f"Model={args.model} Dataset={args.dataset} N={args.n_questions} "
          f"answer_tokens={args.answer_tokens} bridge_n_mask={args.bridge_n_mask}")

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
    print(f"Model loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    # Batch initial retrieval
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"Initial retrieval done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool_kl32"]
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

        # === pool_kl32: KL×entropy extraction + cleaning + n_tokens=32 ===
        seed_ans = baseline_ans  # reuse baseline decode as seed

        # Extract with KL×entropy (the mixed posterior extractor)
        cands = extract_candidates_mixed_posterior(
            model, tokenizer, old_ctx, qtext,
            n_candidates=args.n_candidates,
            n_branch=args.bridge_n_branch,
            n_mask=args.bridge_n_mask,
            extraction_steps=args.bridge_steps,
        )

        # Clean candidates (strip "The answer is:", truncate to 6 words)
        for c in cands:
            if isinstance(c, dict) and "text" in c:
                c["text"] = _clean_bridge_candidate(c["text"], max_words=6)

        # Expand evidence
        pool_passages, new_p = expand_evidence(
            retriever, qtext, seed_ans, cands, initial, args.expand_top_k
        )
        pool_ctx = "\n\n".join(pool_passages)

        # Decode at n_tokens=32 (standard, robust, judge-compatible)
        pool_ans = simple_decode(model, tokenizer, pool_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        # Score
        row = {
            "id": q.get("id", f"q{args.start_idx + qi}"),
            "question": qtext,
            "gold": gold,
            "elapsed": round(time.time() - tq, 1),
        }

        for method, ans in [("baseline", baseline_ans), ("pool_kl32", pool_ans)]:
            s = score(ans, gold)
            row[method] = {"answer": ans, **s}
            for k in totals[method]:
                totals[method][k] += s[k]

        row["pool_kl32_meta"] = {
            "candidates": [c.get("text", "")[:60] if isinstance(c, dict) else str(c)[:60] for c in cands],
            "new_passages": len(new_p),
            "total_passages": len(pool_passages),
        }

        results.append(row)
        n_done = qi + 1
        elapsed = time.time() - tq

        if n_done % args.log_every == 0 or n_done == len(questions):
            avg_b = totals["baseline"]["f1"] / n_done
            avg_p = totals["pool_kl32"]["f1"] / n_done
            cont_b = totals["baseline"]["contain"] / n_done
            cont_p = totals["pool_kl32"]["contain"] / n_done
            print(f"[{n_done}/{len(questions)}] {row['id']} ({elapsed:.1f}s) "
                  f"base={avg_b:.3f}/{cont_b:.1%} pool_kl32={avg_p:.3f}/{cont_p:.1%}",
                  flush=True)

        if n_done % args.save_every == 0 or n_done == len(questions):
            summary = {m: {k: round(v / n_done, 4) for k, v in totals[m].items()} for m in methods}
            out = {
                "summary": summary,
                "results": results,
                "config": vars(args),
                "timing": {"total_sec": round(time.time() - t_start, 1), "per_q": round((time.time() - t_start) / n_done, 1)},
            }
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} questions in {time.time() - t_start:.0f}s", flush=True)
    print(f"Baseline: F1={totals['baseline']['f1']/len(results):.3f} Contain={totals['baseline']['contain']/len(results):.1%}")
    print(f"Pool_KL32: F1={totals['pool_kl32']['f1']/len(results):.3f} Contain={totals['pool_kl32']['contain']/len(results):.1%}")

if __name__ == "__main__":
    main()
