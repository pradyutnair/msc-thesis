"""DNMR Pool with BASE model for extraction, INSTRUCT model for decoding.

Hypothesis: LLaDA-8B-Base has broader posteriors (not collapsed by SFT),
producing more diverse bridge candidates. LLaDA-8B-Instruct is used for
seed decoding and final answer decoding (where confidence matters).

Run:
  python -u dnmr_pool_base_extract.py --model llada --dataset musique \
    --n_questions 50 --output results/pool_base_extract/llada_musique_50q.json
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
    build_short_prompt, build_short_pair, get_mask_id, prepare_logits,
    decode_answer, _clean_bridge_candidate, _neg_entropy,
)
from dllm.pipelines.dream.sampler import sample_tokens
import torch.nn.functional as F
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

@torch.inference_mode()
def extract_candidates_with_model(model, tokenizer, context, question,
                                   n_candidates=3, n_branch=3, n_mask=6,
                                   extraction_steps=12, min_position_mass=0.02):
    """extract_candidates_mixed_posterior but takes explicit model arg."""
    device = model.device
    mask_id = get_mask_id(tokenizer)

    full_ids, base_ids, n_prefix = build_short_pair(tokenizer, context, "", question, n_mask)

    x_base = torch.tensor([base_ids], dtype=torch.long, device=device)
    x_full = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn_base = torch.ones((1, len(base_ids)), dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)

    x_pair = torch.cat([x_base, x_full], dim=0)
    attn_pair = torch.cat([attn_base, attn_full], dim=0)
    out_pair = model(x_pair, attention_mask=attn_pair)
    logits_pair = prepare_logits(out_pair.logits)
    answer_logits_base = logits_pair[0, n_prefix:n_prefix + n_mask]
    answer_logits_full = logits_pair[1, n_prefix:n_prefix + n_mask]

    log_p_full = F.log_softmax(answer_logits_full, dim=-1)
    log_p_base = F.log_softmax(answer_logits_base, dim=-1)
    p_full = log_p_full.exp()
    entropy = -(p_full * log_p_full).sum(dim=-1)
    info_gain = (p_full * (log_p_full - log_p_base)).sum(dim=-1).clamp_min(0.0)
    position_signal = entropy * info_gain

    if position_signal.sum().item() <= 0:
        position_signal = entropy.clamp_min(1e-8)
    if position_signal.sum().item() <= 0:
        position_signal = torch.ones_like(position_signal)
    position_mass = position_signal / position_signal.sum()

    # Log entropy stats
    avg_entropy = entropy.mean().item()
    max_entropy = entropy.max().item()

    selected_positions = [
        pos for pos, mass in enumerate(position_mass.tolist())
        if mass >= min_position_mass
    ]
    if not selected_positions:
        selected_positions = [int(torch.argmax(position_mass).item())]

    branch_canvases = []
    branch_meta = []
    for pos_local in selected_positions:
        pos_global = n_prefix + pos_local
        pos_probs = torch.softmax(answer_logits_full[pos_local] / 0.3, dim=-1)
        top_probs, top_ids = torch.topk(pos_probs, min(n_branch, pos_probs.shape[-1]))
        for token_prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            canvas = list(full_ids)
            canvas[pos_global] = token_id
            branch_canvases.append(canvas)
            branch_meta.append({
                "position": pos_local,
                "position_mass": float(position_mass[pos_local].item()),
                "token_prob": float(token_prob),
            })

    if not branch_canvases:
        return [], avg_entropy

    x_all = torch.tensor(branch_canvases, dtype=torch.long, device=device)
    attn_batch = torch.ones((len(branch_canvases), x_all.shape[1]), dtype=torch.long, device=device)
    neg_ent = _neg_entropy()
    remaining = torch.full((len(branch_canvases),), n_mask - 1, dtype=torch.long, device=device)

    for step in range(extraction_steps):
        active = remaining > 0
        if not active.any():
            break
        active_idx = active.nonzero(as_tuple=True)[0]
        out = model(x_all[active_idx], attention_mask=attn_batch[:len(active_idx)])
        logits_active = prepare_logits(out.logits)
        for j, bi in enumerate(active_idx.tolist()):
            masked_positions = (x_all[bi] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_positions) == 0:
                remaining[bi] = 0
                continue
            conf, sampled = sample_tokens(logits_active[j, masked_positions], temperature=0.1, neg_entropy=neg_ent)
            rem = remaining[bi].item()
            n_commit = min(max(1, rem // extraction_steps), rem)
            if step == extraction_steps - 1:
                n_commit = rem
            _, topk_idx = torch.topk(conf, min(n_commit, len(conf)))
            x_all[bi, masked_positions[topk_idx]] = sampled[topk_idx]
            remaining[bi] -= len(topk_idx)

    # Decode and deduplicate
    candidate_texts = {}
    for bi, meta in enumerate(branch_meta):
        cand_text = tokenizer.decode(
            x_all[bi, n_prefix:n_prefix + n_mask].tolist(),
            skip_special_tokens=True,
        ).strip()
        cand_text = _clean_bridge_candidate(cand_text, max_words=6)
        if not cand_text or len(cand_text) <= 1:
            continue
        key = cand_text.lower()
        if key not in candidate_texts:
            candidate_texts[key] = {"text": cand_text, "init_conf": meta["token_prob"], "position": meta["position"]}

    candidates = list(candidate_texts.values())[:n_candidates]
    return candidates, avg_entropy

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
    parser.add_argument("--bridge_n_mask", type=int, default=6)
    parser.add_argument("--bridge_steps", type=int, default=12)
    parser.add_argument("--bridge_n_branch", type=int, default=3)
    parser.add_argument("--base_model_name", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== DNMR Pool Base-Extract ===", flush=True)
    print(f"Instruct model for decode, Base model for extraction", flush=True)
    print(f"Dataset={args.dataset} N={args.n_questions} answer_tokens={args.answer_tokens} bridge_n_mask={args.bridge_n_mask}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load INSTRUCT model (for decoding)
    instruct_name = "GSAI-ML/LLaDA-8B-Instruct"
    print(f"Loading instruct model: {instruct_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(instruct_name, trust_remote_code=True)
    instruct_model = AutoModel.from_pretrained(instruct_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"Instruct model loaded in {time.time() - t_start:.1f}s", flush=True)

    # Load BASE model (for extraction)
    print(f"Loading base model: {args.base_model_name}", flush=True)
    t_base = time.time()
    base_model = AutoModel.from_pretrained(args.base_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"Base model loaded in {time.time() - t_base:.1f}s", flush=True)

    # Set globals for tokenizer/model type (used by build_short_prompt etc)
    eamd_v2_wiki18.MODEL_REF = instruct_model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = "llada"

    # Check GPU memory
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"GPU memory: {(total_mem - free_mem) / 1e9:.1f}GB used / {total_mem / 1e9:.1f}GB total", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"Initial retrieval done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline", "pool_base_extract"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    entropy_log = {"base_model": [], "instruct_model": []}
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # === baseline: decode with INSTRUCT model ===
        baseline_ans = simple_decode(instruct_model, tokenizer, old_ctx, qtext,
                                     steps=args.steps, n_tokens=args.answer_tokens)
        seed_ans = baseline_ans

        # === Extract with BASE model ===
        cands_base, entropy_base = extract_candidates_with_model(
            base_model, tokenizer, old_ctx, qtext,
            n_candidates=args.n_candidates,
            n_branch=args.bridge_n_branch,
            n_mask=args.bridge_n_mask,
            extraction_steps=args.bridge_steps,
        )
        entropy_log["base_model"].append(entropy_base)

        # Also measure instruct entropy for comparison (first question only or every 10th)
        if qi % 10 == 0:
            _, entropy_inst = extract_candidates_with_model(
                instruct_model, tokenizer, old_ctx, qtext,
                n_candidates=args.n_candidates,
                n_branch=args.bridge_n_branch,
                n_mask=args.bridge_n_mask,
                extraction_steps=args.bridge_steps,
            )
            entropy_log["instruct_model"].append(entropy_inst)

        # Expand evidence using base model's candidates
        pool_passages, new_p = expand_evidence(
            retriever, qtext, seed_ans, cands_base, initial, args.expand_top_k
        )
        pool_ctx = "\n\n".join(pool_passages)

        # === Decode final answer with INSTRUCT model ===
        pool_ans = simple_decode(instruct_model, tokenizer, pool_ctx, qtext,
                                 steps=args.steps, n_tokens=args.answer_tokens)

        row = {
            "id": q.get("id", f"q{args.start_idx + qi}"),
            "question": qtext,
            "gold": gold,
            "elapsed": round(time.time() - tq, 1),
            "base_entropy": round(entropy_base, 4),
        }

        for method, ans in [("baseline", baseline_ans), ("pool_base_extract", pool_ans)]:
            s = score(ans, gold)
            row[method] = {"answer": ans, **s}
            for k in totals[method]:
                totals[method][k] += s[k]

        row["extraction_meta"] = {
            "candidates": [c.get("text", "")[:60] if isinstance(c, dict) else str(c)[:60] for c in cands_base],
            "new_passages": len(new_p),
            "total_passages": len(pool_passages),
        }

        results.append(row)
        n_done = qi + 1
        elapsed = time.time() - tq

        if n_done % args.log_every == 0 or n_done == len(questions):
            avg_b = totals["baseline"]["f1"] / n_done
            avg_p = totals["pool_base_extract"]["f1"] / n_done
            cont_b = totals["baseline"]["contain"] / n_done
            cont_p = totals["pool_base_extract"]["contain"] / n_done
            avg_ent = sum(entropy_log["base_model"]) / len(entropy_log["base_model"])
            print(f"[{n_done}/{len(questions)}] {row['id']} ({elapsed:.1f}s) "
                  f"base_ent={avg_ent:.4f} "
                  f"baseline={avg_b:.3f}/{cont_b:.1%} pool_base={avg_p:.3f}/{cont_p:.1%}",
                  flush=True)

        if n_done % args.save_every == 0 or n_done == len(questions):
            summary = {m: {k: round(v / n_done, 4) for k, v in totals[m].items()} for m in methods}
            avg_base_ent = sum(entropy_log["base_model"]) / max(1, len(entropy_log["base_model"]))
            avg_inst_ent = sum(entropy_log["instruct_model"]) / max(1, len(entropy_log["instruct_model"])) if entropy_log["instruct_model"] else 0
            summary["entropy"] = {
                "base_model_avg": round(avg_base_ent, 4),
                "instruct_model_avg": round(avg_inst_ent, 4),
            }
            out = {
                "summary": summary,
                "results": results,
                "config": vars(args),
                "timing": {"total_sec": round(time.time() - t_start, 1), "per_q": round((time.time() - t_start) / n_done, 1)},
            }
            with open(args.output, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} questions in {time.time() - t_start:.0f}s", flush=True)
    avg_base_ent = sum(entropy_log["base_model"]) / max(1, len(entropy_log["base_model"]))
    avg_inst_ent = sum(entropy_log["instruct_model"]) / max(1, len(entropy_log["instruct_model"])) if entropy_log["instruct_model"] else 0
    print(f"Avg entropy — Base: {avg_base_ent:.4f}, Instruct: {avg_inst_ent:.4f}", flush=True)
    print(f"Baseline: F1={totals['baseline']['f1']/len(results):.3f} Contain={totals['baseline']['contain']/len(results):.1%}", flush=True)
    print(f"Pool_Base: F1={totals['pool_base_extract']['f1']/len(results):.3f} Contain={totals['pool_base_extract']['contain']/len(results):.1%}", flush=True)

if __name__ == "__main__":
    main()
