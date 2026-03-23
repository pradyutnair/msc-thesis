"""
Recursive Branch-and-Verify for Multi-Hop QA.

At each hop:
  1. dLLM proposes candidate bridge entities from token distribution
  2. Retrieve evidence per candidate
  3. Re-denoise with evidence → confidence change scores each path
  4. Prune to best candidate
  5. Continue to next hop with accumulated evidence

Modes:
  - baseline            : single-shot retrieval + generation (no branching)
  - branch_verify       : single-hop branching (original, hop 1 only)
  - recursive_branch    : recursive branching at every hop
"""

import argparse
import json
import os
import sys
import time
import re
import string
import pickle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, dataset, index_name="index_e5_musique_full", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        idx_path = f"/projects/prjs1800/external/arag/data/{dataset}/{index_name}/sentence_index.pkl"
        print(f"Loading index from {idx_path}...", flush=True)
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
        print(f"Index: {len(self.sentences)} sents, {len(self.chunks)} chunks", flush=True)

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k * 3]
        chunk_best = {}
        for i in top_idx:
            cid = self.sentence_to_chunk[i]
            if cid not in chunk_best or sims[i] > chunk_best[cid]:
                chunk_best[cid] = float(sims[i])
        ranked = sorted(chunk_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_f1(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt: return 0, 0, 0
    common = set(pt) & set(gt)
    if not common: return 0, 0, 0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return p, r, 2*p*r/(p+r)


# ---------------------------------------------------------------------------
# dLLM generation with confidence
# ---------------------------------------------------------------------------

def dllm_generate(model, tokenizer, context, question, steps=128, n_tokens=512, temperature=0.1):
    """Generate answer, return (text, avg_confidence)."""
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens
    confidences = []

    for step in range(steps):
        if remaining <= 0: break
        mi = (x == mask_id)
        if not mi.any(): break
        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mp = mi[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mp], temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1: n_commit = remaining
        _, topk = torch.topk(conf, min(n_commit, len(conf)))
        x[0, mp[topk]] = x0[topk]
        confidences.extend(conf[topk].tolist())
        remaining -= len(topk)

    gen_ids = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    return answer, avg_conf


def extract_candidates(model, tokenizer, context, question, n_candidates=3, temperature=0.3):
    """
    Propose candidate answers from dLLM's token distribution.
    Single forward pass → read top-k at first mask position → expand each into a short answer.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id

    prompt = f"{context}\n\nQuestion: {question}\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    n_mask = 20
    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    # Single forward pass → distribution at first mask position
    with torch.no_grad():
        out = model(x, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    first_logits = logits[0, n_prefix] / temperature
    probs = torch.softmax(first_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, n_candidates * 3)

    # Expand each seed token into a short answer
    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        x_cand = torch.tensor([canvas], dtype=torch.long, device=device)
        x_cand[0, n_prefix] = tid
        # Quick 16-step denoise for remaining positions
        rem = n_mask - 1
        for step in range(16):
            if rem <= 0: break
            mi = (x_cand == mask_id)
            if not mi.any(): break
            with torch.no_grad():
                out2 = model(x_cand, attention_mask=attn)
            logits2 = torch.cat([out2.logits[:, :1], out2.logits[:, :-1]], dim=1)
            mp = mi[0].nonzero(as_tuple=True)[0]
            conf2, x02 = sample_tokens(logits2[0, mp], temperature=0.1, neg_entropy=True)
            k = min(max(1, rem // 16), rem)
            if step == 15: k = rem
            _, topk2 = torch.topk(conf2, min(k, len(conf2)))
            x_cand[0, mp[topk2]] = x02[topk2]
            rem -= len(topk2)

        cand_ids = x_cand[0, n_prefix:].tolist()
        cand_text = tokenizer.decode(cand_ids, skip_special_tokens=True).strip()
        cand_text = cand_text.split("\n")[0].split(". ")[0].strip()

        if cand_text and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append({"text": cand_text, "init_conf": prob.item()})
            if len(candidates) >= n_candidates:
                break

    return candidates


# ---------------------------------------------------------------------------
# Recursive Branch-and-Verify
# ---------------------------------------------------------------------------

def recursive_branch_verify(
    model, tokenizer, retriever, question,
    max_hops=3, n_candidates=3, steps=128, n_tokens=512,
):
    """
    Recursively branch at each hop:
      Hop 1: retrieve → propose candidates → verify → select best
      Hop 2: use hop-1 winner to retrieve more → propose candidates → verify → select best
      ...
      Final: generate answer with all accumulated evidence
    """
    accumulated_passages = []
    hop_log = []

    # Initial retrieval
    passages = retriever.retrieve(question, top_k=5)
    accumulated_passages.extend(passages)

    for hop in range(max_hops):
        context = "\n\n".join(accumulated_passages)

        # Propose candidates for this hop
        candidates = extract_candidates(
            model, tokenizer, context, question, n_candidates=n_candidates
        )

        if not candidates:
            break  # No candidates → stop branching

        # Verify each candidate: retrieve with it, re-denoise, measure confidence
        for cand in candidates:
            hop_query = f"{question} {cand['text']}"
            cand_passages = retriever.retrieve(hop_query, top_k=3)
            expanded_context = context + "\n\n" + "\n\n".join(cand_passages)

            _, verified_conf = dllm_generate(
                model, tokenizer, expanded_context, question,
                steps=steps, n_tokens=n_tokens
            )
            cand["verified_conf"] = verified_conf
            cand["conf_gain"] = verified_conf - cand["init_conf"]
            cand["score"] = verified_conf + 0.5 * cand["conf_gain"]
            cand["passages"] = cand_passages

        # Select best candidate
        best = max(candidates, key=lambda c: c["score"])

        hop_log.append({
            "hop": hop + 1,
            "candidates": [
                {"text": c["text"][:50], "init": round(c["init_conf"], 3),
                 "verified": round(c["verified_conf"], 3),
                 "gain": round(c["conf_gain"], 3), "score": round(c["score"], 3),
                 "selected": c is best}
                for c in candidates
            ],
        })

        # Accumulate winning candidate's passages
        accumulated_passages.extend(best["passages"])

        # Check if further branching is useful:
        # If the best candidate's confidence gain is very high, we're confident → stop
        # If all candidates have similar scores, more branching won't help
        scores = [c["score"] for c in candidates]
        score_range = max(scores) - min(scores) if len(scores) > 1 else 0
        if best["conf_gain"] > 0.3 or score_range < 0.01:
            break  # Confident enough or no differentiation → stop branching

    # Final generation with all accumulated evidence
    final_context = "\n\n".join(accumulated_passages)
    answer, final_conf = dllm_generate(
        model, tokenizer, final_context, question,
        steps=steps, n_tokens=n_tokens
    )

    stats = {
        "method": "recursive_branch",
        "n_hops_used": len(hop_log),
        "n_passages": len(accumulated_passages),
        "final_confidence": round(final_conf, 3),
        "hops": hop_log,
    }

    return answer, final_conf, stats


def single_hop_branch_verify(model, tokenizer, retriever, question,
                              n_candidates=3, steps=128, n_tokens=512):
    """Original single-hop branch-verify (hop 1 only, for comparison)."""
    passages = retriever.retrieve(question, top_k=5)
    context = "\n\n".join(passages)

    candidates = extract_candidates(model, tokenizer, context, question, n_candidates=n_candidates)
    if not candidates:
        answer, conf = dllm_generate(model, tokenizer, context, question, steps=steps, n_tokens=n_tokens)
        return answer, conf, {"method": "fallback"}

    for cand in candidates:
        hop2_query = f"{question} {cand['text']}"
        hop2_passages = retriever.retrieve(hop2_query, top_k=3)
        expanded = context + "\n\n" + "\n\n".join(hop2_passages)
        _, verified_conf = dllm_generate(model, tokenizer, expanded, question, steps=steps, n_tokens=n_tokens)
        cand["verified_conf"] = verified_conf
        cand["conf_gain"] = verified_conf - cand["init_conf"]
        cand["score"] = verified_conf + 0.5 * cand["conf_gain"]
        cand["answer_context"] = expanded

    best = max(candidates, key=lambda c: c["score"])
    answer, conf = dllm_generate(model, tokenizer, best["answer_context"], question, steps=steps, n_tokens=n_tokens)

    stats = {
        "method": "branch_verify_1hop",
        "candidates": [{"text": c["text"][:50], "score": round(c["score"], 3)} for c in candidates],
    }
    return answer, conf, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "branch_1hop", "recursive_branch"])
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--max_hops", type=int, default=3)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    retriever = Retriever(args.dataset)

    print(f"Loading {args.model_path}...", flush=True)
    from dataclasses import dataclass
    @dataclass
    class MA:
        model_name_or_path: str = args.model_path
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print("Model loaded.", flush=True)

    qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))[:args.n_questions]
    print(f"Loaded {len(qs)} questions.", flush=True)

    predictions = []
    sum_f1, sum_p, sum_r = 0, 0, 0

    for i, q in enumerate(qs):
        t0 = time.time()

        if args.mode == "baseline":
            passages = retriever.retrieve(q["question"], top_k=5)
            context = "\n\n".join(passages)
            answer, conf = dllm_generate(model, tokenizer, context, q["question"],
                                          steps=args.steps, n_tokens=args.n_tokens)
            stats = {"method": "baseline"}

        elif args.mode == "branch_1hop":
            answer, conf, stats = single_hop_branch_verify(
                model, tokenizer, retriever, q["question"],
                n_candidates=args.n_candidates, steps=args.steps, n_tokens=args.n_tokens)

        elif args.mode == "recursive_branch":
            answer, conf, stats = recursive_branch_verify(
                model, tokenizer, retriever, q["question"],
                max_hops=args.max_hops, n_candidates=args.n_candidates,
                steps=args.steps, n_tokens=args.n_tokens)

        elapsed = time.time() - t0
        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f; sum_p += p; sum_r += r
        contain = q["answer"].lower() in answer.lower()

        predictions.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": answer,
            "mode": args.mode,
            "time": round(elapsed, 2),
            "f1": round(f, 4),
            "contain": contain,
            "stats": stats,
        })

        hop_info = ""
        if "n_hops_used" in stats:
            hop_info = f" hops={stats['n_hops_used']} passages={stats['n_passages']}"
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain}{hop_info}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:120]}", flush=True)

    out_path = os.path.join(args.output_dir, f"rbv_{args.dataset}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")

    n = len(predictions)
    contain_n = sum(1 for item in predictions if item["contain"])
    print(f"\n{'='*60}", flush=True)
    print(f"Recursive BV | {args.mode} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
    print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
    print(f"  Contain:   {contain_n}/{n} = {contain_n/n*100:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
