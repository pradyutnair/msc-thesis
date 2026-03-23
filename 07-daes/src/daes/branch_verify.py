"""
Diffusion Branch-and-Verify for Multi-Hop QA.

Core idea:
1. dLLM proposes hop-1 candidates from its token distribution (single forward pass)
2. Retrieve evidence per candidate
3. Re-denoise with evidence → measure confidence CHANGE
4. Select path with highest confidence gain (evidence-validated)
5. Continue to next hop

Modes:
  - baseline       : single-shot retrieval + confidence denoising (no branching)
  - spread_baseline: single-shot retrieval + SPREAD denoising
  - branch_verify  : branch on hop-1 candidates, verify with evidence, select best path
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
# Retriever (same as before)
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, dataset, index_name="index_e5_musique_full", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        data_dir = f"/projects/prjs1800/external/arag/data/{dataset}"
        idx_path = os.path.join(data_dir, index_name, "sentence_index.pkl")
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
# Core: dLLM generation with confidence tracking
# ---------------------------------------------------------------------------

def dllm_generate(model, tokenizer, context, question, steps=128, n_tokens=512, temperature=0.1):
    """Generate answer and return (answer_text, avg_confidence, per_token_confidences)."""
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
    token_confidences = torch.zeros(n_tokens, device=device)

    for step in range(steps):
        if remaining <= 0:
            break
        mi = (x == mask_id)
        if not mi.any():
            break

        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mp = mi[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mp], temperature=temperature, neg_entropy=True)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining

        _, topk = torch.topk(conf, min(n_commit, len(conf)))
        selected = mp[topk]
        x[0, selected] = x0[topk]

        # Record confidence for committed tokens
        for idx, pos in enumerate(selected):
            local_pos = pos.item() - n_prefix
            if 0 <= local_pos < n_tokens:
                token_confidences[local_pos] = conf[topk[idx]]

        remaining -= len(topk)

    # Extract answer
    gen_ids = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    avg_conf = token_confidences[token_confidences > 0].mean().item() if (token_confidences > 0).any() else 0

    return answer, avg_conf, token_confidences


def extract_hop1_candidates(model, tokenizer, context, question, n_candidates=3, temperature=0.3):
    """
    Get top-N candidate answers for hop 1 from the dLLM's token distribution.
    Uses a single forward pass to read the probability distribution at answer positions.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id

    prompt = f"{context}\n\nQuestion: {question}\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    # Short mask region for hop-1 answer (entity name, typically 1-5 tokens)
    n_mask = 20
    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    # Single forward pass
    with torch.no_grad():
        out = model(x, attention_mask=attn)
    logits = out.logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

    # Get probability distribution at first mask position
    first_mask_logits = logits[0, n_prefix] / temperature
    probs = torch.softmax(first_mask_logits, dim=-1)

    # Top-N candidates (single tokens — often entity names start with distinctive tokens)
    top_probs, top_ids = torch.topk(probs, n_candidates * 3)

    # Generate short answers for each top candidate by seeding the first token
    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        # Seed first token, denoise the rest
        x_cand = torch.tensor([canvas], dtype=torch.long, device=device)
        x_cand[0, n_prefix] = tid

        # Quick denoise remaining positions (16 steps for short answer)
        remaining = n_mask - 1
        for step in range(16):
            if remaining <= 0:
                break
            mi = (x_cand == mask_id)
            if not mi.any():
                break
            with torch.no_grad():
                out2 = model(x_cand, attention_mask=attn)
            logits2 = torch.cat([out2.logits[:, :1], out2.logits[:, :-1]], dim=1)
            mp = mi[0].nonzero(as_tuple=True)[0]
            conf2, x02 = sample_tokens(logits2[0, mp], temperature=0.1, neg_entropy=True)
            k = min(max(1, remaining // 16), remaining)
            if step == 15:
                k = remaining
            _, topk2 = torch.topk(conf2, min(k, len(conf2)))
            x_cand[0, mp[topk2]] = x02[topk2]
            remaining -= len(topk2)

        # Decode candidate answer
        cand_ids = x_cand[0, n_prefix:].tolist()
        cand_text = tokenizer.decode(cand_ids, skip_special_tokens=True).strip()
        # Take first meaningful phrase (before newline or period)
        cand_text = cand_text.split("\n")[0].split(". ")[0].strip()

        if cand_text and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append({
                "text": cand_text,
                "initial_confidence": prob.item(),
            })
            if len(candidates) >= n_candidates:
                break

    return candidates


# ---------------------------------------------------------------------------
# Branch-and-Verify pipeline
# ---------------------------------------------------------------------------

def branch_and_verify(model, tokenizer, retriever, question, n_candidates=3, steps=128, n_tokens=512):
    """
    Full branch-and-verify pipeline:
    1. Initial retrieval → propose hop-1 candidates
    2. Per-candidate: retrieve hop-2 evidence → re-denoise → measure confidence
    3. Select highest-confidence-gain path → return final answer
    """
    stats = {"n_candidates": 0, "selected": -1}

    # Step 1: Initial retrieval
    initial_passages = retriever.retrieve(question, top_k=5)
    initial_context = "\n\n".join(initial_passages)

    # Step 2: Generate hop-1 candidates from dLLM distribution
    candidates = extract_hop1_candidates(model, tokenizer, initial_context, question, n_candidates=n_candidates)
    stats["n_candidates"] = len(candidates)

    if not candidates:
        # Fallback: just generate directly
        answer, conf, _ = dllm_generate(model, tokenizer, initial_context, question, steps=steps, n_tokens=n_tokens)
        stats["method"] = "fallback"
        return answer, conf, stats

    # Step 3: For each candidate, retrieve more evidence and re-denoise
    best_answer = ""
    best_score = -1
    best_idx = -1

    for idx, cand in enumerate(candidates):
        # Retrieve hop-2 evidence using candidate as query context
        hop2_query = f"{question} {cand['text']}"
        hop2_passages = retriever.retrieve(hop2_query, top_k=3)

        # Build expanded context: initial passages + hop-2 passages
        expanded_context = initial_context + "\n\n" + "\n\n".join(hop2_passages)

        # Re-denoise with expanded context → get confidence
        answer, verified_conf, _ = dllm_generate(
            model, tokenizer, expanded_context, question,
            steps=steps, n_tokens=n_tokens
        )

        # Score = confidence change (verified - initial)
        # Higher means evidence SUPPORTED this path
        conf_gain = verified_conf - cand["initial_confidence"]
        # Combined score: verified confidence + confidence gain
        score = verified_conf + 0.5 * conf_gain

        candidates[idx]["verified_confidence"] = verified_conf
        candidates[idx]["conf_gain"] = conf_gain
        candidates[idx]["score"] = score
        candidates[idx]["answer"] = answer

        if score > best_score:
            best_score = score
            best_answer = answer
            best_idx = idx

    stats["selected"] = best_idx
    stats["candidates"] = [
        {"text": c["text"], "init_conf": round(c["initial_confidence"], 3),
         "verified_conf": round(c.get("verified_confidence", 0), 3),
         "conf_gain": round(c.get("conf_gain", 0), 3),
         "score": round(c.get("score", 0), 3)}
        for c in candidates
    ]
    stats["method"] = "branch_verify"

    return best_answer, best_score, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", required=True, choices=["baseline", "branch_verify"])
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--n_candidates", type=int, default=3)
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
            answer, conf, _ = dllm_generate(model, tokenizer, context, q["question"],
                                             steps=args.steps, n_tokens=args.n_tokens)
            stats = {"method": "baseline", "confidence": round(conf, 3)}
        elif args.mode == "branch_verify":
            answer, conf, stats = branch_and_verify(
                model, tokenizer, retriever, q["question"],
                n_candidates=args.n_candidates, steps=args.steps, n_tokens=args.n_tokens
            )

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

        cand_info = ""
        if "candidates" in stats:
            cand_info = " | cands: " + ", ".join(
                f"{c['text'][:20]}({c['score']:.2f})" for c in stats["candidates"]
            )
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain}{cand_info}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:120]}", flush=True)

    out_path = os.path.join(args.output_dir, f"bv_{args.dataset}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for item in predictions:
            f.write(json.dumps(item) + "\n")

    n = len(predictions)
    contain_n = sum(1 for item in predictions if item["contain"])
    print(f"\n{'='*60}", flush=True)
    print(f"Branch-Verify | {args.mode} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
    print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
    print(f"  Contain:   {contain_n}/{n} = {contain_n/n*100:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
