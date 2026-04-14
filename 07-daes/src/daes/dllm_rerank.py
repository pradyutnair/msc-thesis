"""
dLLM-Signal Passage Re-ranking for Multi-Hop QA.

Uses the dLLM's distributional shift (ARAM signal) as a passage relevance score.
For each candidate passage, compute how much it shifts the model's token distribution
at answer positions vs. a no-context prior. Higher shift = more relevant passage.

Pipeline:
  1. Dense retrieval → top-K candidate passages (wide net)
  2. For each passage: ONE batched forward pass → compute signal
  3. Re-rank by signal → select top-N
  4. Generate answer with re-ranked passages

Modes:
  - baseline_5      : Standard top-5 by retriever score (control)
  - baseline_20     : Standard top-20 by retriever score (more context)
  - rerank_signal   : Retrieve top-20, re-rank by dLLM signal, use top-5
  - rerank_entropy  : Retrieve top-20, re-rank by entropy reduction, use top-5
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
# Retriever (returns scored passages for re-ranking)
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, dataset, index_name="index_e5_base_v2", max_chunk_chars=2000):
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

    def retrieve_with_scores(self, query, top_k=20):
        """Return top-k passages WITH retriever scores."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k * 3]
        chunk_best = {}
        for i in top_idx:
            cid = self.sentence_to_chunk[i]
            if cid not in chunk_best or sims[i] > chunk_best[cid]:
                chunk_best[cid] = float(sims[i])
        ranked = sorted(chunk_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"text": self.chunks[cid]["text"][:self.max_chunk_chars],
             "retriever_score": score, "chunk_id": cid}
            for cid, score in ranked
        ]


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
    if not pt or not gt: return 0.0, 0.0, 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0, 0.0, 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return p, r, 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# dLLM Signal: score a single passage
# ---------------------------------------------------------------------------

def compute_passage_signal(model, tokenizer, passage, question, n_tokens=32):
    """
    Compute dLLM signal for a single passage.
    
    Signal = how much this passage shifts the model's token distribution
    at answer positions vs no-context prior.
    
    Uses short canvas (n_tokens=32) since we only need the signal, not a good answer.
    One batched forward pass (conditional + prior).
    
    Returns dict with signal metrics.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id

    # Conditional: passage + question + masks
    prompt_cond = f"{passage}\n\nQuestion: {question}\n\nAnswer:"
    msg_cond = [{"role": "user", "content": prompt_cond}]
    text_cond = tokenizer.apply_chat_template(msg_cond, tokenize=False, add_generation_prompt=True)
    prefix_cond = tokenizer.encode(text_cond, add_special_tokens=False)

    # Prior: question only + masks (no passage)
    prompt_prior = f"\n\nQuestion: {question}\n\nAnswer:"
    msg_prior = [{"role": "user", "content": prompt_prior}]
    text_prior = tokenizer.apply_chat_template(msg_prior, tokenize=False, add_generation_prompt=True)
    prefix_prior = tokenizer.encode(text_prior, add_special_tokens=False)

    # Pad shorter (prior) to match longer (conditional) length
    n_cond = len(prefix_cond)
    n_prior = len(prefix_prior)
    
    # Build canvases with mask tokens
    cond_ids = prefix_cond + [mask_id] * n_tokens
    
    # For prior: pad with mask_id on the LEFT to match cond length
    n_pad = n_cond - n_prior
    prior_ids = [mask_id] * n_pad + prefix_prior + [mask_id] * n_tokens
    
    assert len(cond_ids) == len(prior_ids), f"{len(cond_ids)} vs {len(prior_ids)}"
    seq_len = len(cond_ids)

    # Batched forward pass
    x_cond = torch.tensor([cond_ids], dtype=torch.long, device=device)
    x_prior = torch.tensor([prior_ids], dtype=torch.long, device=device)
    x_batch = torch.cat([x_cond, x_prior], dim=0)
    attn_batch = torch.ones((2, seq_len), dtype=torch.long, device=device)
    # Mask out padding positions in prior's attention
    if n_pad > 0:
        attn_batch[1, :n_pad] = 0

    with torch.no_grad():
        out = model(x_batch, attention_mask=attn_batch)

    # AR-shift
    logits_all = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    logits_cond = logits_all[0]   # [seq_len, V]
    logits_prior = logits_all[1]  # [seq_len, V]

    # Answer positions = last n_tokens positions (the mask canvas)
    answer_start = seq_len - n_tokens
    log_p_cond = F.log_softmax(logits_cond[answer_start:], dim=-1)
    log_p_prior = F.log_softmax(logits_prior[answer_start:], dim=-1)
    p_cond = log_p_cond.exp()
    p_prior = log_p_prior.exp()

    # Symmetric KL divergence per position
    kl_fwd = (p_cond * (log_p_cond - log_p_prior)).sum(dim=-1)
    kl_rev = (p_prior * (log_p_prior - log_p_cond)).sum(dim=-1)
    sym_kl = kl_fwd + kl_rev  # [n_tokens]

    # Entropy of conditional and prior
    h_cond = -(p_cond * log_p_cond).sum(dim=-1)  # [n_tokens]
    h_prior = -(p_prior * log_p_prior).sum(dim=-1)

    return {
        "signal": sym_kl.mean().item(),          # mean symmetric KL
        "signal_max": sym_kl.max().item(),        # max symmetric KL (most shifted position)
        "entropy_reduction": (h_prior - h_cond).mean().item(),  # how much passage reduces uncertainty
        "kl_fwd": kl_fwd.mean().item(),           # one-directional KL
    }


# ---------------------------------------------------------------------------
# Generate answer with given passages
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, passages, question, steps=128, n_tokens=512, temperature=0.1):
    """Standard Dream generation with given passages."""
    device = model.device
    mask_id = tokenizer.mask_token_id
    
    context = "\n\n".join(passages)
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

    for step in range(steps):
        if remaining <= 0:
            break
        mask_idx = (x[0] == mask_id)
        canvas_mask = mask_idx.clone()
        canvas_mask[:n_prefix] = False
        if not canvas_mask.any():
            break
        mask_pos = canvas_mask.nonzero(as_tuple=True)[0]
        n_masked = len(mask_pos)

        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mask_logits = logits[0, mask_pos]
        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining

        _, topk = torch.topk(confidence, min(n_commit, n_masked))
        x[0, mask_pos[topk]] = x0[topk]
        remaining -= len(topk)

    gen_ids = x[0, n_prefix:].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique",
                        choices=["musique", "hotpotqa", "2wikimultihop"])
    parser.add_argument("--n_questions", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50)
    parser.add_argument("--mode", required=True,
                        choices=["baseline_5", "baseline_20", "rerank_signal",
                                 "rerank_entropy", "rerank_kl_fwd", "rerank_signal_max"])
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--initial_k", type=int, default=20, help="Initial retrieval pool size")
    parser.add_argument("--rerank_n", type=int, default=5, help="Number of passages after re-ranking")
    parser.add_argument("--signal_tokens", type=int, default=32, help="Canvas size for signal computation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    index_map = {
        "musique": "index_e5_musique_full",
        "hotpotqa": "index_e5_full",
        "2wikimultihop": "index_e5_full",
    }
    retriever = Retriever(args.dataset, index_name=index_map[args.dataset])

    print(f"Loading {args.model_path}...", flush=True)
    from dataclasses import dataclass
    @dataclass
    class MA:
        model_name_or_path: str = args.model_path
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print("Model loaded.", flush=True)

    all_qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))
    if args.n_questions is not None:
        qs = all_qs[:args.n_questions]
    else:
        qs = all_qs[args.start_idx:args.end_idx]

    print(f"Running {len(qs)} questions | Mode: {args.mode} | "
          f"initial_k={args.initial_k} rerank_n={args.rerank_n}", flush=True)

    tag = f"rerank_{args.dataset}_{args.mode}_{args.start_idx}_{args.end_idx}"
    out_path = os.path.join(args.output_dir, f"{tag}.jsonl")
    open(out_path, "w").close()

    predictions = []
    sum_f1, sum_p = 0, 0

    for i, q in enumerate(qs):
        t0 = time.time()

        if args.mode == "baseline_5":
            # Standard: retrieve top-5, generate
            candidates = retriever.retrieve_with_scores(q["question"], top_k=5)
            passages = [c["text"] for c in candidates]
            rerank_info = None

        elif args.mode == "baseline_20":
            # Naive: retrieve top-20, generate (more context)
            candidates = retriever.retrieve_with_scores(q["question"], top_k=20)
            passages = [c["text"] for c in candidates]
            rerank_info = None

        else:
            # Re-ranking modes: retrieve top-K, score each, select top-N
            candidates = retriever.retrieve_with_scores(q["question"], top_k=args.initial_k)

            # Score each passage with dLLM
            for c in candidates:
                sig = compute_passage_signal(
                    model, tokenizer, c["text"], q["question"],
                    n_tokens=args.signal_tokens
                )
                c.update(sig)

            # Re-rank by chosen metric
            if args.mode == "rerank_signal":
                sort_key = "signal"
            elif args.mode == "rerank_entropy":
                sort_key = "entropy_reduction"
            elif args.mode == "rerank_kl_fwd":
                sort_key = "kl_fwd"
            elif args.mode == "rerank_signal_max":
                sort_key = "signal_max"

            candidates.sort(key=lambda c: c[sort_key], reverse=True)
            selected = candidates[:args.rerank_n]
            passages = [c["text"] for c in selected]

            # Log re-ranking info
            rerank_info = {
                "top5_signal": [round(c.get("signal", 0), 4) for c in candidates[:5]],
                "bot5_signal": [round(c.get("signal", 0), 4) for c in candidates[-5:]],
                "top5_retriever_rank": [
                    next(j for j, cc in enumerate(
                        sorted(candidates, key=lambda x: x["retriever_score"], reverse=True)
                    ) if cc["chunk_id"] == c["chunk_id"])
                    for c in selected
                ],
            }

        # Generate answer with selected passages
        answer = generate_answer(model, tokenizer, passages, q["question"])
        elapsed = time.time() - t0

        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f
        sum_p += p
        contain = q["answer"].lower() in answer.lower()

        pred = {
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": answer,
            "mode": args.mode,
            "n_passages": len(passages),
            "time": round(elapsed, 2),
            "f1": round(f, 4),
            "precision": round(p, 4),
            "contain": contain,
        }
        if rerank_info:
            pred["rerank"] = rerank_info

        predictions.append(pred)
        with open(out_path, "a") as fw:
            fw.write(json.dumps(pred) + "\n")

        extra = ""
        if rerank_info:
            extra = f" sig=[{','.join(f'{s:.2f}' for s in rerank_info['top5_signal'])}]"
            extra += f" orig_rank={rerank_info['top5_retriever_rank']}"
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain} "
              f"words={len(answer.split())}{extra}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:120]}", flush=True)

    n = len(predictions)
    if n > 0:
        cn = sum(1 for p in predictions if p["contain"])
        print(f"\n{'='*60}", flush=True)
        print(f"RERANK | {args.mode} | {args.dataset} | N={n}", flush=True)
        print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
        print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
        print(f"  Contain:   {cn}/{n} = {cn/n*100:.1f}%", flush=True)
        avg_time = np.mean([p["time"] for p in predictions])
        print(f"  Avg time:  {avg_time:.1f}s", flush=True)
        print(f"  Output:    {out_path}", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
