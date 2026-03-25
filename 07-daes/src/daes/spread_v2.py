"""
SPREAD reproduction with author-confirmed weighted scoring.
Faithful to arxiv 2601.11342 + author email (Chuanyue Yu).

Key detail from author: top-k selection uses WEIGHTED SUM of:
  - query-token semantic relevance (cosine similarity → sigmoid)
  - model confidence (neg-entropy)
This was not in the paper but confirmed via email.

Modes:
  - baseline    : Dream default (entropy-based confidence selection)
  - spread_v1   : Paper Algorithm 1 as-written (relevance only — known to fail)
  - spread_weighted : Author's actual implementation (alpha * relevance + (1-alpha) * confidence)
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
    if not pt or not gt:
        return 0.0, 0.0, 0.0
    common = set(pt) & set(gt)
    if not common:
        return 0.0, 0.0, 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return p, r, 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# SPREAD generation
# ---------------------------------------------------------------------------

def spread_generate(
    model, tokenizer, context, question,
    n_tokens=512, steps=128, temperature=0.1,
    mode="spread_weighted", alpha=0.5,
):
    """
    SPREAD generation.

    baseline:         Dream default — entropy confidence selection
    spread_v1:        Paper Algorithm 1 as-written — relevance only (known to fail)
    spread_weighted:  Author's actual impl — alpha * relevance + (1-alpha) * confidence

    h_q is computed from a SEPARATE forward pass on query only (confirmed by author).
    Hidden states from last layer, mean-pooled.
    """
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Build prompt
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    # Canvas: prefix + L mask tokens
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    seq_len = x.shape[1]
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)

    # Encode query SEPARATELY (SPREAD paper Alg 1 line 2, confirmed by author)
    if mode in ("spread_v1", "spread_weighted"):
        q_ids = tokenizer.encode(question, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = model(q_ids, output_hidden_states=True)
        # Last layer hidden states, mean-pool over query tokens
        h_q = q_out.hidden_states[-1].mean(dim=1)  # [1, D]
        h_q = F.normalize(h_q, dim=-1)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens

    # Diagnostics
    cosine_stds = []
    score_stds = []

    for step in range(steps):
        if remaining <= 0:
            break
        mask_idx = (x[0] == mask_id)
        # Canvas region only
        canvas_mask = mask_idx.clone()
        canvas_mask[:n_prefix] = False
        if not canvas_mask.any():
            break

        mask_pos = canvas_mask.nonzero(as_tuple=True)[0]
        n_masked = len(mask_pos)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining

        if mode == "baseline":
            with torch.no_grad():
                out = model(x, attention_mask=attn)
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            mask_logits = logits[0, mask_pos]
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)
            _, topk = torch.topk(confidence, min(n_commit, n_masked))

        elif mode == "spread_v1":
            # Paper Algorithm 1: relevance-only selection
            with torch.no_grad():
                out = model(x, attention_mask=attn, output_hidden_states=True)
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            hs = out.hidden_states[-1]

            mask_logits = logits[0, mask_pos]
            _, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

            # Relevance: cosine similarity → sigmoid
            h_masked = F.normalize(hs[0, mask_pos], dim=-1)
            sim = (h_masked @ h_q.squeeze(0))
            rel = torch.sigmoid(sim)

            cosine_stds.append(sim.std().item())

            # Select by relevance ONLY (paper as-written)
            _, topk = torch.topk(rel, min(n_commit, n_masked))

        elif mode == "spread_weighted":
            # Author's actual implementation: weighted relevance + confidence
            with torch.no_grad():
                out = model(x, attention_mask=attn, output_hidden_states=True)
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            hs = out.hidden_states[-1]

            mask_logits = logits[0, mask_pos]
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

            # Relevance: cosine similarity → sigmoid
            h_masked = F.normalize(hs[0, mask_pos], dim=-1)
            sim = (h_masked @ h_q.squeeze(0))
            rel = torch.sigmoid(sim)

            # Normalize both to [0, 1] range for fair weighting
            # Confidence (neg-entropy) is negative, higher = more confident
            # Shift to [0, 1]: conf_norm = (conf - min) / (max - min)
            if n_masked > 1:
                conf_min = confidence.min()
                conf_max = confidence.max()
                if conf_max > conf_min:
                    conf_norm = (confidence - conf_min) / (conf_max - conf_min)
                else:
                    conf_norm = torch.ones_like(confidence)

                rel_min = rel.min()
                rel_max = rel.max()
                if rel_max > rel_min:
                    rel_norm = (rel - rel_min) / (rel_max - rel_min)
                else:
                    rel_norm = torch.ones_like(rel)
            else:
                conf_norm = torch.ones_like(confidence)
                rel_norm = torch.ones_like(rel)

            # Weighted combination: alpha * relevance + (1-alpha) * confidence
            score = alpha * rel_norm + (1 - alpha) * conf_norm

            cosine_stds.append(sim.std().item())
            score_stds.append(score.std().item())

            _, topk = torch.topk(score, min(n_commit, n_masked))

        # Commit selected tokens
        selected = mask_pos[topk]
        x[0, selected] = x0[topk]
        remaining -= len(topk)

    # Extract answer
    gen_ids = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    n_eos = sum(1 for t in gen_ids if t == eos_id)
    n_content = n_tokens - n_eos
    stats = {
        "mode": mode,
        "n_prefix": n_prefix,
        "n_tokens": n_tokens,
        "steps": steps,
        "n_content_tokens": n_content,
        "answer_words": len(answer.split()),
    }
    if mode == "spread_v1" and cosine_stds:
        stats["mean_cosine_std"] = round(float(np.mean(cosine_stds)), 6)
    if mode == "spread_weighted":
        if cosine_stds:
            stats["mean_cosine_std"] = round(float(np.mean(cosine_stds)), 6)
        if score_stds:
            stats["mean_score_std"] = round(float(np.mean(score_stds)), 6)
        stats["alpha"] = alpha

    return answer, stats


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
                        choices=["baseline", "spread_v1", "spread_weighted"])
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir",
                        default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k_retrieval", type=int, default=5)
    parser.add_argument("--max_chunk_chars", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for relevance in weighted mode (0=confidence only, 1=relevance only)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    index_map = {
        "musique": "index_e5_musique_full",
        "hotpotqa": "index_e5_full",
        "2wikimultihop": "index_e5_full",
    }

    retriever = Retriever(args.dataset, index_name=index_map[args.dataset],
                          max_chunk_chars=args.max_chunk_chars)

    print(f"Loading {args.model_path}...", flush=True)
    from dataclasses import dataclass
    @dataclass
    class MA:
        model_name_or_path: str = args.model_path
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print("Model loaded.", flush=True)

    all_qs = json.load(
        open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json")
    )
    if args.n_questions is not None:
        qs = all_qs[:args.n_questions]
    else:
        qs = all_qs[args.start_idx:args.end_idx]
    print(f"Running {len(qs)} questions ({args.start_idx}-{args.end_idx})", flush=True)
    print(f"Mode: {args.mode} | steps={args.steps} n_tokens={args.n_tokens} "
          f"temp={args.temperature} alpha={args.alpha}", flush=True)

    tag = f"spread_{args.dataset}_{args.mode}_a{args.alpha}_{args.start_idx}_{args.end_idx}"
    out_path = os.path.join(args.output_dir, f"{tag}.jsonl")
    open(out_path, "w").close()

    predictions = []
    sum_f1, sum_p, sum_r = 0, 0, 0

    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"], top_k=args.top_k_retrieval)
        context = "\n\n".join(passages)

        t0 = time.time()
        try:
            answer, stats = spread_generate(
                model, tokenizer, context, q["question"],
                n_tokens=args.n_tokens, steps=args.steps,
                temperature=args.temperature, mode=args.mode,
                alpha=args.alpha,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            answer, stats = "", {"error": str(e)}
        elapsed = time.time() - t0

        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f
        sum_p += p
        sum_r += r
        contain = q["answer"].lower() in answer.lower()

        pred = {
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": answer,
            "mode": args.mode,
            "time": round(elapsed, 2),
            "f1": round(f, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "contain": contain,
            "stats": stats,
        }
        predictions.append(pred)
        with open(out_path, "a") as fw:
            fw.write(json.dumps(pred) + "\n")

        extra = ""
        if "mean_cosine_std" in stats:
            extra += f" cos_std={stats['mean_cosine_std']:.4f}"
        if "mean_score_std" in stats:
            extra += f" score_std={stats['mean_score_std']:.4f}"
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} P={p:.2f} "
              f"contain={contain} words={stats.get('answer_words',0)}{extra}",
              flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:150]}", flush=True)

    n = len(predictions)
    if n > 0:
        contain_n = sum(1 for p in predictions if p["contain"])
        print(f"\n{'='*60}", flush=True)
        print(f"SPREAD | {args.mode} | {args.dataset} | N={n} | alpha={args.alpha}", flush=True)
        print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
        print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
        print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
        print(f"  Contain:   {contain_n}/{n} = {contain_n/n*100:.1f}%", flush=True)
        if args.mode == "spread_v1":
            cs = [p["stats"].get("mean_cosine_std", 0) for p in predictions]
            print(f"  Avg cosine_std: {np.mean(cs):.6f}", flush=True)
        if args.mode == "spread_weighted":
            cs = [p["stats"].get("mean_cosine_std", 0) for p in predictions]
            ss = [p["stats"].get("mean_score_std", 0) for p in predictions]
            print(f"  Avg cosine_std: {np.mean(cs):.6f}", flush=True)
            print(f"  Avg score_std:  {np.mean(ss):.6f}", flush=True)
        print(f"  Output:    {out_path}", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
