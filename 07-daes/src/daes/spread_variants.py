"""
SPREAD variant experiments to find a working configuration.

Based on author email: "weighted sum of query-token semantic relevance and confidence."
Tests multiple combination strategies since the exact formula wasn't specified.

Variants:
  - baseline         : Dream default (confidence only)
  - additive_raw     : confidence + gamma * sigmoid(cosine) — raw values, no normalization
  - multiplicative   : confidence * (1 + gamma * (rel - 0.5)) — confidence-preserving
  - layer20          : Same as additive but using layer 20 hidden states (best variance per prior experiments)
  - short_canvas     : L=32 tokens (fewer masks → higher hidden state variance)
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
# SPREAD variant generation
# ---------------------------------------------------------------------------

def spread_variant_generate(
    model, tokenizer, context, question,
    n_tokens=512, steps=128, temperature=0.1,
    mode="additive_raw", gamma=1.0, hs_layer=-1,
):
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
    seq_len = x.shape[1]
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)

    # Encode query separately (author-confirmed)
    need_hs = mode != "baseline"
    if need_hs:
        q_ids = tokenizer.encode(question, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = model(q_ids, output_hidden_states=True)
        # Use specified layer for h_q (default -1 = last layer)
        h_q = q_out.hidden_states[hs_layer].mean(dim=1)
        h_q = F.normalize(h_q, dim=-1)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens
    cosine_stds = []
    score_stds = []

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

        else:
            with torch.no_grad():
                out = model(x, attention_mask=attn, output_hidden_states=True)
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)

            mask_logits = logits[0, mask_pos]
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

            # Hidden states at mask positions (using specified layer)
            hs = out.hidden_states[hs_layer]
            h_masked = F.normalize(hs[0, mask_pos], dim=-1)
            sim = (h_masked @ h_q.squeeze(0))  # raw cosine [-1, 1]
            rel = torch.sigmoid(sim)  # [0, 1], centered ~0.5

            cosine_stds.append(sim.std().item())

            if mode == "additive_raw":
                # Add raw sigmoid relevance to confidence (subtle nudge)
                # confidence ~ [-10, -0.1], rel ~ [0.47, 0.53]
                # gamma controls how much relevance matters relative to confidence
                score = confidence + gamma * rel
            elif mode == "multiplicative":
                # Confidence-preserving: multiply by (1 + gamma*(rel-0.5))
                # rel-0.5 ~ [-0.03, +0.03], so multiplier ~ [1-0.03*gamma, 1+0.03*gamma]
                score = confidence * (1.0 + gamma * (rel - 0.5))
            elif mode == "layer20":
                # Same as additive_raw but hs_layer is set to 20 (via argument)
                score = confidence + gamma * rel
            elif mode == "softmax_rel":
                # Use softmax over cosine similarities instead of sigmoid
                # This creates more spread even with low variance
                rel_softmax = F.softmax(sim / 0.01, dim=0)  # sharp softmax, temp=0.01
                score = confidence + gamma * rel_softmax
            else:
                score = confidence  # fallback

            score_stds.append(score.std().item())
            _, topk = torch.topk(score, min(n_commit, n_masked))

        selected = mask_pos[topk]
        x[0, selected] = x0[topk]
        remaining -= len(topk)

    gen_ids = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    n_eos = sum(1 for t in gen_ids if t == eos_id)
    stats = {
        "mode": mode, "n_tokens": n_tokens, "steps": steps,
        "n_content_tokens": n_tokens - n_eos,
        "answer_words": len(answer.split()), "gamma": gamma,
    }
    if cosine_stds:
        stats["mean_cosine_std"] = round(float(np.mean(cosine_stds)), 6)
    if score_stds:
        stats["mean_score_std"] = round(float(np.mean(score_stds)), 6)
    return answer, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique",
                        choices=["musique", "hotpotqa", "2wikimultihop"])
    parser.add_argument("--n_questions", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50)
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "additive_raw", "multiplicative",
                                 "layer20", "softmax_rel"])
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k_retrieval", type=int, default=5)
    parser.add_argument("--max_chunk_chars", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Weight for relevance contribution")
    parser.add_argument("--hs_layer", type=int, default=-1,
                        help="Hidden state layer index (-1=last, 20=layer20)")
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
    # Check how many layers the model has
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 'unknown'
    print(f"Model loaded. Layers: {n_layers}", flush=True)

    all_qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))
    if args.n_questions is not None:
        qs = all_qs[:args.n_questions]
    else:
        qs = all_qs[args.start_idx:args.end_idx]

    # For layer20 mode, set hs_layer
    hs_layer = args.hs_layer
    if args.mode == "layer20":
        hs_layer = 20

    print(f"Running {len(qs)} questions | Mode: {args.mode} gamma={args.gamma} hs_layer={hs_layer}", flush=True)

    tag = f"spreadvar_{args.dataset}_{args.mode}_g{args.gamma}_{args.start_idx}_{args.end_idx}"
    out_path = os.path.join(args.output_dir, f"{tag}.jsonl")
    open(out_path, "w").close()

    predictions = []
    sum_f1, sum_p = 0, 0

    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"], top_k=args.top_k_retrieval)
        context = "\n\n".join(passages)

        t0 = time.time()
        try:
            answer, stats = spread_variant_generate(
                model, tokenizer, context, q["question"],
                n_tokens=args.n_tokens, steps=args.steps,
                temperature=args.temperature, mode=args.mode,
                gamma=args.gamma, hs_layer=hs_layer,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            answer, stats = "", {"error": str(e)}
        elapsed = time.time() - t0

        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f; sum_p += p
        contain = q["answer"].lower() in answer.lower()

        pred = {
            "id": q["id"], "question": q["question"],
            "gold_answer": q["answer"], "pred_answer": answer,
            "mode": args.mode, "time": round(elapsed, 2),
            "f1": round(f, 4), "precision": round(p, 4),
            "contain": contain, "stats": stats,
        }
        predictions.append(pred)
        with open(out_path, "a") as fw:
            fw.write(json.dumps(pred) + "\n")

        extra = ""
        if "mean_cosine_std" in stats: extra += f" cos={stats['mean_cosine_std']:.4f}"
        if "mean_score_std" in stats: extra += f" sc={stats['mean_score_std']:.4f}"
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain} "
              f"words={stats.get('answer_words',0)}{extra}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:120]}", flush=True)

    n = len(predictions)
    if n > 0:
        cn = sum(1 for p in predictions if p["contain"])
        print(f"\n{'='*60}", flush=True)
        print(f"SPREAD-VAR | {args.mode} | gamma={args.gamma} | {args.dataset} | N={n}", flush=True)
        print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
        print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
        print(f"  Contain:   {cn}/{n} = {cn/n*100:.1f}%", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
