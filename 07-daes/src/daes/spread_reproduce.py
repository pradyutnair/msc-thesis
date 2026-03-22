"""
Faithful SPREAD reproduction on MuSiQue.

Matches SPREAD paper setup:
  - L = 512 new tokens (mask canvas)
  - T = 128 denoising steps
  - k = L/T = 4 tokens per step
  - temperature = 0.1
  - Retrieved context: top-5 passages, each truncated to 2000 chars
  - Input: [chat_template: query + context] + [MASK]*512
  - h_q: separate forward pass on query tokens only
  - F1 evaluated on full denoised output (no EOS trimming)

Modes:
  - baseline  : Dream default (entropy-based confidence selection)
  - spread    : SPREAD Algorithm 1 (query-relevance selection)
  - espread   : our extension (query + evidence relevance)
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
    def __init__(self, dataset: str, model_name="intfloat/e5-base-v2",
                 index_name="index_e5_base_v2", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        data_dir = f"/projects/prjs1800/external/arag/data/{dataset}"
        index_path = os.path.join(data_dir, index_name, "sentence_index.pkl")
        print(f"Loading index from {index_path}...", flush=True)
        with open(index_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        print(f"Loading {model_name}...", flush=True)
        self.model = SentenceTransformer(model_name, device="cpu")
        print(f"Index: {len(self.sentences)} sentences, {len(self.chunks)} chunks", flush=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return top-k passage texts, each truncated to max_chunk_chars."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k * 3]
        chunk_best = {}
        for i in top_idx:
            cid = self.sentence_to_chunk[i]
            if cid not in chunk_best or sims[i] > chunk_best[cid]:
                chunk_best[cid] = float(sims[i])
        ranked = sorted(chunk_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        passages = []
        for cid, _ in ranked:
            text = self.chunks[cid]["text"][:self.max_chunk_chars]
            passages.append(text)
        return passages


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
    f = 2 * p * r / (p + r)
    return p, r, f


# ---------------------------------------------------------------------------
# SPREAD generation (faithful to Algorithm 1)
# ---------------------------------------------------------------------------

def spread_generate(
    model, tokenizer, query: str, context: str,
    L: int = 512, T: int = 128, temperature: float = 0.1,
    mode: str = "spread", alpha: float = 0.5,
) -> tuple[str, dict]:
    """
    Faithful SPREAD Algorithm 1 implementation.

    Input to model: [chat_template(query + context)] [MASK]*L
    h_q: separate forward pass on query tokens only (Alg 1 line 2)
    """
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Step 1: Build the full input
    # Format: chat template with query + context as user message, then mask canvas
    prompt = f"{context}\n\nQuestion: {query}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    # Canvas: prefix (fixed) + L mask tokens
    canvas = prefix_ids + [mask_id] * L
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    seq_len = x.shape[1]
    attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    # Step 2: Encode query separately (Alg 1 line 2: h_q <- M.encode_query(q))
    q_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    with torch.no_grad():
        q_out = model(q_ids, output_hidden_states=True)
    # Use last hidden layer, mean-pool over all query tokens
    h_q = q_out.hidden_states[-1].mean(dim=1)  # [1, D]
    h_q = F.normalize(h_q, dim=-1)

    # Denoising schedule: k = L/T tokens per step
    k = max(1, L // T)
    remaining = L
    stats = {"mode": mode, "n_prefix": n_prefix, "L": L, "T": T, "k": k}

    # Step 3: Denoising loop (Alg 1 lines 4-20)
    for step in range(T):
        if remaining <= 0:
            break
        mask_idx = (x == mask_id)
        if not mask_idx.any():
            break

        # Forward pass with hidden states (Alg 1 line 6)
        with torch.no_grad():
            out = model(x, attention_mask=attn_mask, output_hidden_states=True)
        logits = out.logits
        hs = out.hidden_states[-1]  # [1, seq_len, D]

        # AR-shift logits (Dream-specific: position i predicts token at i+1)
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        # Get currently masked positions (Alg 1 line 7)
        mask_pos = mask_idx[0].nonzero(as_tuple=True)[0]
        n_masked = len(mask_pos)

        # Predict tokens at all masked positions (temperature sampling)
        mask_logits = logits[0, mask_pos]  # [n_masked, vocab]
        # Use Dream's entropy-based sampling for token prediction
        _, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

        # How many to commit this step
        n_commit = min(k, remaining)
        if step == T - 1:
            n_commit = remaining  # commit all on last step

        if mode == "baseline":
            # Dream default: select by confidence (entropy-based)
            confidence, _ = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)
            _, topk = torch.topk(confidence, min(n_commit, n_masked))

        elif mode == "spread":
            # SPREAD Alg 1 lines 9-15: select by query relevance
            h_masked = hs[0, mask_pos]  # [n_masked, D]
            h_masked = F.normalize(h_masked, dim=-1)
            # Cosine similarity to query (Alg 1 line 11)
            sim = (h_masked @ h_q.squeeze(0))  # [n_masked]
            # Sigmoid normalization (Alg 1 line 12)
            rel = torch.sigmoid(sim)
            _, topk = torch.topk(rel, min(n_commit, n_masked))

        elif mode == "espread":
            # Our extension: query + evidence relevance
            h_masked = F.normalize(hs[0, mask_pos], dim=-1)
            # Query relevance
            sim_q = (h_masked @ h_q.squeeze(0))
            rel_q = torch.sigmoid(sim_q)
            # Evidence relevance: pool hidden states over prefix (evidence) positions
            ev_pos = torch.arange(n_prefix, device=device)
            h_ev = F.normalize(hs[0, ev_pos].mean(dim=0, keepdim=True), dim=-1)
            sim_e = (h_masked @ h_ev.squeeze(0))
            rel_e = torch.sigmoid(sim_e)
            rel = alpha * rel_q + (1 - alpha) * rel_e
            _, topk = torch.topk(rel, min(n_commit, n_masked))

        # Commit selected tokens (Alg 1 lines 16-18)
        selected = mask_pos[topk]
        x[0, selected] = x0[topk]
        remaining -= len(topk)

    # Extract answer: full denoised output after prefix (no EOS trimming — match SPREAD eval)
    gen_ids = x[0, n_prefix:].tolist()
    # Remove only remaining masks (should be 0), keep everything including EOS
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    stats["answer_len"] = len(answer.split())
    stats["n_eos"] = sum(1 for t in gen_ids if t == eos_id)
    return answer, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", required=True, choices=["baseline", "spread", "espread"])
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--L", type=int, default=512, help="Mask canvas size (new tokens)")
    parser.add_argument("--T", type=int, default=128, help="Denoising steps")
    parser.add_argument("--top_k", type=int, default=5, help="Retrieval top-k")
    parser.add_argument("--max_chunk_chars", type=int, default=2000, help="Max chars per retrieved passage")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load retriever
    retriever = Retriever(args.dataset, max_chunk_chars=args.max_chunk_chars)

    # Load model
    print(f"Loading {args.model_path}...", flush=True)
    from dataclasses import dataclass
    @dataclass
    class MA:
        model_name_or_path: str = args.model_path
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print("Model loaded.", flush=True)

    # Load questions
    qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))
    qs = qs[:args.n_questions]
    print(f"Loaded {len(qs)} questions.", flush=True)

    predictions = []
    sum_f1, sum_p, sum_r = 0, 0, 0

    for i, q in enumerate(qs):
        # Retrieve passages (truncated to max_chunk_chars each)
        passages = retriever.retrieve(q["question"], top_k=args.top_k)
        context = "\n\n".join(passages)

        t0 = time.time()
        try:
            answer, stats = spread_generate(
                model, tokenizer, q["question"], context,
                L=args.L, T=args.T, mode=args.mode, alpha=args.alpha,
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

        predictions.append({
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
            "answer_words": stats.get("answer_len", 0),
            "n_prefix": stats.get("n_prefix", 0),
        })

        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} P={p:.2f} R={r:.2f} contain={contain} words={stats.get('answer_len',0)}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:150]}", flush=True)

    out_path = os.path.join(args.output_dir, f"spread_repro_{args.dataset}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for p_item in predictions:
            f.write(json.dumps(p_item) + "\n")

    n = len(predictions)
    contain_n = sum(1 for p_item in predictions if p_item["contain"])
    print(f"\n{'='*60}", flush=True)
    print(f"SPREAD Reproduction | {args.mode} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
    print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
    print(f"  Contain:   {contain_n}/{n} = {contain_n/n*100:.1f}%", flush=True)
    print(f"  Avg words: {sum(p['answer_words'] for p in predictions)/n:.0f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
