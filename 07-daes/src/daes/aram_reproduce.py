"""
ARAM: Adaptive Retrieval-Augmented Masked Diffusion reproduction.
Faithful to arxiv 2603.17677.

Key detail: lambda_t is PER-TOKEN and PER-STEP (not a single scalar per step).
Each masked position gets its own guidance scale based on its local SNR.

Modes:
  - baseline : Dream default (entropy-based confidence selection, single forward pass)
  - aram     : ARAM adaptive SNR guidance (two-pass batched, per-token guided logits)
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
# Prior construction
# ---------------------------------------------------------------------------

def build_cond_and_prior(tokenizer, context, question, n_tokens):
    """
    Build conditional and prior input tensors of IDENTICAL length.

    Conditional: chat_template(context + question) + [MASK]*n_tokens
    Prior:       same but context token positions replaced with mask_id

    Per ARAM paper: p_prior = p_theta(.|x_t, q) — query without context.
    We mask context tokens with mask_id to keep lengths identical for batching.
    """
    mask_id = tokenizer.mask_token_id

    # Build prompt WITH context
    prompt_cond = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    msg_cond = [{"role": "user", "content": prompt_cond}]
    text_cond = tokenizer.apply_chat_template(msg_cond, tokenize=False, add_generation_prompt=True)
    prefix_cond = tokenizer.encode(text_cond, add_special_tokens=False)

    # Build prompt WITHOUT context (to find context token span)
    prompt_prior = f"\n\nQuestion: {question}\n\nAnswer:"
    msg_prior = [{"role": "user", "content": prompt_prior}]
    text_prior = tokenizer.apply_chat_template(msg_prior, tokenize=False, add_generation_prompt=True)
    prefix_prior = tokenizer.encode(text_prior, add_special_tokens=False)

    n_ctx = len(prefix_cond) - len(prefix_prior)

    # Find where context starts (divergence point between cond and prior tokenizations)
    ctx_start = 0
    for i in range(min(len(prefix_cond), len(prefix_prior))):
        if prefix_cond[i] != prefix_prior[i]:
            ctx_start = i
            break

    ctx_end = ctx_start + n_ctx

    # Build prior: replace context tokens with mask_id
    prior_prefix = list(prefix_cond)
    for i in range(ctx_start, ctx_end):
        prior_prefix[i] = mask_id

    # Add mask canvas
    cond_ids = prefix_cond + [mask_id] * n_tokens
    prior_ids = prior_prefix + [mask_id] * n_tokens

    assert len(cond_ids) == len(prior_ids), f"Length mismatch: {len(cond_ids)} vs {len(prior_ids)}"

    n_prefix = len(prefix_cond)
    return cond_ids, prior_ids, n_prefix, ctx_start, ctx_end


# ---------------------------------------------------------------------------
# ARAM generation (per-token adaptive guidance — faithful to paper)
# ---------------------------------------------------------------------------

def aram_generate(
    model, tokenizer, context, question,
    n_tokens=512, steps=128, temperature=0.1,
    lambda_max=1.0, beta=0.5, eps=1e-6,
    mode="aram",
):
    """
    ARAM adaptive SNR-guided generation.

    CRITICAL: lambda_t is computed PER-TOKEN (not per-step).
    Each masked position gets its own guidance scale from its local SNR.

    From paper Algorithm 1:
      for each masked position:
        Signal_i = D_KL(p_cond_i || p_prior_i) + D_KL(p_prior_i || p_cond_i)
        Noise_i  = H(p_cond_i)
        lambda_i = lambda_max * tanh(beta * Signal_i / (Noise_i + eps))
        l_guided_i = l_prior_i + lambda_i * (l_cond_i - l_prior_i)
    """
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Build conditional and prior inputs
    cond_ids, prior_ids, n_prefix, ctx_start, ctx_end = build_cond_and_prior(
        tokenizer, context, question, n_tokens
    )
    seq_len = len(cond_ids)

    # Conditional canvas
    x = torch.tensor([cond_ids], dtype=torch.long, device=device)
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)

    # Prior canvas (for ARAM mode)
    if mode == "aram":
        x_prior = torch.tensor([prior_ids], dtype=torch.long, device=device)
        attn_prior = torch.ones((1, seq_len), dtype=torch.long, device=device)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens

    # Diagnostics
    signal_traj = []  # mean signal per step (for logging)
    noise_traj = []
    lambda_traj = []  # mean lambda per step

    for step in range(steps):
        if remaining <= 0:
            break
        mask_idx = (x[0] == mask_id)
        # Only look at canvas region (after prefix)
        canvas_mask = mask_idx.clone()
        canvas_mask[:n_prefix] = False
        if not canvas_mask.any():
            break

        mask_pos = canvas_mask.nonzero(as_tuple=True)[0]
        n_masked = len(mask_pos)

        if mode == "baseline":
            # Single forward pass, confidence selection
            with torch.no_grad():
                out = model(x, attention_mask=attn)
            logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            mask_logits = logits[0, mask_pos]
            confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

        elif mode == "aram":
            # Sync prior canvas: copy committed tokens from cond to prior
            x_prior[0, n_prefix:] = x[0, n_prefix:]

            # Batched forward pass: [cond, prior]
            x_batch = torch.cat([x, x_prior], dim=0)
            attn_batch = torch.cat([attn, attn_prior], dim=0)

            with torch.no_grad():
                out = model(x_batch, attention_mask=attn_batch)

            # AR-shift
            logits_all = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            logits_cond = logits_all[0]   # [seq_len, vocab]
            logits_prior = logits_all[1]  # [seq_len, vocab]

            # Distributions at masked positions
            log_p_cond = F.log_softmax(logits_cond[mask_pos], dim=-1)   # [n_masked, V]
            log_p_prior = F.log_softmax(logits_prior[mask_pos], dim=-1) # [n_masked, V]
            p_cond = log_p_cond.exp()
            p_prior = log_p_prior.exp()

            # === PER-TOKEN Signal & Noise (ARAM Algorithm 1 lines 4-9) ===

            # Signal_i = D_KL(p_cond_i || p_prior_i) + D_KL(p_prior_i || p_cond_i)
            kl_fwd = (p_cond * (log_p_cond - log_p_prior)).sum(dim=-1)  # [n_masked]
            kl_rev = (p_prior * (log_p_prior - log_p_cond)).sum(dim=-1)  # [n_masked]
            signal_per_token = kl_fwd + kl_rev  # [n_masked]

            # Noise_i = H(p_cond_i) = -sum(p_cond_i * log(p_cond_i))
            noise_per_token = -(p_cond * log_p_cond).sum(dim=-1)  # [n_masked]

            # lambda_i = lambda_max * tanh(beta * Signal_i / (Noise_i + eps))
            lambda_per_token = lambda_max * torch.tanh(
                beta * signal_per_token / (noise_per_token + eps)
            )  # [n_masked]

            # === Per-token guided logits (Eq. 13) ===
            # l_guided_i = l_prior_i + lambda_i * (l_cond_i - l_prior_i)
            logits_cond_masked = logits_cond[mask_pos]    # [n_masked, V]
            logits_prior_masked = logits_prior[mask_pos]  # [n_masked, V]

            # Expand lambda to broadcast: [n_masked, 1]
            lam = lambda_per_token.unsqueeze(-1)  # [n_masked, 1]
            logits_guided = logits_prior_masked + lam * (logits_cond_masked - logits_prior_masked)

            # Sample from guided logits
            confidence, x0 = sample_tokens(logits_guided, temperature=temperature, neg_entropy=True)

            # Log diagnostics (means for readability)
            signal_traj.append(signal_per_token.mean().item())
            noise_traj.append(noise_per_token.mean().item())
            lambda_traj.append(lambda_per_token.mean().item())

        # How many to commit this step
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining

        _, topk = torch.topk(confidence, min(n_commit, n_masked))
        selected = mask_pos[topk]
        x[0, selected] = x0[topk]
        remaining -= len(topk)

    # Extract answer
    gen_ids = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Stats
    n_eos = sum(1 for t in gen_ids if t == eos_id)
    n_content = n_tokens - n_eos
    stats = {
        "mode": mode,
        "n_prefix": n_prefix,
        "n_tokens": n_tokens,
        "steps": steps,
        "n_content_tokens": n_content,
        "answer_words": len(answer.split()),
        "n_ctx_tokens": ctx_end - ctx_start,
    }
    if mode == "aram" and signal_traj:
        stats["mean_signal"] = round(float(np.mean(signal_traj)), 4)
        stats["mean_noise"] = round(float(np.mean(noise_traj)), 4)
        stats["mean_lambda"] = round(float(np.mean(lambda_traj)), 4)
        stats["signal_first5"] = [round(s, 4) for s in signal_traj[:5]]
        stats["lambda_first5"] = [round(l, 4) for l in lambda_traj[:5]]

    return answer, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique",
                        choices=["musique", "hotpotqa", "2wikimultihop"])
    parser.add_argument("--n_questions", type=int, default=None,
                        help="Number of questions (overrides start/end)")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50)
    parser.add_argument("--mode", required=True, choices=["baseline", "aram"])
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir",
                        default="/projects/prjs1800/msc-thesis/07-daes/results")
    # ARAM hyperparameters (Dream defaults from paper Table 4)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-6)
    # Generation settings
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_k_retrieval", type=int, default=5)
    parser.add_argument("--max_chunk_chars", type=int, default=2000)
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
          f"temp={args.temperature} lambda_max={args.lambda_max} beta={args.beta}",
          flush=True)

    tag = f"aram_{args.dataset}_{args.mode}_{args.start_idx}_{args.end_idx}"
    out_path = os.path.join(args.output_dir, f"{tag}.jsonl")
    open(out_path, "w").close()

    predictions = []
    sum_f1, sum_p, sum_r = 0, 0, 0

    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"], top_k=args.top_k_retrieval)
        context = "\n\n".join(passages)

        t0 = time.time()
        try:
            answer, stats = aram_generate(
                model, tokenizer, context, q["question"],
                n_tokens=args.n_tokens, steps=args.steps,
                temperature=args.temperature,
                lambda_max=args.lambda_max, beta=args.beta, eps=args.eps,
                mode=args.mode,
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
        if "mean_signal" in stats:
            extra = f" sig={stats['mean_signal']:.3f} lam={stats['mean_lambda']:.3f}"
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} P={p:.2f} "
              f"contain={contain} words={stats.get('answer_words',0)}{extra}",
              flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:150]}", flush=True)

    n = len(predictions)
    if n > 0:
        contain_n = sum(1 for p in predictions if p["contain"])
        print(f"\n{'='*60}", flush=True)
        print(f"ARAM | {args.mode} | {args.dataset} | N={n}", flush=True)
        print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
        print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
        print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
        print(f"  Contain:   {contain_n}/{n} = {contain_n/n*100:.1f}%", flush=True)
        if args.mode == "aram":
            sigs = [p["stats"].get("mean_signal", 0) for p in predictions]
            lams = [p["stats"].get("mean_lambda", 0) for p in predictions]
            print(f"  Avg signal: {np.mean(sigs):.4f}", flush=True)
            print(f"  Avg lambda: {np.mean(lams):.4f}", flush=True)
        print(f"  Output:    {out_path}", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
