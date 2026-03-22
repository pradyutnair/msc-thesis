"""
RAG + SPREAD on MuSiQue — proper retrieval, not gold evidence.

Uses ARAG's E5-base-v2 semantic search to retrieve passages, then runs
SPREAD/E-SPREAD denoising with Dream-7B.

Modes:
  - "baseline"     : confidence-based token selection
  - "spread"       : query-relevance token selection
  - "espread"      : query + evidence relevance token selection
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
# Retriever (standalone, no ARAG dependency)
# ---------------------------------------------------------------------------

class SimpleRetriever:
    """Minimal E5-based retriever using pre-built ARAG indices."""

    def __init__(self, dataset: str, model_name: str = "intfloat/e5-base-v2",
                 index_name: str = "index_e5_base_v2"):
        from sentence_transformers import SentenceTransformer

        data_dir = f"/projects/prjs1800/external/arag/data/{dataset}"
        index_path = os.path.join(data_dir, index_name, "sentence_index.pkl")

        print(f"Loading index from {index_path}...")
        with open(index_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]

        print(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name, device="cpu")
        print(f"Index loaded: {len(self.sentences)} sentences, {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k chunks by semantic similarity."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1][:top_k * 3]

        chunk_scores = {}
        for idx in top_indices:
            cid = self.sentence_to_chunk[idx]
            if cid not in chunk_scores or sims[idx] > chunk_scores[cid]:
                chunk_scores[cid] = float(sims[idx])

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for cid, score in ranked:
            results.append({
                "chunk_id": cid,
                "text": self.chunks[cid]["text"],
                "score": score,
            })
        return results


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
        return {"precision": 0, "recall": 0, "f1": 0}
    common = set(pt) & set(gt)
    if not common:
        return {"precision": 0, "recall": 0, "f1": 0}
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    f = 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------------------
# SPREAD generation
# ---------------------------------------------------------------------------

def spread_generate(
    model, tokenizer, prompt: str, query: str,
    n_tokens: int = 512, steps: int = 128, temperature: float = 0.1,
    mode: str = "spread", alpha: float = 0.5,
    evidence_positions: list[int] | None = None,
) -> tuple[str, dict]:
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    # Tokenize prompt
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prompt = len(prompt_ids)

    # Canvas: prompt + masks
    canvas_ids = prompt_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas_ids], dtype=torch.long, device=device)
    T = x.shape[1]
    attention_mask = torch.ones((1, T), dtype=torch.long, device=device)

    # Query embedding (SPREAD Alg 1 line 2)
    q_tokens = tokenizer.encode(query, return_tensors="pt").to(device)
    with torch.no_grad():
        q_out = model(q_tokens, output_hidden_states=True)
    h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)

    if evidence_positions is None:
        evidence_positions = list(range(n_prompt))

    n_masks = n_tokens
    tps = max(1, n_masks // steps)
    remaining = n_masks
    stats = {"mode": mode, "steps_used": 0}

    for step in range(steps):
        if remaining <= 0:
            break
        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        with torch.no_grad():
            out = model(x, attention_mask=attention_mask, output_hidden_states=True)
        logits = out.logits
        hs = out.hidden_states[-1]

        # AR-shift
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_pos = mask_index[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mask_pos], temperature=temperature, neg_entropy=True)

        k = min(tps, remaining)
        if step == steps - 1:
            k = remaining

        if mode == "baseline":
            _, topk_idx = torch.topk(conf, min(k, len(conf)))
        elif mode == "spread":
            h_m = F.normalize(hs[0, mask_pos], dim=-1)
            rel = torch.sigmoid((h_m @ h_q.squeeze(0)).float())
            _, topk_idx = torch.topk(rel, min(k, len(rel)))
        elif mode == "espread":
            h_m = F.normalize(hs[0, mask_pos], dim=-1)
            rel_q = torch.sigmoid((h_m @ h_q.squeeze(0)).float())
            ev_pos = torch.tensor(evidence_positions, device=device)
            ev_pos = ev_pos[ev_pos < T]
            if len(ev_pos) > 0:
                h_ev = F.normalize(hs[0, ev_pos].mean(dim=0, keepdim=True), dim=-1)
                rel_e = torch.sigmoid((h_m @ h_ev.squeeze(0)).float())
                rel = alpha * rel_q + (1 - alpha) * rel_e
            else:
                rel = rel_q
            _, topk_idx = torch.topk(rel, min(k, len(rel)))

        x[0, mask_pos[topk_idx]] = x0[topk_idx]
        remaining -= len(topk_idx)
        stats["steps_used"] = step + 1

    # Extract answer — keep all non-mask tokens (match SPREAD protocol)
    gen = x[0, n_prompt:].tolist()
    kept = [t for t in gen if t != mask_id]
    answer = tokenizer.decode(kept, skip_special_tokens=True).strip()
    stats["answer_tokens"] = len(kept)
    return answer, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "spread", "espread"])
    parser.add_argument("--model_path", type=str, default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--output_dir", type=str, default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load retriever (CPU)
    retriever = SimpleRetriever(args.dataset)

    # Load Dream-7B (GPU)
    print(f"Loading Dream-7B from {args.model_path}...")
    from dataclasses import dataclass
    @dataclass
    class MA:
        model_name_or_path: str = args.model_path
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())

    print(f"Loading {args.n_questions} questions from {args.dataset}...")
    path = f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"
    questions = json.load(open(path))[:args.n_questions]

    predictions = []
    total_f1, total_p, total_r = 0, 0, 0

    for i, q in enumerate(questions):
        # Retrieve passages
        results = retriever.retrieve(q["question"], top_k=args.top_k)
        evidence = "\n\n".join(
            f"[Passage {j+1}] {r['text']}" for j, r in enumerate(results)
        )

        prompt = (
            f"Answer the question based on the given information.\n\n"
            f"{evidence}\n\n"
            f"Question: {q['question']}\n\n"
            f"Answer:"
        )

        # Evidence token positions (approximate: everything before "Question:")
        prompt_ids = tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            ), add_special_tokens=False
        )
        ev_positions = list(range(int(len(prompt_ids) * 0.8)))

        t0 = time.time()
        try:
            pred, stats = spread_generate(
                model, tokenizer, prompt, q["question"],
                n_tokens=args.n_tokens, steps=args.steps,
                mode=args.mode, alpha=args.alpha,
                evidence_positions=ev_positions,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            pred, stats = "", {"error": str(e)}
        elapsed = time.time() - t0

        metrics = compute_f1(pred, q["answer"])
        total_f1 += metrics["f1"]
        total_p += metrics["precision"]
        total_r += metrics["recall"]

        predictions.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": pred,
            "mode": args.mode,
            "time": round(elapsed, 2),
            "f1": round(metrics["f1"], 4),
            "n_passages": len(results),
        })

        gold_in_pred = q["answer"].lower() in pred.lower()
        print(f"[{i+1}/{len(questions)}] ({elapsed:.1f}s) F1={metrics['f1']:.2f} contain={gold_in_pred} Q: {q['question'][:55]}...")
        print(f"  Gold: {q['answer']}")
        print(f"  Pred: {pred[:120]}")

    out_path = os.path.join(args.output_dir, f"rag_{args.dataset}_{args.mode}.jsonl")
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    n = len(predictions)
    contain = sum(1 for p in predictions if p["gold_answer"].lower() in p["pred_answer"].lower())
    print(f"\n{'='*60}")
    print(f"RAG + {args.mode} | Dataset: {args.dataset} | N={n} | top_k={args.top_k}")
    print(f"  F1:        {total_f1/n*100:.1f}%")
    print(f"  Precision: {total_p/n*100:.1f}%")
    print(f"  Recall:    {total_r/n*100:.1f}%")
    print(f"  Contain:   {contain}/{n} = {contain/n*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
