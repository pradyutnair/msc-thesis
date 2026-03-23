"""
Track 1: SPREAD reproduction on LLaDA-8B-Instruct.
LLaDA uses full masked diffusion (not block-diffusion like Dream).
Mask hidden states might have more variance → SPREAD relevance might work.

Tests: baseline (confidence) vs spread (relevance) on 50 MuSiQue questions.
"""
import argparse, json, os, sys, time, re, string, pickle
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from dataclasses import dataclass


class Retriever:
    def __init__(self, index_name, precomputed_path, max_chunk_chars=2000):
        idx_path = f"/projects/prjs1800/external/arag/data/musique/{index_name}/sentence_index.pkl"
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences, self.embeddings = idx["sentences"], idx["embeddings"]
        self.sentence_to_chunk, self.chunks = idx["sentence_to_chunk"], idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        with open(precomputed_path, "rb") as f:
            qd = pickle.load(f)
        self.q_texts, self.q_embs = qd["questions"], qd["embeddings"]

    def retrieve(self, query, top_k=5):
        try:
            qi = self.q_texts.index(query)
            q_emb = self.q_embs[qi]
        except ValueError:
            return []
        sims = np.dot(self.embeddings, q_emb)
        top = np.argsort(sims)[::-1][:top_k * 3]
        cb = {}
        for i in top:
            cid = self.sentence_to_chunk[i]
            if cid not in cb or sims[i] > cb[cid]: cb[cid] = float(sims[i])
        ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_metrics(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return {"p": 0, "r": 0, "f1": 0}
    common = set(pt) & set(gt)
    if not common: return {"p": 0, "r": 0, "f1": 0}
    p, r = len(common)/len(pt), len(common)/len(gt)
    return {"p": p, "r": r, "f1": 2*p*r/(p+r)}

def copy_rate(pred, ctx):
    pw, cw = set(normalize(pred).split()), set(normalize(ctx).split())
    return len(pw & cw) / len(pw) if pw else 0


def generate_llada(model, tokenizer, context, question, L=512, T=128,
                   temperature=0.1, mode="baseline", h_q=None):
    """LLaDA generation with SPREAD support."""
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * L
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k = max(1, L // T)
    remaining = L

    # Track relevance stats
    rel_stds = []

    for step in range(T):
        if remaining <= 0: break
        mi = (x == mask_id)
        if not mi.any(): break

        with torch.no_grad():
            out = model(x, attention_mask=attn, output_hidden_states=True)
        logits = out.logits
        hs = out.hidden_states[-1]

        # LLaDA does NOT use AR-shifted logits (unlike Dream)
        # LLaDA's logits[i] directly predicts token at position i
        mp = mi[0].nonzero(as_tuple=True)[0]
        mask_logits = logits[0, mp]

        # Sample tokens
        conf, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

        nc = min(k, remaining)
        if step == T - 1: nc = remaining

        if mode == "baseline":
            _, topk = torch.topk(conf, min(nc, len(conf)))
        elif mode == "spread":
            h_m = F.normalize(hs[0, mp], dim=-1)
            sim = (h_m @ h_q.squeeze(0)).float()
            rel = torch.sigmoid(sim)
            if step < 3:
                rel_stds.append(rel.std().item())
            _, topk = torch.topk(rel, min(nc, len(rel)))

        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)

    gen = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return answer, {"rel_stds": rel_stds}


def main():
    retriever = Retriever(
        "index_nvembed_v2_musique_full",
        "/projects/prjs1800/external/arag/data/musique/index_nvembed_v2_musique_full/query_embeddings.pkl"
    )

    print("Loading LLaDA-8B-Instruct...", flush=True)
    @dataclass
    class MA:
        model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print(f"  mask_token_id: {tokenizer.mask_token_id}", flush=True)
    print(f"  eos_token_id: {tokenizer.eos_token_id}", flush=True)

    qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[:50]

    for mode in ["baseline", "spread"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Running LLaDA {mode} on 50 MuSiQue questions", flush=True)
        print(f"{'='*60}", flush=True)

        sum_f1, sum_p, sum_cr = 0, 0, 0
        predictions = []

        for i, q in enumerate(qs):
            passages = retriever.retrieve(q["question"])
            context = "\n\n".join(passages)

            # Encode query for SPREAD
            h_q = None
            if mode == "spread":
                q_ids = tokenizer.encode(q["question"], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    q_out = model(q_ids, output_hidden_states=True)
                h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)

            t0 = time.time()
            try:
                answer, stats = generate_llada(model, tokenizer, context, q["question"],
                                                mode=mode, h_q=h_q)
            except Exception as e:
                import traceback; traceback.print_exc()
                answer, stats = "", {}
            elapsed = time.time() - t0

            m = compute_metrics(answer, q["answer"])
            cr = copy_rate(answer, context)
            sum_f1 += m["f1"]; sum_p += m["p"]; sum_cr += cr
            contain = q["answer"].lower() in answer.lower()
            words = len(answer.split())

            predictions.append({
                "id": q["id"], "question": q["question"],
                "gold_answer": q["answer"], "pred_answer": answer,
                "mode": mode, "f1": round(m["f1"], 4), "precision": round(m["p"], 4),
                "contain": contain, "words": words, "cr": round(cr, 4),
            })

            extra = ""
            if stats.get("rel_stds"):
                extra = f" rel_std={stats['rel_stds'][0]:.4f}"
            print(f"[{i+1}/50] ({elapsed:.1f}s) F1={m['f1']:.2f} P={m['p']:.2f} w={words}{extra}", flush=True)
            print(f"  Gold: {q['answer']}  Pred: {answer[:80]}", flush=True)

        n = len(predictions)
        cn = sum(1 for p in predictions if p["contain"])
        out = f"/projects/prjs1800/msc-thesis/07-daes/results/llada_{mode}_musique.jsonl"
        with open(out, "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        print(f"\n{'='*60}", flush=True)
        print(f"LLaDA | {mode} | musique | N={n}", flush=True)
        print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
        print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
        print(f"  CR:        {sum_cr/n*100:.1f}%", flush=True)
        print(f"  Contain:   {cn}/{n} = {cn/n*100:.1f}%", flush=True)
        print(f"  Avg words: {sum(p['words'] for p in predictions)/n:.0f}", flush=True)
        print(f"  SPREAD paper LLaDA: P=23.91 (low-conf) → 26.67 (SPREAD)", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
