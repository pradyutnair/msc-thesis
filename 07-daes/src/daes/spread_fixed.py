"""
Fixed SPREAD reproduction.

Key fixes:
1. Suppress EOS during denoising (force model to fill all positions with content)
2. Use SPREAD relevance for selection only among non-EOS candidates
3. Measure Precision (SPREAD's Table 2 metric) not just F1

Modes: baseline, spread
"""
import argparse, json, os, sys, time, re, string, pickle
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens


# --- Retriever with precomputed queries ---
class Retriever:
    def __init__(self, dataset, index_name, precomputed_path=None, max_chunk_chars=2000):
        idx_path = f"/projects/prjs1800/external/arag/data/{dataset}/{index_name}/sentence_index.pkl"
        print(f"Loading index from {idx_path}...", flush=True)
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.q_texts = None
        self.q_embs = None
        if precomputed_path:
            with open(precomputed_path, "rb") as f:
                qd = pickle.load(f)
            self.q_texts = qd["questions"]
            self.q_embs = qd["embeddings"]
            print(f"  Loaded {len(self.q_texts)} precomputed query embeddings", flush=True)
        print(f"  Index: {len(self.sentences)} sents, {len(self.chunks)} chunks", flush=True)

    def retrieve(self, query, top_k=5):
        if self.q_texts is not None:
            try:
                qi = self.q_texts.index(query)
                q_emb = self.q_embs[qi]
            except ValueError:
                return []
        else:
            return []
        sims = np.dot(self.embeddings, q_emb)
        top = np.argsort(sims)[::-1][:top_k * 3]
        cb = {}
        for i in top:
            cid = self.sentence_to_chunk[i]
            if cid not in cb or sims[i] > cb[cid]:
                cb[cid] = float(sims[i])
        ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]


# --- Metrics ---
def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_metrics(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return {"p": 0, "r": 0, "f1": 0}
    common = set(pt) & set(gt)
    if not common:
        return {"p": 0, "r": 0, "f1": 0}
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return {"p": p, "r": r, "f1": 2*p*r/(p+r)}

def copy_rate(pred, context):
    pw = set(normalize(pred).split())
    cw = set(normalize(context).split())
    if not pw: return 0
    return len(pw & cw) / len(pw)


# --- Fixed SPREAD generation ---
def generate(model, tokenizer, context, question, L=512, T=128,
             temperature=0.1, mode="baseline"):
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * L
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    # Encode query for SPREAD
    h_q = None
    if mode == "spread":
        q_ids = tokenizer.encode(question, return_tensors="pt").to(device)
        with torch.no_grad():
            q_out = model(q_ids, output_hidden_states=True)
        h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)

    k = max(1, L // T)
    remaining = L

    for step in range(T):
        if remaining <= 0:
            break
        mi = (x == mask_id)
        if not mi.any():
            break

        with torch.no_grad():
            if mode == "spread":
                out = model(x, attention_mask=attn, output_hidden_states=True)
                hs = out.hidden_states[-1]
            else:
                out = model(x, attention_mask=attn)

        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mp = mi[0].nonzero(as_tuple=True)[0]

        # Suppress EOS token — force the model to generate content
        logits[0, mp, eos_id] = -float("inf")

        # Sample tokens (with EOS suppressed, model must predict content)
        conf, x0 = sample_tokens(logits[0, mp], temperature=temperature, neg_entropy=True)

        nc = min(k, remaining)
        if step == T - 1:
            nc = remaining

        if mode == "baseline":
            _, topk = torch.topk(conf, min(nc, len(conf)))
        elif mode == "spread":
            h_m = F.normalize(hs[0, mp], dim=-1)
            rel = torch.sigmoid((h_m @ h_q.squeeze(0)).float())
            _, topk = torch.topk(rel, min(nc, len(rel)))

        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)

    gen = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return answer


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", required=True, choices=["baseline", "spread"])
    parser.add_argument("--index_name", default="index_nvembed_v2_musique_full")
    parser.add_argument("--precomputed_queries", default="/projects/prjs1800/external/arag/data/musique/index_nvembed_v2_musique_full/query_embeddings.pkl")
    parser.add_argument("--L", type=int, default=512)
    parser.add_argument("--T", type=int, default=128)
    args = parser.parse_args()

    os.makedirs("/projects/prjs1800/msc-thesis/07-daes/results", exist_ok=True)
    retriever = Retriever(args.dataset, args.index_name, args.precomputed_queries)

    from dataclasses import dataclass
    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print("Model loaded.", flush=True)

    qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))[:args.n_questions]

    sum_f1, sum_p, sum_r, sum_cr = 0, 0, 0, 0
    predictions = []

    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"])
        context = "\n\n".join(passages)
        t0 = time.time()
        try:
            answer = generate(model, tokenizer, context, q["question"],
                              L=args.L, T=args.T, mode=args.mode)
        except Exception as e:
            import traceback; traceback.print_exc()
            answer = ""
        elapsed = time.time() - t0

        m = compute_metrics(answer, q["answer"])
        cr = copy_rate(answer, context)
        sum_f1 += m["f1"]; sum_p += m["p"]; sum_r += m["r"]; sum_cr += cr
        contain = q["answer"].lower() in answer.lower()
        words = len(answer.split())

        predictions.append({
            "id": q["id"], "question": q["question"],
            "gold_answer": q["answer"], "pred_answer": answer,
            "mode": args.mode, "time": round(elapsed, 2),
            "f1": round(m["f1"], 4), "precision": round(m["p"], 4),
            "contain": contain, "words": words, "cr": round(cr, 4),
        })

        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={m['f1']:.2f} P={m['p']:.2f} CR={cr:.2f} w={words} contain={contain}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:120]}", flush=True)

    n = len(predictions)
    contain_n = sum(1 for p in predictions if p["contain"])
    out = f"/projects/prjs1800/msc-thesis/07-daes/results/fixed_{args.dataset}_{args.mode}.jsonl"
    with open(out, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'='*60}", flush=True)
    print(f"FIXED | {args.mode} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
    print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
    print(f"  CR:        {sum_cr/n*100:.1f}%", flush=True)
    print(f"  Contain:   {contain_n}/{n} = {contain_n/n*100:.1f}%", flush=True)
    print(f"  Avg words: {sum(p['words'] for p in predictions)/n:.0f}", flush=True)
    print(f"  TARGET:    F1~30.6% CR~77.7% (Table 1), P~37.8% (Table 2 SPREAD)", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
