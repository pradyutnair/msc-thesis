"""
ARAM on LLaDA-8B. Adapted from aram_reproduce.py.
Key differences from Dream version:
- mask_token_id = 126336 (not 151666)
- NO AR-shift (LLaDA predicts position i directly)
- Model loaded via AutoModel (not dllm.utils)
"""
import argparse, json, os, sys, time, re, string, pickle
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class Retriever:
    def __init__(self, dataset, index_name="index_e5_musique_full", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        idx_path = f"/projects/prjs1800/external/arag/data/{dataset}/{index_name}/sentence_index.pkl"
        with open(idx_path, "rb") as f: idx = pickle.load(f)
        self.sentences, self.embeddings = idx["sentences"], idx["embeddings"]
        self.sentence_to_chunk, self.chunks = idx["sentence_to_chunk"], idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k * 3]
        cb = {}
        for i in top_idx:
            cid = self.sentence_to_chunk[i]
            if cid not in cb or sims[i] > cb[cid]: cb[cid] = float(sims[i])
        ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]

def normalize_answer(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation); return " ".join(s.split())
def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt: return 0, 0, 0
    common = set(pt) & set(gt)
    if not common: return 0, 0, 0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return p, r, 2*p*r/(p+r)

MASK_ID = 126336

def build_cond_and_prior(tokenizer, context, question, n_tokens):
    prompt_cond = ("You are a helpful assistant. Give a direct, concise answer (1-10 words). "
                   "No explanations.\n\n"
                   "Example:\nContext: The Eiffel Tower is in Paris.\n"
                   "Question: Where is the Eiffel Tower?\nAnswer: Paris, France\n\n"
                   f"Context:\n{context}\n\nQuestion: {question}\nAnswer:")
    msg_cond = [{"role": "user", "content": prompt_cond}]
    text_cond = tokenizer.apply_chat_template(msg_cond, tokenize=False, add_generation_prompt=True)
    prefix_cond = tokenizer.encode(text_cond, add_special_tokens=False)

    prompt_prior = ("You are a helpful assistant. Give a direct, concise answer (1-10 words). "
                    "No explanations.\n\n"
                    "Example:\nContext: The Eiffel Tower is in Paris.\n"
                    "Question: Where is the Eiffel Tower?\nAnswer: Paris, France\n\n"
                    f"Context:\nNo relevant context available.\n\nQuestion: {question}\nAnswer:")
    msg_prior = [{"role": "user", "content": prompt_prior}]
    text_prior = tokenizer.apply_chat_template(msg_prior, tokenize=False, add_generation_prompt=True)
    prefix_prior = tokenizer.encode(text_prior, add_special_tokens=False)

    n_ctx = len(prefix_cond) - len(prefix_prior)
    ctx_start = 0
    for i in range(min(len(prefix_cond), len(prefix_prior))):
        if prefix_cond[i] != prefix_prior[i]:
            ctx_start = i; break
    ctx_end = ctx_start + n_ctx
    prior_prefix = list(prefix_cond)
    for i in range(ctx_start, ctx_end):
        prior_prefix[i] = MASK_ID

    cond_ids = prefix_cond + [MASK_ID] * n_tokens
    prior_ids = prior_prefix + [MASK_ID] * n_tokens
    assert len(cond_ids) == len(prior_ids)
    return cond_ids, prior_ids, len(prefix_cond)

def generate(model, tokenizer, context, question, n_tokens=512, steps=128,
             temperature=0.1, lambda_max=1.0, beta=0.1, eps=1e-6, mode="aram"):
    device = model.device
    cond_ids, prior_ids, n_prefix = build_cond_and_prior(tokenizer, context, question, n_tokens)
    seq_len = len(cond_ids)
    x = torch.tensor([cond_ids], dtype=torch.long, device=device)
    attn = torch.ones((1, seq_len), dtype=torch.long, device=device)

    if mode == "aram":
        x_prior = torch.tensor([prior_ids], dtype=torch.long, device=device)
        attn_prior = torch.ones((1, seq_len), dtype=torch.long, device=device)

    k = max(1, n_tokens // steps); remaining = n_tokens
    signal_traj, lambda_traj = [], []

    for step in range(steps):
        if remaining <= 0: break
        mi = (x[0] == MASK_ID); mi[:n_prefix] = False
        if not mi.any(): break
        mp = mi.nonzero(as_tuple=True)[0]; n_masked = len(mp)

        if mode == "baseline":
            with torch.no_grad():
                out = model(x, attention_mask=attn)
            logits = out.logits  # NO AR-shift for LLaDA
            mask_logits = logits[0, mp]
            probs = torch.softmax(mask_logits / max(temperature, 1e-8), dim=-1)
            confidence = probs.max(dim=-1).values
            x0 = probs.argmax(dim=-1)
        elif mode == "aram":
            x_prior[0, n_prefix:] = x[0, n_prefix:]
            x_batch = torch.cat([x, x_prior], dim=0)
            attn_batch = torch.cat([attn, attn_prior], dim=0)
            with torch.no_grad():
                out = model(x_batch, attention_mask=attn_batch)
            logits_cond = out.logits[0]  # NO AR-shift
            logits_prior = out.logits[1]

            log_p_cond = F.log_softmax(logits_cond[mp], dim=-1)
            log_p_prior = F.log_softmax(logits_prior[mp], dim=-1)
            p_cond = log_p_cond.exp(); p_prior = log_p_prior.exp()

            kl_fwd = (p_cond * (log_p_cond - log_p_prior)).sum(dim=-1)
            kl_rev = (p_prior * (log_p_prior - log_p_cond)).sum(dim=-1)
            signal_per_token = kl_fwd + kl_rev
            noise_per_token = -(p_cond * log_p_cond).sum(dim=-1)
            lambda_per_token = lambda_max * torch.tanh(beta * signal_per_token / (noise_per_token + eps))

            lam = lambda_per_token.unsqueeze(-1)
            logits_guided = logits_prior[mp] + lam * (logits_cond[mp] - logits_prior[mp])

            probs = torch.softmax(logits_guided / max(temperature, 1e-8), dim=-1)
            confidence = probs.max(dim=-1).values
            x0 = probs.argmax(dim=-1)

            signal_traj.append(signal_per_token.mean().item())
            lambda_traj.append(lambda_per_token.mean().item())

        nc = min(k, remaining)
        if step == steps - 1: nc = remaining
        _, topk = torch.topk(confidence, min(nc, n_masked))
        x[0, mp[topk]] = x0[topk]; remaining -= len(topk)

    answer = tokenizer.decode(x[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
    stats = {"mode": mode}
    if signal_traj:
        stats["mean_signal"] = round(float(np.mean(signal_traj)), 4)
        stats["mean_lambda"] = round(float(np.mean(lambda_traj)), 4)
    return answer, stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--mode", required=True, choices=["baseline", "aram"])
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    args = parser.parse_args()

    index_map = {"musique": "index_e5_musique_full", "hotpotqa": "index_e5_full", "2wikimultihop": "index_e5_full"}
    retriever = Retriever(args.dataset, index_name=index_map[args.dataset])

    print(f"Loading GSAI-ML/LLaDA-8B-Instruct...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    print(f"Loaded. Mode={args.mode} beta={args.beta}", flush=True)

    qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))[:args.n_questions]
    out_path = os.path.join(args.output_dir, f"llada_aram_{args.mode}_{args.dataset}_{args.n_questions}q.jsonl")
    open(out_path, "w").close()
    sum_f1 = 0
    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"], top_k=5)
        context = "\n\n".join(passages)
        t0 = time.time()
        answer, stats = generate(model, tokenizer, context, q["question"], mode=args.mode,
                                  lambda_max=args.lambda_max, beta=args.beta)
        elapsed = time.time() - t0
        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f; contain = q["answer"].lower() in answer.lower()
        pred = {"id": q["id"], "gold": q["answer"], "pred": answer, "f1": round(f,4), "contain": contain, "stats": stats}
        with open(out_path, "a") as fw: fw.write(json.dumps(pred) + "\n")
        extra = f" sig={stats.get('mean_signal','')}" if "mean_signal" in stats else ""
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain} words={len(answer.split())}{extra}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:100]}", flush=True)

    n = len(qs); cn = sum(1 for line in open(out_path) if json.loads(line)["contain"])
    print(f"\n{'='*60}\nLLaDA ARAM | {args.mode} | {args.dataset} | N={n}\n  F1: {sum_f1/n*100:.1f}%\n  Contain: {cn}/{n} = {cn/n*100:.1f}%\n{'='*60}", flush=True)

if __name__ == "__main__":
    main()
