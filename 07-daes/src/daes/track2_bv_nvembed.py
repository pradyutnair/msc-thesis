"""
Track 2: Branch-and-Verify with NV-Embed-v2 retrieval.
Uses pre-computed query embeddings. Tests if branch-verify gains hold with stronger retrieval.
"""
import argparse, json, os, sys, time, re, string, pickle
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from dataclasses import dataclass


class Retriever:
    """E5-base-v2 retriever. One model, one index. No mixing."""
    def __init__(self, dataset, index_name="index_e5_musique_full", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        idx_path = f"/projects/prjs1800/external/arag/data/{dataset}/{index_name}/sentence_index.pkl"
        print(f"Loading index from {idx_path}...", flush=True)
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences, self.embeddings = idx["sentences"], idx["embeddings"]
        self.sentence_to_chunk, self.chunks = idx["sentence_to_chunk"], idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
        print(f"  Index: {len(self.sentences)} sents, {len(self.chunks)} chunks", flush=True)

    def retrieve(self, query, top_k=5):
        """Same E5 model for every query — initial and branch."""
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
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

def compute_f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return 0, 0, 0
    common = set(pt) & set(gt)
    if not common: return 0, 0, 0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return p, r, 2*p*r/(p+r)


def dllm_generate(model, tokenizer, sampler, config, context, question):
    """Generate with Dream's native sample()."""
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    output = sampler.sample([inputs], config)
    seq = output.sequences[0]
    answer = tokenizer.decode(seq[len(inputs):], skip_special_tokens=True).strip()

    # Get confidence from a quick forward pass
    with torch.no_grad():
        out = model(seq.unsqueeze(0))
    logits = out.logits[0, len(inputs)-1:len(inputs)+10]  # first ~10 answer positions
    probs = torch.softmax(logits, dim=-1)
    avg_conf = probs.max(dim=-1).values.mean().item()
    return answer, avg_conf


def extract_candidates(model, tokenizer, context, question, n_candidates=3):
    """Get candidate answers from dLLM token distribution."""
    mask_id = tokenizer.mask_token_id
    prompt = f"{context}\n\nQuestion: {question}\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    canvas = prefix_ids + [mask_id] * 20
    x = torch.tensor([canvas], dtype=torch.long, device=model.device)
    attn = torch.ones_like(x)

    with torch.no_grad():
        out = model(x, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    probs = torch.softmax(logits[0, n_prefix] / 0.3, dim=-1)
    top_probs, top_ids = torch.topk(probs, n_candidates * 3)

    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        x_c = torch.tensor([canvas], dtype=torch.long, device=model.device)
        x_c[0, n_prefix] = tid
        rem = 19
        for step in range(16):
            if rem <= 0: break
            mi = (x_c == mask_id)
            if not mi.any(): break
            with torch.no_grad():
                o2 = model(x_c, attention_mask=attn)
            l2 = torch.cat([o2.logits[:, :1], o2.logits[:, :-1]], dim=1)
            mp = mi[0].nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=True)
            k = min(max(1, rem // 16), rem)
            if step == 15: k = rem
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            rem -= len(tk)

        cand_text = tokenizer.decode(x_c[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
        cand_text = cand_text.split("\n")[0].split(". ")[0].strip()
        if cand_text and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append({"text": cand_text, "init_conf": prob.item()})
            if len(candidates) >= n_candidates: break
    return candidates


def branch_verify(model, tokenizer, sampler, config, retriever, question, n_candidates=3):
    """Branch-and-verify with NV-Embed-v2 retrieval."""
    passages = retriever.retrieve(question, top_k=5)
    context = "\n\n".join(passages)

    candidates = extract_candidates(model, tokenizer, context, question, n_candidates)
    if not candidates:
        answer, conf = dllm_generate(model, tokenizer, sampler, config, context, question)
        return answer, conf, {"method": "fallback"}

    best_answer, best_score, best_idx = "", -1, -1
    for idx, cand in enumerate(candidates):
        hop2_query = f"{question} {cand['text']}"
        hop2_passages = retriever.retrieve(hop2_query, top_k=3)
        expanded = context + "\n\n" + "\n\n".join(hop2_passages)
        answer, verified_conf = dllm_generate(model, tokenizer, sampler, config, expanded, question)
        conf_gain = verified_conf - cand["init_conf"]
        score = verified_conf + 0.5 * conf_gain
        cand["verified_conf"] = verified_conf
        cand["score"] = score
        cand["answer"] = answer
        if score > best_score:
            best_score, best_answer, best_idx = score, answer, idx

    stats = {
        "method": "branch_verify",
        "candidates": [{"text": c["text"][:40], "score": round(c.get("score", 0), 3)} for c in candidates],
        "selected": best_idx,
    }
    return best_answer, best_score, stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=50)
    parser.add_argument("--mode", type=str, default=None, help="If set, run only this mode")
    parser.add_argument("--dataset", type=str, default="musique", choices=["musique", "hotpotqa", "2wikimultihop"])
    args = parser.parse_args()

    ds = args.dataset
    index_map = {
        "musique": "index_e5_musique_full",
        "hotpotqa": "index_e5_full",
        "2wikimultihop": "index_e5_full",
    }
    retriever = Retriever(ds, index_name=index_map[ds])

    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    sampler = DreamSampler(model=model, tokenizer=tokenizer)
    config = DreamSamplerConfig(steps=128, max_new_tokens=512, temperature=0.1, alg="entropy", return_dict=True)
    print("Model loaded.", flush=True)

    all_qs = json.load(open(f"/projects/prjs1800/external/arag/data/{ds}/questions.json"))
    qs = all_qs[args.start_idx:args.end_idx]
    print(f"Running questions {args.start_idx}-{args.end_idx} ({len(qs)} questions)", flush=True)

    modes = [args.mode] if args.mode else ["baseline", "branch_verify"]
    for run_mode in modes:
        print(f"\n{'='*60}\nRunning {run_mode} on 50 MuSiQue (NV-Embed-v2)\n{'='*60}", flush=True)
        sum_f1, sum_p = 0, 0
        predictions = []

        for i, q in enumerate(qs):
            t0 = time.time()
            if run_mode == "baseline":
                passages = retriever.retrieve(q["question"])
                context = "\n\n".join(passages)
                answer, conf = dllm_generate(model, tokenizer, sampler, config, context, q["question"])
                stats = {"method": "baseline"}
            else:
                answer, conf, stats = branch_verify(model, tokenizer, sampler, config, retriever, q["question"])
            elapsed = time.time() - t0

            p, r, f = compute_f1(answer, q["answer"])
            sum_f1 += f; sum_p += p
            contain = q["answer"].lower() in answer.lower()
            predictions.append({
                "id": q["id"], "question": q["question"],
                "gold_answer": q["answer"], "pred_answer": answer,
                "mode": run_mode, "f1": round(f, 4), "contain": contain,
            })
            with open(out, 'a') as _fw:
                _fw.write(json.dumps(predictions[-1]) + '\n')
            print(f"[{i+1}/50] ({elapsed:.1f}s) F1={f:.2f} contain={contain} {answer[:60]}", flush=True)

        n = len(predictions)
        cn = sum(1 for p in predictions if p["contain"])
        out = f"/projects/prjs1800/msc-thesis/07-daes/results/scale_{ds}_{run_mode}_{args.start_idx}_{args.end_idx}.jsonl"
        open(out, 'w').close()  # init for incremental writes
        with open(out, "w") as f:
            for p in predictions: f.write(json.dumps(p) + "\n")
        print(f"\n{'='*60}", flush=True)
        print(f"NV-BV | {run_mode} | musique | N={n}", flush=True)
        print(f"  F1:      {sum_f1/n*100:.1f}%", flush=True)
        print(f"  Prec:    {sum_p/n*100:.1f}%", flush=True)
        print(f"  Contain: {cn}/{n} = {cn/n*100:.1f}%", flush=True)
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
