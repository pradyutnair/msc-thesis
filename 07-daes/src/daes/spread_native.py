"""
SPREAD using Dream's NATIVE sample() with hook-based token selection override.

Instead of reimplementing the denoising loop, use DreamSampler.sample() and
inject SPREAD's relevance-based selection via generation_tokens_hook_func.
"""
import argparse, json, os, sys, time, re, string, pickle
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig
from dataclasses import dataclass


# --- Retriever ---
class Retriever:
    def __init__(self, dataset, index_name, precomputed_path=None, max_chunk_chars=2000):
        idx_path = f"/projects/prjs1800/external/arag/data/{dataset}/{index_name}/sentence_index.pkl"
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.q_texts, self.q_embs = None, None
        if precomputed_path:
            with open(precomputed_path, "rb") as f:
                qd = pickle.load(f)
            self.q_texts, self.q_embs = qd["questions"], qd["embeddings"]

    def retrieve(self, query, top_k=5):
        try:
            qi = self.q_texts.index(query)
            q_emb = self.q_embs[qi]
        except (ValueError, AttributeError):
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
    if not pt or not gt: return {"p": 0, "r": 0, "f1": 0}
    common = set(pt) & set(gt)
    if not common: return {"p": 0, "r": 0, "f1": 0}
    p, r = len(common)/len(pt), len(common)/len(gt)
    return {"p": p, "r": r, "f1": 2*p*r/(p+r)}

def copy_rate(pred, context):
    pw = set(normalize(pred).split())
    cw = set(normalize(context).split())
    if not pw: return 0
    return len(pw & cw) / len(pw)


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--mode", required=True, choices=["baseline", "spread"])
    parser.add_argument("--index_name", default="index_nvembed_v2_musique_full")
    parser.add_argument("--precomputed_queries", default="/projects/prjs1800/external/arag/data/musique/index_nvembed_v2_musique_full/query_embeddings.pkl")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    retriever = Retriever(args.dataset, args.index_name, args.precomputed_queries)

    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    sampler = DreamSampler(model=model, tokenizer=tokenizer)
    print("Model loaded.", flush=True)

    # Config for Dream's native sample()
    config = DreamSamplerConfig(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=0.1,
        alg="entropy",
        return_dict=True,
    )

    qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))[:args.n_questions]
    predictions = []
    sum_f1, sum_p, sum_r, sum_cr = 0, 0, 0, 0

    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"])
        context = "\n\n".join(passages)
        prompt = f"{context}\n\nQuestion: {q['question']}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

        t0 = time.time()
        try:
            if args.mode == "baseline":
                # Dream's native sample() — handles everything correctly
                output = sampler.sample([inputs], config)
                seq = output.sequences[0]
                answer = tokenizer.decode(seq[len(inputs):], skip_special_tokens=True).strip()

            elif args.mode == "spread":
                # Encode query
                q_ids = tokenizer.encode(q["question"], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    q_out = model(q_ids, output_hidden_states=True)
                h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)

                # Use infill() with SPREAD hook
                mask_id = tokenizer.mask_token_id
                canvas = inputs + [mask_id] * args.max_new_tokens

                def spread_logits_hook(step, x, logits):
                    """At each step, bias logits toward query-relevant positions."""
                    # Get hidden states for current canvas
                    with torch.no_grad():
                        hs_out = model(x, output_hidden_states=True)
                    hs = hs_out.hidden_states[-1]

                    # Find mask positions
                    mi = (x == mask_id)
                    if not mi.any():
                        return logits
                    mp = mi[0].nonzero(as_tuple=True)[0]

                    # Compute relevance
                    h_m = F.normalize(hs[0, mp], dim=-1)
                    rel = torch.sigmoid((h_m @ h_q.squeeze(0)).float())

                    # Boost logits at high-relevance positions (additive bias)
                    # This biases confidence scoring toward query-relevant positions
                    rel_bias = (rel - rel.mean()) * 10.0  # scale the signal
                    for idx_val, pos in enumerate(mp):
                        logits[0, pos] += rel_bias[idx_val]

                    return logits

                output = sampler.infill([canvas], config,
                                         generation_logits_hook_func=spread_logits_hook)
                seq = output.sequences[0]
                answer = tokenizer.decode(seq[len(inputs):], skip_special_tokens=True).strip()

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
            "mode": args.mode, "f1": round(m["f1"], 4),
            "precision": round(m["p"], 4), "contain": contain,
            "words": words, "cr": round(cr, 4), "time": round(elapsed, 2),
        })
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={m['f1']:.2f} P={m['p']:.2f} CR={cr:.2f} w={words} contain={contain}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:120]}", flush=True)

    n = len(predictions)
    cn = sum(1 for p in predictions if p["contain"])
    out = f"/projects/prjs1800/msc-thesis/07-daes/results/native_{args.dataset}_{args.mode}.jsonl"
    with open(out, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(f"\n{'='*60}", flush=True)
    print(f"NATIVE | {args.mode} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
    print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
    print(f"  CR:        {sum_cr/n*100:.1f}%", flush=True)
    print(f"  Contain:   {cn}/{n} = {cn/n*100:.1f}%", flush=True)
    print(f"  Avg words: {sum(p['words'] for p in predictions)/n:.0f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
