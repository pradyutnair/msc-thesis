"""
Proper SFT eval using the SAME generation code as aram_reproduce.py baseline.
Loads model via dllm.utils.get_model() (proven to get 27.8% F1),
then applies LoRA adapter on top.
"""
import argparse, json, os, sys, time, re, string, pickle
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens

class Retriever:
    def __init__(self, dataset, index_name="index_e5_base_v2", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        data_dir = f"/projects/prjs1800/external/arag/data/{dataset}"
        idx_path = os.path.join(data_dir, index_name, "sentence_index.pkl")
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
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt: return 0, 0, 0
    common = set(pt) & set(gt)
    if not common: return 0, 0, 0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return p, r, 2*p*r/(p+r)

def generate(model, tokenizer, context, question, steps=128, n_tokens=512, temperature=0.1):
    """EXACT same generation as aram_reproduce.py baseline."""
    device = model.device
    mask_id = tokenizer.mask_token_id
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)
    k = max(1, n_tokens // steps)
    remaining = n_tokens
    for step in range(steps):
        if remaining <= 0: break
        mi = (x[0] == mask_id)
        mi[:n_prefix] = False
        if not mi.any(): break
        mp = mi.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        conf, x0 = sample_tokens(logits[0, mp], temperature=temperature, neg_entropy=True)
        nc = min(k, remaining)
        if step == steps - 1: nc = remaining
        _, topk = torch.topk(conf, min(nc, len(conf)))
        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)
    return tokenizer.decode(x[0, n_prefix:].tolist(), skip_special_tokens=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    args = parser.parse_args()

    index_map = {"musique": "index_e5_musique_full", "hotpotqa": "index_e5_full", "2wikimultihop": "index_e5_full"}
    retriever = Retriever(args.dataset, index_name=index_map[args.dataset])

    from dataclasses import dataclass
    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...", flush=True)
        if not hasattr(model, 'prepare_inputs_for_generation'):
            model.prepare_inputs_for_generation = lambda *a, **k: {}
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        model = model.cuda().eval()
        tag = "sft"
    else:
        tag = "base"

    print(f"Model loaded ({tag}). Running {args.n_questions}q on {args.dataset}", flush=True)
    qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))[:args.n_questions]

    out_path = os.path.join(args.output_dir, f"proper_{tag}_{args.dataset}_{args.n_questions}q.jsonl")
    open(out_path, "w").close()
    sum_f1 = 0
    preds = []
    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"], top_k=5)
        context = "\n\n".join(passages)
        t0 = time.time()
        answer = generate(model, tokenizer, context, q["question"])
        elapsed = time.time() - t0
        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f
        contain = q["answer"].lower() in answer.lower()
        pred = {"id": q["id"], "gold": q["answer"], "pred": answer, "f1": round(f,4), "contain": contain}
        preds.append(pred)
        with open(out_path, "a") as fw: fw.write(json.dumps(pred) + "\n")
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain} | {answer[:80]}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)

    n = len(preds)
    cn = sum(1 for p in preds if p["contain"])
    print(f"\n{'='*60}", flush=True)
    print(f"PROPER EVAL | {tag} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:      {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Contain: {cn}/{n} = {cn/n*100:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
