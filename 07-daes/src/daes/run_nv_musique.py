"""Branch-verify with NV-Embed-v2 for ALL retrieval (initial + branch queries).
Single venv, single retriever, clean setup."""
import argparse, json, sys, time, re, string, pickle, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from transformers import AutoModel, AutoTokenizer as NVTokenizer
from dataclasses import dataclass


class NVRetriever:
    """NV-Embed-v2 retriever. Same model for ALL queries."""
    def __init__(self, index_path, max_chunk_chars=2000):
        with open(index_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        print(f"Loading NV-Embed-v2...", flush=True)
        self.model = AutoModel.from_pretrained(
            "nvidia/NV-Embed-v2", trust_remote_code=True, torch_dtype=torch.float16
        ).cuda().eval()
        self.query_instruction = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
        print(f"Index: {len(self.sentences)} sents, {len(self.chunks)} chunks", flush=True)

    def encode_query(self, query):
        with torch.no_grad():
            emb = self.model.encode([query], instruction=self.query_instruction, max_length=512)
            return F.normalize(emb, dim=-1).cpu().numpy()[0]

    def retrieve(self, query, top_k=5):
        q_emb = self.encode_query(query)
        sims = np.dot(self.embeddings, q_emb)
        top = np.argsort(sims)[::-1][:top_k * 3]
        cb = {}
        for i in top:
            cid = self.sentence_to_chunk[i]
            if cid not in cb or sims[i] > cb[cid]:
                cb[cid] = float(sims[i])
        ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]


def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return 0
    common = set(pt) & set(gt)
    if not common: return 0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return 2*p*r/(p+r)


def dllm_generate(dream_sampler, dream_config, dream_tokenizer, context, question):
    prompt = context + "\n\nQuestion: " + question + "\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    inputs = dream_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    output = dream_sampler.sample([inputs], dream_config)
    seq = output.sequences[0]
    answer = dream_tokenizer.decode(seq[len(inputs):], skip_special_tokens=True).strip()
    # Confidence from first 10 answer tokens
    with torch.no_grad():
        out = dream_sampler.model(seq.unsqueeze(0))
    logits = out.logits[0, len(inputs)-1:len(inputs)+10]
    probs = torch.softmax(logits, dim=-1)
    avg_conf = probs.max(dim=-1).values.mean().item()
    return answer, avg_conf


def extract_candidates(dream_model, dream_tokenizer, context, question, n=3):
    mask_id = dream_tokenizer.mask_token_id
    prompt = context + "\n\nQuestion: " + question + "\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = dream_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = dream_tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    canvas = prefix_ids + [mask_id] * 20
    x = torch.tensor([canvas], dtype=torch.long, device=dream_model.device)
    attn = torch.ones_like(x)
    with torch.no_grad():
        out = dream_model(x, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    probs = torch.softmax(logits[0, n_prefix] / 0.3, dim=-1)
    top_probs, top_ids = torch.topk(probs, n * 3)
    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        x_c = torch.tensor([canvas], dtype=torch.long, device=dream_model.device)
        x_c[0, n_prefix] = tid
        rem = 19
        for step in range(16):
            if rem <= 0: break
            mi = (x_c == mask_id)
            if not mi.any(): break
            with torch.no_grad():
                o2 = dream_model(x_c, attention_mask=attn)
            l2 = torch.cat([o2.logits[:, :1], o2.logits[:, :-1]], dim=1)
            mp = mi[0].nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=True)
            k = min(max(1, rem // 16), rem)
            if step == 15: k = rem
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            rem -= len(tk)
        text = dream_tokenizer.decode(x_c[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append({"text": text, "init_conf": prob.item()})
            if len(candidates) >= n: break
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument("--mode", required=True, choices=["baseline", "branch_verify"])
    args = parser.parse_args()

    retriever = NVRetriever(
        "/projects/prjs1800/external/arag/data/musique/index_nvembed_v2_musique_full/sentence_index.pkl"
    )

    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    dream_model = dllm.utils.get_model(model_args=MA()).eval()
    dream_tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    dream_sampler = DreamSampler(model=dream_model, tokenizer=dream_tokenizer)
    dream_config = DreamSamplerConfig(steps=128, max_new_tokens=512, temperature=0.1, alg="entropy", return_dict=True)
    print("Both models loaded.", flush=True)

    qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[args.start_idx:args.end_idx]
    print(f"Running {args.mode} on questions {args.start_idx}-{args.end_idx} ({len(qs)}q)", flush=True)

    predictions = []
    sf = 0

    for i, q in enumerate(qs):
        t0 = time.time()
        if args.mode == "baseline":
            passages = retriever.retrieve(q["question"], top_k=5)
            answer, conf = dllm_generate(dream_sampler, dream_config, dream_tokenizer,
                                          "\n\n".join(passages), q["question"])
        else:
            passages = retriever.retrieve(q["question"], top_k=5)
            context = "\n\n".join(passages)
            cands = extract_candidates(dream_model, dream_tokenizer, context, q["question"])
            if not cands:
                answer, conf = dllm_generate(dream_sampler, dream_config, dream_tokenizer,
                                              context, q["question"])
            else:
                best_answer, best_score = "", -1
                for c in cands:
                    hop2 = retriever.retrieve(q["question"] + " " + c["text"], top_k=3)
                    expanded = context + "\n\n" + "\n\n".join(hop2)
                    ans, verified_conf = dllm_generate(dream_sampler, dream_config, dream_tokenizer,
                                                        expanded, q["question"])
                    score = verified_conf + 0.5 * (verified_conf - c["init_conf"])
                    if score > best_score:
                        best_score, best_answer = score, ans
                answer = best_answer
        elapsed = time.time() - t0
        sc = f1(answer, q["answer"])
        sf += sc
        contain = q["answer"].lower() in answer.lower()
        predictions.append({
            "id": q["id"], "question": q["question"],
            "gold_answer": q["answer"], "pred_answer": answer,
            "mode": args.mode, "f1": round(sc, 4), "contain": contain,
        })
        with open(out, 'a') as _fw:
            _fw.write(json.dumps(predictions[-1]) + '\n')
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={sc:.2f} contain={contain} {answer[:50]}", flush=True)

    out = f"/projects/prjs1800/msc-thesis/07-daes/results/nv1k_{args.mode}_{args.start_idx}_{args.end_idx}.jsonl"
    open(out, 'w').close()  # init for incremental writes
    with open(out, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    n = len(predictions)
    cn = sum(1 for p in predictions if p["contain"])
    print(f"\n{args.mode} [{args.start_idx}-{args.end_idx}]: F1={sf/n*100:.1f}% Contain={cn}/{n}={cn/n*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
