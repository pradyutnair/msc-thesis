"""Candidate count ablation: how many candidates are optimal? n=1,2,3,5."""
import argparse, json, os, sys, time, re, string, pickle, random, numpy as np, torch
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data")
parser.add_argument("--n_questions", type=int, default=50)
parser.add_argument("--n_candidates", type=int, required=True)
args = parser.parse_args()

with open(os.path.join(args.data_dir, "musique/index_e5_musique_full/sentence_index.pkl"), "rb") as f:
    idx = pickle.load(f)
e5 = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

def retrieve(query, top_k=5):
    q_emb = e5.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(idx["embeddings"], q_emb)
    top = np.argsort(sims)[::-1][:top_k * 3]
    cb = {}
    for i in top:
        cid = idx["sentence_to_chunk"][i]
        if cid not in cb or sims[i] > cb[cid]: cb[cid] = float(sims[i])
    ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [idx["chunks"][cid]["text"][:2000] for cid, _ in ranked]

@dataclass
class MA:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
model = dllm.utils.get_model(model_args=MA()).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=MA())
sampler = DreamSampler(model=model, tokenizer=tokenizer)
config = DreamSamplerConfig(steps=128, max_new_tokens=512, temperature=0.1, alg="entropy", return_dict=True)
mask_id = tokenizer.mask_token_id

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

def generate(context, question):
    prompt = context + "\n\nQuestion: " + question + "\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    output = sampler.sample([inputs], config)
    return tokenizer.decode(output.sequences[0][len(inputs):], skip_special_tokens=True).strip()

def get_candidates(context, question, n):
    prompt = context + "\n\nQuestion: " + question + "\n\nThe answer is:"
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
    top_probs, top_ids = torch.topk(probs, n * 3)
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
        text = tokenizer.decode(x_c[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append(text)
            if len(candidates) >= n: break
    return candidates

qs = json.load(open(os.path.join(args.data_dir, "musique/questions.json")))[:args.n_questions]
out_path = "results/ablation_ncandidates_%d.jsonl" % args.n_candidates
os.makedirs("results", exist_ok=True)
open(out_path, "w").close()

sf, cn = 0, 0
for i, q in enumerate(qs):
    t0 = time.time()
    passages = retrieve(q["question"], top_k=5)
    context = "\n\n".join(passages)
    cands = get_candidates(context, q["question"], n=args.n_candidates)
    if not cands:
        answer = generate(context, q["question"])
    else:
        c = random.choice(cands)
        hop2 = retrieve(q["question"] + " " + c, top_k=3)
        expanded = context + "\n\n" + "\n\n".join(hop2)
        answer = generate(expanded, q["question"])
    elapsed = time.time() - t0
    sc = f1(answer, q["answer"])
    sf += sc
    if q["answer"].lower() in answer.lower(): cn += 1
    result = {"n_candidates": args.n_candidates, "id": q["id"],
              "gold_answer": q["answer"], "pred_answer": answer,
              "f1": round(sc, 4), "contain": q["answer"].lower() in answer.lower()}
    with open(out_path, "a") as fw:
        fw.write(json.dumps(result) + "\n")
    if i < 3 or i == args.n_questions - 1:
        print("[%d/%d] (%.1fs) F1=%.2f %s" % (i+1, args.n_questions, elapsed, sc, answer[:50]), flush=True)

print("\nn_candidates=%d: F1=%.1f%% Contain=%d/%d=%.1f%%" % (
    args.n_candidates, sf/len(qs)*100, cn, len(qs), cn/len(qs)*100), flush=True)
