"""SPREAD with weighted sum of relevance + confidence (per author clarification).
Tests multiple alpha values: score = alpha * confidence + (1-alpha) * relevance."""
import argparse, json, sys, time, re, string, pickle, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

# Retriever
idx_path = "/projects/prjs1800/external/arag/data/musique/index_e5_musique_full/sentence_index.pkl"
with open(idx_path, "rb") as f:
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

# Model
@dataclass
class MA:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
model = dllm.utils.get_model(model_args=MA()).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=MA())
mask_id = tokenizer.mask_token_id
eos_id = tokenizer.eos_token_id
print("Ready.", flush=True)

# Metrics
def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_metrics(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return 0, 0, 0
    common = set(pt) & set(gt)
    if not common: return 0, 0, 0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return p, r, 2*p*r/(p+r)

def generate_spread_hybrid(context, question, alpha=0.5):
    """SPREAD with weighted sum: score = alpha*confidence + (1-alpha)*relevance."""
    prompt = context + "\n\nQuestion: " + question + "\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * 512
    x = torch.tensor([canvas], dtype=torch.long, device=model.device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=model.device)

    # Encode query separately (confirmed by author)
    q_ids = tokenizer.encode(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        q_out = model(q_ids, output_hidden_states=True)
    h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)

    k = 4  # 512/128
    remaining = 512

    for step in range(128):
        if remaining <= 0: break
        mi = (x == mask_id)
        if not mi.any(): break

        with torch.no_grad():
            out = model(x, attention_mask=attn, output_hidden_states=True)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        hs = out.hidden_states[-1]
        mp = mi[0].nonzero(as_tuple=True)[0]

        # Confidence (neg_entropy, same as Dream entropy baseline)
        conf, x0 = sample_tokens(logits[0, mp], temperature=0.1, neg_entropy=True)

        # Relevance
        h_m = F.normalize(hs[0, mp], dim=-1)
        sim = (h_m @ h_q.squeeze(0)).float()
        rel = torch.sigmoid(sim)

        # Normalize both to [0,1]
        if conf.max() > conf.min():
            conf_norm = (conf - conf.min()) / (conf.max() - conf.min())
        else:
            conf_norm = torch.zeros_like(conf)
        if rel.max() > rel.min():
            rel_norm = (rel - rel.min()) / (rel.max() - rel.min())
        else:
            rel_norm = torch.zeros_like(rel)

        # Weighted sum (per author: "weighted sum of relevance and confidence")
        score = alpha * conf_norm + (1 - alpha) * rel_norm

        nc = min(k, remaining)
        if step == 127: nc = remaining
        _, topk = torch.topk(score, min(nc, len(score)))

        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)

    gen = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return answer

# Run
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=True)
args = parser.parse_args()

qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[:50]

sum_p, sum_r, sum_f1 = 0, 0, 0
cn = 0
for i, q in enumerate(qs):
    passages = retrieve(q["question"])
    context = "\n\n".join(passages)
    t0 = time.time()
    answer = generate_spread_hybrid(context, q["question"], alpha=args.alpha)
    elapsed = time.time() - t0
    p, r, f = compute_metrics(answer, q["answer"])
    sum_p += p; sum_r += r; sum_f1 += f
    if q["answer"].lower() in answer.lower(): cn += 1
    if i < 5 or i == 49:
        print("[%d/50] (%.1fs) P=%.2f F1=%.2f %s" % (i+1, elapsed, p, f, answer[:60]), flush=True)

n = 50
print("\nalpha=%.1f: P=%.1f%% R=%.1f%% F1=%.1f%% Contain=%d/50=%.1f%%" % (
    args.alpha, sum_p/n*100, sum_r/n*100, sum_f1/n*100, cn, cn/n*100), flush=True)
