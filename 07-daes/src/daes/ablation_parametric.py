"""Ablation: does the dLLM use retrieved evidence or parametric knowledge?"""
import json, sys, torch, pickle, numpy as np
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

with open("/projects/prjs1800/external/arag/data/musique/index_e5_musique_full/sentence_index.pkl", "rb") as f:
    idx = pickle.load(f)
st = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

def retrieve(q, top_k=5):
    q_emb = st.encode([q], normalize_embeddings=True)[0]
    sims = np.dot(idx["embeddings"], q_emb)
    top = np.argsort(sims)[::-1][:top_k*3]
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
mask_id = tokenizer.mask_token_id

def get_top_tokens(context, q, n=5):
    if context:
        prompt = context + "\n\nQuestion: " + q + "\n\nThe answer is:"
    else:
        prompt = "Question: " + q + "\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    canvas = ids + [mask_id] * 20
    x = torch.tensor([canvas], dtype=torch.long, device=model.device)
    attn = torch.ones_like(x)
    with torch.no_grad():
        out = model(x, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    probs = torch.softmax(logits[0, len(ids)] / 0.3, dim=-1)
    top_p, top_i = torch.topk(probs, n)
    return [(tokenizer.decode([t]).strip(), round(p.item(), 3)) for t, p in zip(top_i, top_p)]

qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[:20]
same, diff = 0, 0
print("Ablation: parametric vs evidence-grounded candidates (20 questions)", flush=True)
print("=" * 60, flush=True)
for i, q in enumerate(qs):
    passages = retrieve(q["question"])
    ctx = "\n\n".join(passages)
    with_ctx = get_top_tokens(ctx, q["question"])
    without_ctx = get_top_tokens("", q["question"])
    top1_same = with_ctx[0][0] == without_ctx[0][0]
    if top1_same: same += 1
    else: diff += 1
    w = [t for t,_ in with_ctx[:3]]
    wo = [t for t,_ in without_ctx[:3]]
    print("[%d] %s" % (i+1, q["question"][:65]), flush=True)
    print("  Gold: %s" % q["answer"], flush=True)
    print("  WITH context:    %s" % w, flush=True)
    print("  WITHOUT context: %s" % wo, flush=True)
    print("  Top-1 changed: %s" % (not top1_same), flush=True)
    print(flush=True)

print("=" * 60, flush=True)
print("Top-1 SAME: %d/20 (%d%%)" % (same, same*5), flush=True)
print("Top-1 CHANGED: %d/20 (%d%%)" % (diff, diff*5), flush=True)
