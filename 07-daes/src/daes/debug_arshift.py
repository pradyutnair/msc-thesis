"""Test: SPREAD with vs without AR shift on 3 questions."""
import json, torch, pickle, numpy as np, torch.nn.functional as F, sys, re, string
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from dataclasses import dataclass

with open("/projects/prjs1800/external/arag/data/musique/index_nvembed_v2_musique_full/sentence_index.pkl", "rb") as f:
    idx = pickle.load(f)
with open("/projects/prjs1800/external/arag/data/musique/index_nvembed_v2_musique_full/query_embeddings.pkl", "rb") as f:
    qdata = pickle.load(f)

@dataclass
class MA:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
model = dllm.utils.get_model(model_args=MA()).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=MA())
mask_id = tokenizer.mask_token_id
eos_id = tokenizer.eos_token_id

def retrieve(question):
    qi = qdata["questions"].index(question)
    q_emb = qdata["embeddings"][qi]
    sims = np.dot(idx["embeddings"], q_emb)
    top = np.argsort(sims)[::-1][:15]
    cb = {}
    for i in top:
        cid = idx["sentence_to_chunk"][i]
        if cid not in cb or sims[i] > cb[cid]:
            cb[cid] = float(sims[i])
    ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:5]
    return [idx["chunks"][cid]["text"][:2000] for cid, _ in ranked]

def run(q, mode, ar_shift):
    passages = retrieve(q["question"])
    context = "\n\n".join(passages)
    prompt = context + "\n\nQuestion: " + q["question"] + "\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    canvas = prefix_ids + [mask_id] * 512
    x = torch.tensor([canvas], dtype=torch.long, device=model.device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=model.device)

    h_q = None
    if mode == "spread":
        q_ids = tokenizer.encode(q["question"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            q_out = model(q_ids, output_hidden_states=True)
        h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)

    k = 4
    remaining = 512
    for step in range(128):
        if remaining <= 0:
            break
        mi = (x == mask_id)
        if not mi.any():
            break
        with torch.no_grad():
            out = model(x, attention_mask=attn, output_hidden_states=True)
        logits = out.logits
        if ar_shift:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        hs = out.hidden_states[-1]
        mp = mi[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mp], temperature=0.1, neg_entropy=True)
        nc = min(k, remaining)
        if step == 127:
            nc = remaining
        if mode == "baseline":
            _, topk = torch.topk(conf, min(nc, len(conf)))
        else:
            h_m = F.normalize(hs[0, mp], dim=-1)
            rel = torch.sigmoid((h_m @ h_q.squeeze(0)).float())
            _, topk = torch.topk(rel, min(nc, len(rel)))
        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)

    gen = x[0, n_prefix:].tolist()
    n_eos = sum(1 for t in gen if t == eos_id)
    n_content = 512 - n_eos
    answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return answer, n_content, n_eos

qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[:3]
for i, q in enumerate(qs):
    print("Q%d: %s" % (i+1, q["question"][:65]), flush=True)
    print("Gold: %s" % q["answer"], flush=True)
    for mode in ["baseline", "spread"]:
        for shift in [True, False]:
            ans, nc, ne = run(q, mode, shift)
            label = "%s shift=%s" % (mode, shift)
            print("  %-25s content=%3d eos=%3d answer: %s" % (label, nc, ne, ans[:60]), flush=True)
    print(flush=True)
