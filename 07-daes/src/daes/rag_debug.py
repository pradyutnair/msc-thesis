"""Debug script: 1 question, verbose timing, find where it hangs."""
import json, torch, torch.nn.functional as F, sys, time, pickle, numpy as np
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")

print("Step 1: Loading retriever...", flush=True)
t0 = time.time()
with open("/projects/prjs1800/external/arag/data/musique/index_e5_base_v2/sentence_index.pkl", "rb") as f:
    idx = pickle.load(f)
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
print(f"  Retriever loaded in {time.time()-t0:.1f}s. {len(idx['sentences'])} sentences", flush=True)

print("Step 2: Loading Dream-7B...", flush=True)
t0 = time.time()
import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from dataclasses import dataclass

@dataclass
class MA:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
model = dllm.utils.get_model(model_args=MA()).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=MA())
print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

# Load 1 question
q = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[0]
print(f"  Q: {q['question']}", flush=True)
print(f"  Gold: {q['answer']}", flush=True)

# Retrieve top-5
print("Step 3: Retrieving...", flush=True)
q_emb = st_model.encode([q["question"]], normalize_embeddings=True)[0]
sims = np.dot(idx["embeddings"], q_emb)
top_ids = np.argsort(sims)[::-1][:15]
chunk_scores = {}
for i in top_ids:
    cid = idx["sentence_to_chunk"][i]
    if cid not in chunk_scores or sims[i] > chunk_scores[cid]:
        chunk_scores[cid] = float(sims[i])
ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:5]
passages = []
for j, (cid, score) in enumerate(ranked):
    text = idx["chunks"][cid]["text"]
    passages.append(f"[Passage {j+1}] {text}")
evidence = "\n\n".join(passages)
print(f"  Retrieved {len(ranked)} passages, total chars: {len(evidence)}", flush=True)

# Build canvas
print("Step 4: Building canvas...", flush=True)
prompt = f"Answer the question based on the given information.\n\n{evidence}\n\nQuestion: {q['question']}\n\nAnswer:"
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_ids = tokenizer.encode(input_text, add_special_tokens=False)
mask_id = tokenizer.mask_token_id
eos_id = tokenizer.eos_token_id
n_mask = 128
canvas = prompt_ids + [mask_id] * n_mask
print(f"  Canvas: {len(prompt_ids)} prompt + {n_mask} masks = {len(canvas)} total tokens", flush=True)

# Generate with 32 steps, NO output_hidden_states first (baseline)
print("Step 5: Baseline generation (32 steps, no hidden states)...", flush=True)
x = torch.tensor([canvas], dtype=torch.long, device=model.device)
attn = torch.ones_like(x)
tps = max(1, n_mask // 32)
remaining = n_mask

for step in range(32):
    if remaining <= 0:
        break
    mi = (x == mask_id)
    if not mi.any():
        break
    t0 = time.time()
    with torch.no_grad():
        out = model(x, attention_mask=attn)
    fwd_time = time.time() - t0
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    mp = mi[0].nonzero(as_tuple=True)[0]
    conf, x0 = sample_tokens(logits[0, mp], temperature=0.1, neg_entropy=True)
    k = min(tps, remaining)
    if step == 31:
        k = remaining
    _, ti = torch.topk(conf, min(k, len(conf)))
    x[0, mp[ti]] = x0[ti]
    remaining -= len(ti)
    if step < 3 or step == 31:
        print(f"  Step {step}: fwd={fwd_time:.2f}s, committed {len(ti)}, remaining={remaining}", flush=True)

gen = x[0, len(prompt_ids):].tolist()
kept = [t for t in gen if t != mask_id]
answer = tokenizer.decode(kept, skip_special_tokens=True).strip()
n_eos = sum(1 for t in gen if t == eos_id)
print(f"\nBaseline result: {len(kept)} kept tokens ({n_eos} EOS)", flush=True)
print(f"Answer: {answer[:200]}", flush=True)

# Now test with output_hidden_states=True (for SPREAD)
print("\nStep 6: SPREAD generation (32 steps, WITH hidden states)...", flush=True)
x2 = torch.tensor([canvas], dtype=torch.long, device=model.device)
remaining2 = n_mask

# Encode query
q_tokens = tokenizer.encode(q["question"], return_tensors="pt").to(model.device)
with torch.no_grad():
    q_out = model(q_tokens, output_hidden_states=True)
h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)
print("  Query encoded", flush=True)

for step in range(32):
    if remaining2 <= 0:
        break
    mi = (x2 == mask_id)
    if not mi.any():
        break
    t0 = time.time()
    with torch.no_grad():
        out = model(x2, attention_mask=attn, output_hidden_states=True)
    fwd_time = time.time() - t0
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    hs = out.hidden_states[-1]
    mp = mi[0].nonzero(as_tuple=True)[0]
    conf, x0 = sample_tokens(logits[0, mp], temperature=0.1, neg_entropy=True)
    # SPREAD: query relevance
    h_m = F.normalize(hs[0, mp], dim=-1)
    rel = torch.sigmoid((h_m @ h_q.squeeze(0)).float())
    k = min(tps, remaining2)
    if step == 31:
        k = remaining2
    _, ti = torch.topk(rel, min(k, len(rel)))
    x2[0, mp[ti]] = x0[ti]
    remaining2 -= len(ti)
    if step < 3 or step == 31:
        print(f"  Step {step}: fwd={fwd_time:.2f}s, committed {len(ti)}, remaining={remaining2}", flush=True)

gen2 = x2[0, len(prompt_ids):].tolist()
kept2 = [t for t in gen2 if t != mask_id]
answer2 = tokenizer.decode(kept2, skip_special_tokens=True).strip()
n_eos2 = sum(1 for t in gen2 if t == eos_id)
print(f"\nSPREAD result: {len(kept2)} kept tokens ({n_eos2} EOS)", flush=True)
print(f"Answer: {answer2[:200]}", flush=True)
print("\nDone!", flush=True)
