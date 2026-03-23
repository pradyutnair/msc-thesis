"""Build MuSiQue index with NV-Embed-v2 using direct HuggingFace API (no sentence-transformers)."""
import json, pickle, numpy as np, time, os, torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

print("Step 1: Loading MuSiQue paragraphs...", flush=True)
data = [json.loads(l) for l in open("/projects/prjs1800/datasets/musique/musique_full_v1.0_dev.jsonl")]
seen = set()
paragraphs = []
for d in data:
    for p in d["paragraphs"]:
        key = (p.get("title", ""), p.get("paragraph_text", ""))
        if key not in seen:
            seen.add(key)
            paragraphs.append({"title": p.get("title", ""), "text": p.get("paragraph_text", "")})
print(f"  {len(paragraphs)} unique paragraphs", flush=True)

sentences, s2c, chunks = [], [], {}
for i, p in enumerate(paragraphs):
    cid = str(i)
    chunks[cid] = {"id": cid, "text": "[" + p["title"] + "] " + p["text"]}
    text = p["text"].strip()
    if not text:
        continue
    for s in [s.strip() for s in text.replace(". ", ".\n").split("\n") if s.strip()]:
        sentences.append(s)
        s2c.append(cid)
print(f"  {len(sentences)} sentences, {len(chunks)} chunks", flush=True)

print("Step 2: Loading NV-Embed-v2...", flush=True)
model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True, torch_dtype=torch.float16)
model = model.cuda().eval()
# NV-Embed-v2 has its own encode method
max_length = 512
print("  Model loaded", flush=True)

print("Step 3: Encoding sentences...", flush=True)
t0 = time.time()
bs = 32  # small batches for 7B model
all_emb = []

for i in range(0, len(sentences), bs):
    batch = sentences[i:i+bs]
    with torch.no_grad():
        # NV-Embed-v2 expects passage format
        emb = model.encode(batch, instruction="", max_length=max_length)
        emb = F.normalize(emb, dim=-1)
    all_emb.append(emb.cpu().numpy())
    if (i // bs) % 50 == 0:
        print(f"  {i+len(batch)}/{len(sentences)} ({time.time()-t0:.1f}s)", flush=True)

embeddings = np.vstack(all_emb)
print(f"  Done: {embeddings.shape} in {time.time()-t0:.1f}s", flush=True)

# Also encode all 1000 ARAG queries
print("Step 4: Encoding queries...", flush=True)
qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))
q_texts = [q["question"] for q in qs]

query_instruction = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
q_embs = []
for i in range(0, len(q_texts), bs):
    batch = q_texts[i:i+bs]
    with torch.no_grad():
        emb = model.encode(batch, instruction=query_instruction, max_length=max_length)
        emb = F.normalize(emb, dim=-1)
    q_embs.append(emb.cpu().numpy())
q_embeddings = np.vstack(q_embs)
print(f"  {len(q_texts)} queries encoded: {q_embeddings.shape}", flush=True)

print("Step 5: Saving...", flush=True)
out_dir = "/projects/prjs1800/external/arag/data/musique/index_nvembed_v2_musique_full"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "sentence_index.pkl"), "wb") as f:
    pickle.dump({"sentences": sentences, "embeddings": embeddings,
                 "sentence_to_chunk": s2c, "chunks": chunks}, f)
with open(os.path.join(out_dir, "query_embeddings.pkl"), "wb") as f:
    pickle.dump({"questions": q_texts, "embeddings": q_embeddings}, f)
print(f"  Saved to {out_dir}", flush=True)
print("Done!", flush=True)
