"""Build MuSiQue index with GTE-Qwen2-1.5B-instruct (stronger retriever)."""
import json, pickle, numpy as np, time, os

# Monkey-patch DynamicCache for transformers 4.57+ compatibility
from transformers import DynamicCache
if not hasattr(DynamicCache, "get_usable_length"):
    def _get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)
    DynamicCache.get_usable_length = _get_usable_length

from sentence_transformers import SentenceTransformer

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

sentences = []
sentence_to_chunk = []
chunks = {}
for i, p in enumerate(paragraphs):
    cid = str(i)
    chunks[cid] = {"id": cid, "text": "[" + p["title"] + "] " + p["text"]}
    text = p["text"].strip()
    if not text:
        continue
    sents = [s.strip() for s in text.replace(". ", ".\n").split("\n") if s.strip()]
    for s in sents:
        sentences.append(s)
        sentence_to_chunk.append(cid)
print(f"  {len(sentences)} sentences from {len(chunks)} chunks", flush=True)

print("Step 2: Loading GTE-Qwen2-1.5B-instruct...", flush=True)
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device="cuda")
print(f"  Loaded. Dim: {model.get_sentence_embedding_dimension()}", flush=True)

print("Step 3: Encoding sentences...", flush=True)
t0 = time.time()
batch_size = 256
all_emb = []
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i+batch_size]
    emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    all_emb.append(emb)
    if (i // batch_size) % 20 == 0:
        print(f"  {i+len(batch)}/{len(sentences)} ({time.time()-t0:.1f}s)", flush=True)
embeddings = np.vstack(all_emb)
print(f"  Done: {embeddings.shape} in {time.time()-t0:.1f}s", flush=True)

print("Step 4: Saving index...", flush=True)
out_dir = "/projects/prjs1800/external/arag/data/musique/index_gte_qwen2_musique_full"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "sentence_index.pkl"), "wb") as f:
    pickle.dump({"sentences": sentences, "embeddings": embeddings,
                 "sentence_to_chunk": sentence_to_chunk, "chunks": chunks}, f)
print(f"  Saved to {out_dir}", flush=True)
print("Done!", flush=True)
