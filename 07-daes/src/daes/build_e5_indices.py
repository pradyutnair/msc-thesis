"""Build E5-base-v2 indices for HotpotQA and 2WikiMH from native paragraph pools."""
import json, pickle, numpy as np, time, os
from sentence_transformers import SentenceTransformer


def build_index(dataset_name, paragraphs, output_dir):
    print(f"\n{'='*60}", flush=True)
    print(f"Building E5 index for {dataset_name}: {len(paragraphs)} paragraphs", flush=True)

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

    print("  Encoding...", flush=True)
    t0 = time.time()
    bs = 512
    all_emb = []
    for i in range(0, len(sentences), bs):
        emb = model.encode(sentences[i:i+bs], normalize_embeddings=True, show_progress_bar=False)
        all_emb.append(emb)
        if (i // bs) % 20 == 0:
            print(f"    {i+len(sentences[i:i+bs])}/{len(sentences)} ({time.time()-t0:.1f}s)", flush=True)
    embeddings = np.vstack(all_emb)
    print(f"  Done: {embeddings.shape} in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sentence_index.pkl"), "wb") as f:
        pickle.dump({"sentences": sentences, "embeddings": embeddings,
                     "sentence_to_chunk": s2c, "chunks": chunks}, f)
    print(f"  Saved to {output_dir}", flush=True)


# Load E5 once
print("Loading E5-base-v2...", flush=True)
model = SentenceTransformer("intfloat/e5-base-v2", device="cuda")

# HotpotQA
data = json.load(open("/projects/prjs1800/datasets/hotpotqa/hotpot_dev_distractor_v1.json"))
seen = set()
paras = []
for d in data:
    for title, sents in d["context"]:
        if title not in seen:
            seen.add(title)
            paras.append({"title": title, "text": " ".join(sents)})
build_index("HotpotQA", paras, "/projects/prjs1800/external/arag/data/hotpotqa/index_e5_full")

# 2WikiMH
data2 = json.load(open("/projects/prjs1800/datasets/2wikimultihopqa/2wikimultihopqa_validation.json"))
seen2 = set()
paras2 = []
for d in data2:
    ctx = d["context"]
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        if title not in seen2:
            seen2.add(title)
            paras2.append({"title": title, "text": " ".join(sents)})
build_index("2WikiMH", paras2, "/projects/prjs1800/external/arag/data/2wikimultihop/index_e5_full")

print("\nAll done!", flush=True)
