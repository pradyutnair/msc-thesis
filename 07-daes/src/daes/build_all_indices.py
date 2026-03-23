"""Build NV-Embed-v2 indices for HotpotQA and 2WikiMH from their native paragraph pools.
Run with nvembed-venv (transformers 4.42.4)."""
import json, pickle, numpy as np, time, os, torch, sys
import torch.nn.functional as F
from transformers import AutoModel

def build_index(dataset_name, data_path, data_loader_fn, arag_questions_path, output_dir):
    print(f"\n{'='*60}", flush=True)
    print(f"Building index for {dataset_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load paragraphs
    print("Loading paragraphs...", flush=True)
    paragraphs = data_loader_fn(data_path)
    print(f"  {len(paragraphs)} unique paragraphs", flush=True)

    # Split into sentences
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

    # Encode with NV-Embed-v2
    print("Encoding sentences...", flush=True)
    t0 = time.time()
    bs = 64
    all_emb = []
    for i in range(0, len(sentences), bs):
        emb = model.encode(sentences[i:i+bs], instruction="", max_length=512)
        emb = F.normalize(emb, dim=-1)
        all_emb.append(emb.cpu().numpy())
        if (i // bs) % 100 == 0:
            print(f"  {i+bs}/{len(sentences)} ({time.time()-t0:.1f}s)", flush=True)
    embeddings = np.vstack(all_emb)
    print(f"  Done: {embeddings.shape} in {time.time()-t0:.1f}s", flush=True)

    # Encode ARAG questions
    print("Encoding queries...", flush=True)
    qs = json.load(open(arag_questions_path))
    q_texts = [q["question"] for q in qs]
    query_instruction = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
    q_embs = []
    for i in range(0, len(q_texts), bs):
        emb = model.encode(q_texts[i:i+bs], instruction=query_instruction, max_length=512)
        emb = F.normalize(emb, dim=-1)
        q_embs.append(emb.cpu().numpy())
    q_embeddings = np.vstack(q_embs)
    print(f"  {len(q_texts)} queries: {q_embeddings.shape}", flush=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sentence_index.pkl"), "wb") as f:
        pickle.dump({"sentences": sentences, "embeddings": embeddings,
                     "sentence_to_chunk": s2c, "chunks": chunks}, f)
    with open(os.path.join(output_dir, "query_embeddings.pkl"), "wb") as f:
        pickle.dump({"questions": q_texts, "embeddings": q_embeddings}, f)
    print(f"  Saved to {output_dir}", flush=True)


def load_hotpotqa(path):
    data = json.load(open(path))
    seen = set()
    paras = []
    for d in data:
        for title, sents in d["context"]:
            key = title
            if key not in seen:
                seen.add(key)
                paras.append({"title": title, "text": " ".join(sents)})
    return paras


def load_2wikimh(path):
    data = json.load(open(path))
    seen = set()
    paras = []
    for d in data:
        for title, sents in d["context"]:
            key = title
            if key not in seen:
                seen.add(key)
                paras.append({"title": title, "text": " ".join(sents)})
    return paras


# Load model once
print("Loading NV-Embed-v2...", flush=True)
model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True, torch_dtype=torch.float16)
model = model.cuda().eval()
print("Model loaded.", flush=True)

# Build HotpotQA index
build_index(
    "HotpotQA",
    "/projects/prjs1800/datasets/hotpotqa/hotpot_dev_distractor_v1.json",
    load_hotpotqa,
    "/projects/prjs1800/external/arag/data/hotpotqa/questions.json",
    "/projects/prjs1800/external/arag/data/hotpotqa/index_nvembed_v2_full",
)

# Build 2WikiMH index
build_index(
    "2WikiMH",
    "/projects/prjs1800/datasets/2wikimultihopqa/2wikimultihopqa_validation.json",
    load_2wikimh,
    "/projects/prjs1800/external/arag/data/2wikimultihop/questions.json",
    "/projects/prjs1800/external/arag/data/2wikimultihop/index_nvembed_v2_full",
)

print("\nAll indices built!", flush=True)
