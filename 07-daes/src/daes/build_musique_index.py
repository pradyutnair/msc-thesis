"""Build E5-base-v2 sentence index from MuSiQue's native paragraph pool."""
import json
import pickle
import numpy as np
import sys
import time

def main():
    print("Step 1: Loading MuSiQue paragraphs...", flush=True)
    t0 = time.time()

    # Load all paragraphs from MuSiQue dev set
    data = [json.loads(l) for l in open("/projects/prjs1800/datasets/musique/musique_full_v1.0_dev.jsonl")]

    # Deduplicate paragraphs by (title, paragraph_text)
    seen = set()
    paragraphs = []
    for d in data:
        for p in d["paragraphs"]:
            key = (p.get("title", ""), p.get("paragraph_text", ""))
            if key not in seen:
                seen.add(key)
                paragraphs.append({
                    "title": p.get("title", ""),
                    "text": p.get("paragraph_text", ""),
                    "is_supporting": p.get("is_supporting", False),
                })

    print(f"  {len(data)} questions, {len(paragraphs)} unique paragraphs ({time.time()-t0:.1f}s)", flush=True)

    # Split paragraphs into sentences for sentence-level indexing
    print("Step 2: Splitting into sentences...", flush=True)
    sentences = []
    sentence_to_chunk = []
    chunks = {}

    for i, p in enumerate(paragraphs):
        chunk_id = str(i)
        chunks[chunk_id] = {"id": chunk_id, "text": f"[{p['title']}] {p['text']}"}

        # Split on sentence boundaries (period + space)
        text = p["text"].strip()
        if not text:
            continue
        # Simple sentence split
        sents = [s.strip() for s in text.replace(". ", ".\n").split("\n") if s.strip()]
        for s in sents:
            sentences.append(s)
            sentence_to_chunk.append(chunk_id)

    print(f"  {len(sentences)} sentences from {len(chunks)} chunks", flush=True)

    # Encode with E5-base-v2
    print("Step 3: Encoding with E5-base-v2...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("intfloat/e5-base-v2", device="cuda")

    t0 = time.time()
    batch_size = 512
    all_embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(emb)
        if (i // batch_size) % 10 == 0:
            print(f"  Encoded {i+len(batch)}/{len(sentences)} ({time.time()-t0:.1f}s)", flush=True)

    embeddings = np.vstack(all_embeddings)
    print(f"  Done encoding: {embeddings.shape} ({time.time()-t0:.1f}s)", flush=True)

    # Save index
    print("Step 4: Saving index...", flush=True)
    import os
    out_dir = "/projects/prjs1800/external/arag/data/musique/index_e5_musique_full"
    os.makedirs(out_dir, exist_ok=True)

    index_data = {
        "sentences": sentences,
        "embeddings": embeddings,
        "sentence_to_chunk": sentence_to_chunk,
        "chunks": chunks,
    }

    with open(os.path.join(out_dir, "sentence_index.pkl"), "wb") as f:
        pickle.dump(index_data, f)

    print(f"  Saved to {out_dir}/sentence_index.pkl", flush=True)
    print(f"  {len(sentences)} sentences, {len(chunks)} chunks, embeddings {embeddings.shape}", flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
