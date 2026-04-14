"""
Generate matched-domain SFT data using Qwen-2.5-7B + E5 retrieval.
Same retriever, same passages, same questions as eval.
"""
import json, os, sys, time, pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Retriever:
    def __init__(self, dataset, index_name="index_e5_musique_full", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        idx_path = f"/projects/prjs1800/external/arag/data/{dataset}/{index_name}/sentence_index.pkl"
        with open(idx_path, "rb") as f: idx = pickle.load(f)
        self.sentences, self.embeddings = idx["sentences"], idx["embeddings"]
        self.sentence_to_chunk, self.chunks = idx["sentence_to_chunk"], idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k * 3]
        cb = {}
        for i in top_idx:
            cid = self.sentence_to_chunk[i]
            if cid not in cb or sims[i] > cb[cid]: cb[cid] = float(sims[i])
        ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]

def main():
    retriever = Retriever("musique")
    print("Retriever loaded", flush=True)

    print("Loading Qwen-2.5-7B-Instruct...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True
    )
    print("Qwen loaded", flush=True)

    qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))
    # Use questions 50-500 for training (keep 0-50 for eval)
    train_qs = qs[50:500]
    print(f"Generating answers for {len(train_qs)} questions...", flush=True)

    results = []
    for i, q in enumerate(train_qs):
        passages = retriever.retrieve(q["question"], top_k=5)
        context = "\n\n".join(passages)

        prompt = f"""Context:
{context}

Question: {q["question"]}

Answer the question using the context. Give ONLY the answer in 1-10 words. No explanation."""

        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)
        answer = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Take first line only
        answer = answer.split("\n")[0].strip()

        results.append({
            "question": f"{context}\n\nQuestion: {q['question']}",
            "thinking_trajectories": [""],
            "attempt": answer,
            "gold_answer": q["answer"],
        })

        if i < 5 or i % 50 == 0:
            print(f"[{i+1}/{len(train_qs)}] Q: {q['question'][:60]}", flush=True)
            print(f"  Qwen: {answer[:80]}", flush=True)
            print(f"  Gold: {q['answer']}", flush=True)

    out_path = "/projects/prjs1800/msc-thesis/07-daes/data/qwen_matched_sft.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} examples to {out_path}", flush=True)

if __name__ == "__main__":
    main()
