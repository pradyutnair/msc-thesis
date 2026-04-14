"""
Evaluate SFT-trained Dream-7B on multi-hop QA.

Loads the LoRA-adapted model, retrieves passages, generates with the
reasoning template, and evaluates F1.
"""

import argparse
import json
import os
import sys
import time
import re
import string
import pickle
import numpy as np
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import sample_tokens


# ---------------------------------------------------------------------------
# Retriever (same as other scripts)
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self, dataset, index_name="index_e5_base_v2", max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        data_dir = f"/projects/prjs1800/external/arag/data/{dataset}"
        idx_path = os.path.join(data_dir, index_name, "sentence_index.pkl")
        print(f"Loading index from {idx_path}...", flush=True)
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
        print(f"Index: {len(self.sentences)} sents, {len(self.chunks)} chunks", flush=True)

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:top_k * 3]
        chunk_best = {}
        for i in top_idx:
            cid = self.sentence_to_chunk[i]
            if cid not in chunk_best or sims[i] > chunk_best[cid]:
                chunk_best[cid] = float(sims[i])
        ranked = sorted(chunk_best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_f1(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt: return 0.0, 0.0, 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0, 0.0, 0.0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return p, r, 2 * p * r / (p + r)


def extract_answer_from_response(text):
    """Extract answer - try tags first, then return full text."""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No tags - return full text (for answer-only models)
    # Clean up any remaining tags
    text = re.sub(r'<[^>]+>', '', text).strip()
    return text


def generate_answer(model, tokenizer, context, question, steps=128, n_tokens=20, temperature=0.1):
    """Generate with reasoning template using Dream's denoising."""
    device = model.device
    mask_id = tokenizer.mask_token_id

    SYSTEM_PROMPT = """You are a multi-hop question answering assistant. Given context passages and a question, reason step by step to find the answer.

Respond in the following format:
<reasoning>
Your step-by-step reasoning here
</reasoning>
<answer>
Your concise answer (1-10 words)
</answer>"""

    prompt_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": prompt_content}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.bool, device=device)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens

    for step in range(steps):
        if remaining <= 0:
            break
        mask_idx = (x[0] == mask_id)
        canvas_mask = mask_idx.clone()
        canvas_mask[:n_prefix] = False
        if not canvas_mask.any():
            break
        mask_pos = canvas_mask.nonzero(as_tuple=True)[0]
        n_masked = len(mask_pos)

        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mask_logits = logits[0, mask_pos]
        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, n_masked))
        x[0, mask_pos[topk]] = x0[topk]
        remaining -= len(topk)

    gen_ids = x[0, n_prefix:].tolist()
    response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    answer = extract_answer_from_response(response)
    return answer, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--model_path", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--lora_path", default=None, help="Path to LoRA adapter")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    index_map = {
        "musique": "index_e5_musique_full",
        "hotpotqa": "index_e5_full",
        "2wikimultihop": "index_e5_full",
    }
    retriever = Retriever(args.dataset, index_name=index_map[args.dataset])

    print(f"Loading {args.model_path}...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    if args.lora_path:
        print(f"Loading LoRA from {args.lora_path}...", flush=True)
        from peft import PeftModel
        # Dream doesn't have prepare_inputs_for_generation - add for PEFT
        if not hasattr(model, 'prepare_inputs_for_generation'):
            model.prepare_inputs_for_generation = lambda *args, **kwargs: {}
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()  # merge for faster inference

    model = model.cuda().eval()
    print("Model loaded.", flush=True)

    all_qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))
    qs = all_qs[:args.n_questions]

    tag = "sft" if args.lora_path else "base"
    out_path = os.path.join(args.output_dir, f"eval_{tag}_{args.dataset}_{len(qs)}q.jsonl")
    open(out_path, "w").close()

    sum_f1 = 0
    predictions = []

    for i, q in enumerate(qs):
        passages = retriever.retrieve(q["question"], top_k=5)
        context = "\n\n".join(passages)

        t0 = time.time()
        answer, full_response = generate_answer(
            model, tokenizer, context, q["question"],
            steps=args.steps, n_tokens=args.n_tokens,
        )
        elapsed = time.time() - t0

        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f
        contain = q["answer"].lower() in answer.lower()

        pred = {
            "id": q["id"], "question": q["question"],
            "gold_answer": q["answer"], "pred_answer": answer,
            "full_response": full_response[:500],
            "f1": round(f, 4), "contain": contain, "time": round(elapsed, 2),
        }
        predictions.append(pred)
        with open(out_path, "a") as fw:
            fw.write(json.dumps(pred) + "\n")

        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:100]}", flush=True)
        if full_response != answer:
            print(f"  Full: {full_response[:150]}", flush=True)

    n = len(predictions)
    cn = sum(1 for p in predictions if p["contain"])
    print(f"\n{'='*60}", flush=True)
    print(f"EVAL | {tag} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:      {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Contain: {cn}/{n} = {cn/n*100:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
