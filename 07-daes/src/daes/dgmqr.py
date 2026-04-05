"""
DG-MQR: Diffusion-Guided Multi-Query Retrieval for Multi-Hop QA.

Modes:
  baseline : E5 top-5 → Dream-7B single generation
  pool     : E5 top-5 → dLLM candidates → per-candidate retrieval → pool ALL passages → single generation
  vote     : E5 top-5 → dLLM candidates → per-candidate retrieval → generate per-candidate → majority vote
  few_shot : Same as pool but with few-shot exemplars in prompt
"""

import argparse, json, os, sys, time, re, string, pickle, random
import numpy as np, torch, torch.nn.functional as F
from collections import Counter

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from dataclasses import dataclass

random.seed(42)

# ---------------------------------------------------------------------------
# Few-shot exemplars
# ---------------------------------------------------------------------------
FEW_SHOT_PREFIX = """Here are examples showing the expected answer format:

Example 1:
Context: The Eiffel Tower is located in Paris, France. It was completed in 1889.
Question: Where is the Eiffel Tower located?
Answer: Paris, France

Example 2:
Context: Albert Einstein was born on March 14, 1879 in Ulm, Germany.
Question: When was Albert Einstein born?
Answer: March 14, 1879

Example 3:
Context: The Great Wall of China stretches over 13,000 miles.
Question: What is the capital of Japan?
Answer: Tokyo

Now answer the following question concisely (1-10 words).

"""

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, dataset, max_chunk_chars=2000):
        from sentence_transformers import SentenceTransformer
        index_map = {
            "musique": "index_e5_musique_full",
            "hotpotqa": "index_e5_full",
            "2wikimultihop": "index_e5_full",
        }
        idx_path = f"/projects/prjs1800/external/arag/data/{dataset}/{index_map[dataset]}/sentence_index.pkl"
        print(f"Loading index from {idx_path}...", flush=True)
        with open(idx_path, "rb") as f:
            idx = pickle.load(f)
        self.sentences = idx["sentences"]
        self.embeddings = idx["embeddings"]
        self.sentence_to_chunk = idx["sentence_to_chunk"]
        self.chunks = idx["chunks"]
        self.max_chunk_chars = max_chunk_chars
        self.model = SentenceTransformer("intfloat/e5-base-v2", device="cpu")
        print(f"  Index: {len(self.sentences)} sents, {len(self.chunks)} chunks", flush=True)

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top = np.argsort(sims)[::-1][:top_k * 3]
        cb = {}
        for i in top:
            cid = self.sentence_to_chunk[i]
            if cid not in cb or sims[i] > cb[cid]:
                cb[cid] = float(sims[i])
        ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[cid]["text"][:self.max_chunk_chars] for cid, _ in ranked]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_f1(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not pt or not gt:
        return 0, 0, 0
    common = set(pt) & set(gt)
    if not common:
        return 0, 0, 0
    p = len(common) / len(pt)
    r = len(common) / len(gt)
    return p, r, 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Dream-7B generation
# ---------------------------------------------------------------------------
def dllm_generate(model, tokenizer, context, question, steps=128, n_tokens=512,
                  temperature=0.1, few_shot=False):
    """Full denoising with confidence tracking. Returns (answer, avg_conf, per_token_conf)."""
    device = model.device
    mask_id = tokenizer.mask_token_id

    if few_shot:
        prompt = FEW_SHOT_PREFIX + f"{context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens
    token_confidences = torch.zeros(n_tokens, device=device)

    for step in range(steps):
        if remaining <= 0:
            break
        mi = (x == mask_id)
        if not mi.any():
            break
        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mp = mi[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mp], temperature=temperature, neg_entropy=True)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(conf, min(n_commit, len(conf)))
        selected = mp[topk]
        x[0, selected] = x0[topk]

        for idx, pos in enumerate(selected):
            local_pos = pos.item() - n_prefix
            if 0 <= local_pos < n_tokens:
                token_confidences[local_pos] = conf[topk[idx]]
        remaining -= len(topk)

    gen_ids = x[0, n_prefix:].tolist()
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    avg_conf = token_confidences[token_confidences > 0].mean().item() if (token_confidences > 0).any() else 0
    return answer, avg_conf, token_confidences


def extract_candidates(model, tokenizer, context, question, n_candidates=3):
    """Extract bridge entity candidates from dLLM token distribution (single forward pass)."""
    device = model.device
    mask_id = tokenizer.mask_token_id

    prompt = f"{context}\n\nQuestion: {question}\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    n_mask = 20
    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(x, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    probs = torch.softmax(logits[0, n_prefix] / 0.3, dim=-1)
    top_probs, top_ids = torch.topk(probs, n_candidates * 3)

    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        x_c = torch.tensor([canvas], dtype=torch.long, device=device)
        x_c[0, n_prefix] = tid
        rem = n_mask - 1
        for step in range(16):
            if rem <= 0:
                break
            mi = (x_c == mask_id)
            if not mi.any():
                break
            with torch.no_grad():
                o2 = model(x_c, attention_mask=attn)
            l2 = torch.cat([o2.logits[:, :1], o2.logits[:, :-1]], dim=1)
            mp = mi[0].nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=True)
            k = min(max(1, rem // 16), rem)
            if step == 15:
                k = rem
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            rem -= len(tk)

        cand_text = tokenizer.decode(x_c[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
        cand_text = cand_text.split("\n")[0].split(". ")[0].strip()
        if cand_text and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append({"text": cand_text, "init_conf": prob.item()})
            if len(candidates) >= n_candidates:
                break
    return candidates


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def run_pool(model, tokenizer, retriever, question, n_candidates=3, few_shot=False):
    """Pool all candidate-retrieved passages into one context, single generation."""
    # Stage 1: initial retrieval
    initial_passages = retriever.retrieve(question, top_k=5)
    initial_context = "\n\n".join(initial_passages)

    # Stage 2: candidate extraction
    candidates = extract_candidates(model, tokenizer, initial_context, question, n_candidates)
    if not candidates:
        answer, conf, _ = dllm_generate(model, tokenizer, initial_context, question, few_shot=few_shot)
        return answer, {"method": "pool_fallback", "n_candidates": 0}

    # Stage 3: per-candidate retrieval → pool all
    all_passages = list(initial_passages)  # start with initial top-5
    seen_texts = set(p[:100] for p in all_passages)
    for cand in candidates:
        hop2_query = f"{question} {cand['text']}"
        hop2_passages = retriever.retrieve(hop2_query, top_k=3)
        for p in hop2_passages:
            if p[:100] not in seen_texts:
                all_passages.append(p)
                seen_texts.add(p[:100])

    # Stage 4: single generation with pooled context
    pooled_context = "\n\n".join(all_passages)
    answer, conf, _ = dllm_generate(model, tokenizer, pooled_context, question, few_shot=few_shot)

    stats = {
        "method": "pool",
        "n_candidates": len(candidates),
        "n_passages": len(all_passages),
        "candidates": [c["text"][:40] for c in candidates],
    }
    return answer, stats


def run_vote(model, tokenizer, retriever, question, n_candidates=3, few_shot=False):
    """Generate one answer per candidate's evidence, majority vote."""
    # Stage 1: initial retrieval
    initial_passages = retriever.retrieve(question, top_k=5)
    initial_context = "\n\n".join(initial_passages)

    # Stage 2: candidate extraction
    candidates = extract_candidates(model, tokenizer, initial_context, question, n_candidates)
    if not candidates:
        answer, conf, _ = dllm_generate(model, tokenizer, initial_context, question, few_shot=few_shot)
        return answer, {"method": "vote_fallback", "n_candidates": 0}

    # Stage 3+4: per-candidate retrieval + generation
    answers = []
    for cand in candidates:
        hop2_query = f"{question} {cand['text']}"
        hop2_passages = retriever.retrieve(hop2_query, top_k=3)
        expanded_context = initial_context + "\n\n" + "\n\n".join(hop2_passages)
        answer, conf, _ = dllm_generate(model, tokenizer, expanded_context, question, few_shot=few_shot)
        cand["answer"] = answer
        cand["conf"] = conf
        answers.append(answer)

    # Stage 5: majority vote via normalized string matching
    normalized = [normalize_answer(a) for a in answers]
    counter = Counter(normalized)
    most_common_norm, count = counter.most_common(1)[0]

    if count > 1:
        # Majority exists — pick the original answer that matches
        for a, n in zip(answers, normalized):
            if n == most_common_norm:
                selected_answer = a
                break
        selection = "majority"
    else:
        # No majority — pick shortest answer (Dream produces concise = more likely correct)
        selected_answer = min(answers, key=lambda a: len(normalize_answer(a).split()))
        selection = "shortest"

    stats = {
        "method": "vote",
        "n_candidates": len(candidates),
        "answers": [a[:60] for a in answers],
        "normalized": normalized,
        "selection": selection,
        "candidates": [c["text"][:40] for c in candidates],
    }
    return selected_answer, stats


def run_baseline(model, tokenizer, retriever, question, few_shot=False):
    """Single-shot retrieval + generation."""
    passages = retriever.retrieve(question, top_k=5)
    context = "\n\n".join(passages)
    answer, conf, _ = dllm_generate(model, tokenizer, context, question, few_shot=few_shot)
    return answer, {"method": "baseline"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DG-MQR: Diffusion-Guided Multi-Query Retrieval")
    parser.add_argument("--dataset", default="musique", choices=["musique", "hotpotqa", "2wikimultihop"])
    parser.add_argument("--mode", required=True, choices=["baseline", "pool", "vote"])
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None, help="If set, overrides n_questions")
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--few_shot", action="store_true", help="Add few-shot exemplars to prompt")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine question range
    if args.end_idx is not None:
        start, end = args.start_idx, args.end_idx
    else:
        start, end = args.start_idx, args.start_idx + args.n_questions

    # Load retriever
    retriever = Retriever(args.dataset)

    # Load Dream-7B
    print("Loading Dream-7B...", flush=True)
    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())
    print("Model loaded.", flush=True)

    # Load questions
    all_qs = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))
    qs = all_qs[start:end]
    print(f"Running {args.mode} on {args.dataset} [{start}:{end}] ({len(qs)} questions)", flush=True)

    # Output path
    fs_tag = "_fewshot" if args.few_shot else ""
    out_path = os.path.join(args.output_dir, f"dgmqr_{args.mode}{fs_tag}_{args.dataset}_{start}_{end}.jsonl")

    predictions = []
    sum_f1, sum_p, sum_r = 0, 0, 0

    for i, q in enumerate(qs):
        t0 = time.time()

        if args.mode == "baseline":
            answer, stats = run_baseline(model, tokenizer, retriever, q["question"],
                                         few_shot=args.few_shot)
        elif args.mode == "pool":
            answer, stats = run_pool(model, tokenizer, retriever, q["question"],
                                     n_candidates=args.n_candidates, few_shot=args.few_shot)
        elif args.mode == "vote":
            answer, stats = run_vote(model, tokenizer, retriever, q["question"],
                                     n_candidates=args.n_candidates, few_shot=args.few_shot)

        elapsed = time.time() - t0
        p, r, f = compute_f1(answer, q["answer"])
        sum_f1 += f
        sum_p += p
        sum_r += r
        contain = q["answer"].lower() in answer.lower()

        pred = {
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "pred_answer": answer,
            "mode": args.mode,
            "few_shot": args.few_shot,
            "time": round(elapsed, 2),
            "f1": round(f, 4),
            "contain": contain,
            "stats": stats,
        }
        predictions.append(pred)

        # Incremental write
        with open(out_path, "a") as fw:
            fw.write(json.dumps(pred) + "\n")

        cand_info = ""
        if "candidates" in stats:
            cand_info = f" | cands: {stats['candidates']}"
        sel_info = ""
        if "selection" in stats:
            sel_info = f" | sel={stats['selection']}"
        print(f"[{i+1}/{len(qs)}] ({elapsed:.1f}s) F1={f:.2f} contain={contain}{sel_info}{cand_info}", flush=True)
        print(f"  Gold: {q['answer']}", flush=True)
        print(f"  Pred: {answer[:120]}", flush=True)

    # Summary
    n = len(predictions)
    cn = sum(1 for p in predictions if p["contain"])
    print(f"\n{'='*60}", flush=True)
    print(f"DG-MQR | {args.mode}{fs_tag} | {args.dataset} | N={n} [{start}:{end}]", flush=True)
    print(f"  F1:        {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Precision: {sum_p/n*100:.1f}%", flush=True)
    print(f"  Recall:    {sum_r/n*100:.1f}%", flush=True)
    print(f"  Contain:   {cn}/{n} = {cn/n*100:.1f}%", flush=True)
    print(f"  Output:    {out_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
