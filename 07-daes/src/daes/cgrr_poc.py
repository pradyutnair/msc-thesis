"""
CGRR Proof-of-Concept: Confidence-Guided Retrieve-and-Remask for LLaDA.

Tests whether mid-denoising retrieval + selective remasking improves answers.

For each question, runs 4 conditions:
  1. baseline:      standard denoising (no retrieval)
  2. retrieve_only: retrieve based on question, denoise with evidence from scratch
  3. cgrr:          denoise halfway -> extract query from partial text -> retrieve ->
                    remask low-confidence tokens -> re-denoise with evidence
  4. oracle:        denoising with gold context prepended (upper bound)

Go/no-go: if CGRR > baseline on even a few questions, the mechanism works.
"""

import argparse, json, os, sys, time, re, string, copy
import numpy as np, torch
import faiss, sqlite3
from transformers import AutoTokenizer, AutoModel

# ---- Retriever (reuse from react_dllm) ----
class DenseRetriever:
    def __init__(self, index_path, db_path, encoder_name="intfloat/e5-base-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading FAISS index from {index_path}...", flush=True)
        self.index = faiss.read_index(index_path)
        print(f"Index loaded: {self.index.ntotal} vectors", flush=True)
        self.db = sqlite3.connect(db_path)
        self.encoder = SentenceTransformer(encoder_name, device="cpu")
        print("Retriever ready.", flush=True)

    def search(self, queries, top_k=5):
        if isinstance(queries, str):
            queries = [queries]
        passages = []
        for query in queries:
            emb = self.encoder.encode([f"query: {query}"], normalize_embeddings=True)
            scores, ids = self.index.search(emb.astype(np.float32), top_k)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0: continue
                cursor = self.db.cursor()
                cursor.execute("SELECT title, contents FROM passages WHERE id = ?", (int(idx) + 1,))
                row = cursor.fetchone()
                if row:
                    passages.append(f"[{row[0]}] {row[1][:400]}")
        return passages


# ---- Metrics ----
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt: return 0.0
    common = set(pt) & set(gt)
    if not common: return 0.0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return 2*p*r/(p+r)


# ---- Core denoising with CGRR support ----
def denoise(model, tokenizer, prompt_text, max_new_tokens=256, steps=128,
            temperature=0.1, mask_id=126336,
            remask_step=None, retriever=None, remask_threshold=0.5,
            question_text=None):
    device = model.device
    prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    
    canvas = prefix_ids + [mask_id] * max_new_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.bool, device=device)
    
    k = max(1, max_new_tokens // steps)
    remaining = max_new_tokens
    gen_start = n_prefix
    
    diagnostics = {
        "partial_text_at_remask": None,
        "retrieved_passages": None,
        "n_remasked": 0,
        "confidence_stats": {},
    }
    
    for step in range(steps):
        if remaining <= 0:
            break
        
        mi = (x[0] == mask_id)
        mi[:gen_start] = False
        if not mi.any():
            break
        mp = mi.nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = out.logits
        
        mask_logits = logits[0, mp]
        if temperature > 0:
            mask_logits = mask_logits / temperature
        probs = torch.softmax(mask_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        x0 = probs.argmax(dim=-1)
        
        # ---- CGRR: Retrieve-and-Remask at specified step ----
        if remask_step is not None and step == remask_step and retriever is not None:
            gen_ids = x[0, n_prefix:].tolist()
            partial_text = tokenizer.decode(
                [t for t in gen_ids if t != mask_id],
                skip_special_tokens=True
            ).strip()
            diagnostics["partial_text_at_remask"] = partial_text
            
            if partial_text and len(partial_text) > 5:
                query = f"{question_text} {partial_text[:200]}"
            else:
                query = question_text
            
            passages = retriever.search(query, top_k=5)
            diagnostics["retrieved_passages"] = passages
            evidence_text = "\n".join(passages[:3])
            
            new_prompt = (
                f"Evidence:\n{evidence_text}\n\n"
                f"Based on the evidence above, answer the question concisely.\n\n"
                f"Question: {question_text}\n\nAnswer:"
            )
            new_prefix_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
            new_n_prefix = len(new_prefix_ids)
            
            committed_mask = (x[0, n_prefix:] != mask_id)
            committed_positions = committed_mask.nonzero(as_tuple=True)[0]
            committed_tokens = x[0, n_prefix:][committed_mask].clone()
            
            all_gen_logits = logits[0, n_prefix:]
            if temperature > 0:
                all_gen_logits = all_gen_logits / temperature
            all_gen_probs = torch.softmax(all_gen_logits, dim=-1)
            all_gen_conf = all_gen_probs.max(dim=-1).values
            committed_conf = all_gen_conf[committed_mask]
            
            diagnostics["confidence_stats"] = {
                "mean": committed_conf.mean().item() if len(committed_conf) > 0 else 0,
                "std": committed_conf.std().item() if len(committed_conf) > 1 else 0,
                "min": committed_conf.min().item() if len(committed_conf) > 0 else 0,
                "max": committed_conf.max().item() if len(committed_conf) > 0 else 0,
                "n_committed": len(committed_conf),
            }
            
            keep_mask = committed_conf >= remask_threshold
            n_remasked = (~keep_mask).sum().item()
            diagnostics["n_remasked"] = n_remasked
            
            kept_tokens = committed_tokens[keep_mask]
            kept_rel_positions = committed_positions[keep_mask]
            
            new_gen_len = max_new_tokens
            new_canvas = new_prefix_ids + [mask_id] * new_gen_len
            x_new = torch.tensor([new_canvas], dtype=torch.long, device=device)
            
            for rel_pos, tok in zip(kept_rel_positions, kept_tokens):
                new_pos = new_n_prefix + rel_pos.item()
                if new_pos < len(new_canvas):
                    x_new[0, new_pos] = tok
            
            x = x_new
            attn = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
            gen_start = new_n_prefix
            n_prefix = new_n_prefix
            
            remaining = (x[0, gen_start:] == mask_id).sum().item()
            k = max(1, remaining // max(1, steps - step - 1))
            
            print(f"    [CGRR step {step}] partial='{partial_text[:80]}' "
                  f"remasked={n_remasked}/{len(committed_conf)} "
                  f"conf_mean={diagnostics['confidence_stats']['mean']:.3f}", flush=True)
            continue
        
        # ---- Normal unmasking ----
        nc = min(k, remaining)
        if step == steps - 1:
            nc = remaining
        _, topk = torch.topk(confidence, min(nc, len(confidence)))
        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)
    
    gen_ids = x[0, gen_start:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, diagnostics


# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_questions", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--remask_frac", type=float, default=0.5)
    parser.add_argument("--remask_threshold", type=float, default=0.5)
    args = parser.parse_args()
    
    MASK_ID = 126336
    MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
    
    retriever = DenseRetriever(
        index_path="/projects/prjs1800/datasets/flashrag/indexes/e5_IVFPQ.index",
        db_path="/projects/prjs1800/datasets/flashrag/wiki18_100w.db",
    )
    
    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    print("Model loaded.", flush=True)
    
    ds_path = f"/projects/prjs1800/datasets/flashrag/{args.dataset}/test.jsonl"
    questions = [json.loads(l) for l in open(ds_path)][:args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)
    
    remask_step = int(args.steps * args.remask_frac)
    
    results = []
    for i, q in enumerate(questions):
        question = q["question"]
        gold_answers = q["golden_answers"]
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(questions)}] Q: {question}", flush=True)
        print(f"  Gold: {gold_answers}", flush=True)
        
        base_prompt = f"Answer the following question concisely (1-10 words).\n\nQuestion: {question}\n\nAnswer:"
        
        # --- Condition 1: Baseline ---
        print(f"\n  [BASELINE]", flush=True)
        t0 = time.time()
        baseline_text, _ = denoise(model, tokenizer, base_prompt,
            max_new_tokens=args.max_new_tokens, steps=args.steps, mask_id=MASK_ID)
        baseline_time = time.time() - t0
        baseline_f1 = max(compute_f1(baseline_text, ga) for ga in gold_answers)
        print(f"  [BASELINE] '{baseline_text[:100]}' F1={baseline_f1:.3f} ({baseline_time:.1f}s)", flush=True)
        
        # --- Condition 2: Oracle ---
        gold_context = ""
        if "context" in q:
            if isinstance(q["context"], list):
                for ctx in q["context"]:
                    if isinstance(ctx, list) and len(ctx) == 2:
                        gold_context += f"[{ctx[0]}] {' '.join(ctx[1]) if isinstance(ctx[1], list) else ctx[1]}\n"
                    elif isinstance(ctx, dict):
                        gold_context += f"[{ctx.get('title','')}] {ctx.get('paragraph_text','')}\n"
            elif isinstance(q["context"], str):
                gold_context = q["context"]
        
        oracle_f1 = 0.0
        oracle_text = ""
        oracle_time = 0.0
        if gold_context:
            oracle_prompt = f"Evidence:\n{gold_context[:2000]}\n\nBased on the evidence, answer concisely.\n\nQuestion: {question}\n\nAnswer:"
            print(f"\n  [ORACLE]", flush=True)
            t0 = time.time()
            oracle_text, _ = denoise(model, tokenizer, oracle_prompt,
                max_new_tokens=args.max_new_tokens, steps=args.steps, mask_id=MASK_ID)
            oracle_time = time.time() - t0
            oracle_f1 = max(compute_f1(oracle_text, ga) for ga in gold_answers)
            print(f"  [ORACLE] '{oracle_text[:100]}' F1={oracle_f1:.3f} ({oracle_time:.1f}s)", flush=True)
        
        # --- Condition 3: CGRR ---
        print(f"\n  [CGRR] remask at step {remask_step}/{args.steps}", flush=True)
        t0 = time.time()
        cgrr_text, diag = denoise(model, tokenizer, base_prompt,
            max_new_tokens=args.max_new_tokens, steps=args.steps, mask_id=MASK_ID,
            remask_step=remask_step, retriever=retriever,
            remask_threshold=args.remask_threshold, question_text=question)
        cgrr_time = time.time() - t0
        cgrr_f1 = max(compute_f1(cgrr_text, ga) for ga in gold_answers)
        print(f"  [CGRR] '{cgrr_text[:100]}' F1={cgrr_f1:.3f} ({cgrr_time:.1f}s)", flush=True)
        print(f"  [CGRR] Partial: '{diag.get('partial_text_at_remask','')[:80]}'", flush=True)
        print(f"  [CGRR] Remasked: {diag.get('n_remasked',0)}, conf: {diag.get('confidence_stats',{})}", flush=True)
        
        # --- Condition 4: Retrieve-only ---
        print(f"\n  [RETRIEVE-ONLY]", flush=True)
        passages = retriever.search(question, top_k=5)
        evidence_text = "\n".join(passages[:3])
        retonly_prompt = f"Evidence:\n{evidence_text}\n\nBased on the evidence, answer concisely.\n\nQuestion: {question}\n\nAnswer:"
        t0 = time.time()
        retonly_text, _ = denoise(model, tokenizer, retonly_prompt,
            max_new_tokens=args.max_new_tokens, steps=args.steps, mask_id=MASK_ID)
        retonly_time = time.time() - t0
        retonly_f1 = max(compute_f1(retonly_text, ga) for ga in gold_answers)
        print(f"  [RETRIEVE-ONLY] '{retonly_text[:100]}' F1={retonly_f1:.3f} ({retonly_time:.1f}s)", flush=True)
        
        result = {
            "question": question, "gold": gold_answers,
            "baseline": {"text": baseline_text, "f1": baseline_f1, "time": baseline_time},
            "oracle": {"text": oracle_text, "f1": oracle_f1, "time": oracle_time},
            "cgrr": {"text": cgrr_text, "f1": cgrr_f1, "time": cgrr_time,
                     "diagnostics": {k: v for k, v in diag.items() if k != "retrieved_passages"}},
            "retrieve_only": {"text": retonly_text, "f1": retonly_f1, "time": retonly_time},
        }
        results.append(result)
    
    # Summary
    n = len(results)
    print(f"\n{'='*60}", flush=True)
    print(f"CGRR PROOF-OF-CONCEPT | LLaDA | {args.dataset} | N={n}", flush=True)
    print(f"  remask_step={remask_step}/{args.steps} remask_threshold={args.remask_threshold}", flush=True)
    print(f"{'='*60}", flush=True)
    
    for method in ["baseline", "retrieve_only", "cgrr", "oracle"]:
        f1s = [r[method]["f1"] for r in results]
        mean_f1 = np.mean(f1s) * 100
        wins = sum(1 for r in results if r[method]["f1"] > r["baseline"]["f1"])
        print(f"  {method:15s}: F1={mean_f1:5.1f}%  wins_vs_baseline={wins}/{n}", flush=True)
    
    print(f"\nPer-question breakdown:", flush=True)
    print(f"{'Q':>3} {'Baseline':>10} {'Ret-Only':>10} {'CGRR':>10} {'Oracle':>10} {'Winner':>12}", flush=True)
    for i, r in enumerate(results):
        scores = {
            "baseline": r["baseline"]["f1"],
            "ret_only": r["retrieve_only"]["f1"],
            "cgrr": r["cgrr"]["f1"],
            "oracle": r["oracle"]["f1"],
        }
        winner = max(scores, key=scores.get)
        print(f"{i+1:3d} {scores['baseline']:10.3f} {scores['ret_only']:10.3f} "
              f"{scores['cgrr']:10.3f} {scores['oracle']:10.3f} {winner:>12}", flush=True)
    
    out_path = f"/projects/prjs1800/msc-thesis/07-daes/results/cgrr_poc_{args.dataset}_{n}q.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
