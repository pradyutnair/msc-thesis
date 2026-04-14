"""
CGRR v2: Fix the "all committed by halfway" problem.

Changes from v1:
  1. Remask at step ~25% (when ~half tokens committed, more variance)
  2. Percentile-based remasking: remask bottom P% of committed tokens by confidence
     regardless of absolute threshold
  3. Also try multiple remask points (25%, 50%, 75%)
  4. Track confidence distribution over denoising steps
"""

import argparse, json, os, sys, time, re, string
import numpy as np, torch
import faiss, sqlite3
from transformers import AutoTokenizer, AutoModel

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


def denoise_with_cgrr(model, tokenizer, prompt_text, max_new_tokens=256, steps=128,
                       temperature=0.1, mask_id=126336,
                       retriever=None, question_text=None,
                       remask_at_frac=0.25, remask_pct=0.5):
    """
    Denoising with CGRR v2: percentile-based remasking.
    
    remask_at_frac: fraction of steps at which to trigger CGRR
    remask_pct: fraction of committed tokens to remask (bottom P% by confidence)
    """
    device = model.device
    prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    
    canvas = prefix_ids + [mask_id] * max_new_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.bool, device=device)
    
    k = max(1, max_new_tokens // steps)
    remaining = max_new_tokens
    gen_start = n_prefix
    
    remask_step = int(steps * remask_at_frac)
    cgrr_triggered = False
    
    diagnostics = {
        "partial_text": None,
        "n_committed_at_remask": 0,
        "n_remasked": 0,
        "confidence_stats": {},
        "conf_trajectory": [],  # track confidence over steps
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
        
        # Track how many tokens are committed and their confidence
        n_committed = max_new_tokens - len(mp)
        if step % 10 == 0 or step == remask_step:
            diagnostics["conf_trajectory"].append({
                "step": step,
                "n_committed": n_committed,
                "n_masked": len(mp),
                "mask_conf_mean": confidence.mean().item(),
                "mask_conf_min": confidence.min().item() if len(confidence) > 0 else 0,
            })
        
        # ---- CGRR v2: Percentile-based retrieve-and-remask ----
        if step == remask_step and retriever is not None and not cgrr_triggered:
            cgrr_triggered = True
            
            # Decode partial text
            gen_ids = x[0, n_prefix:].tolist()
            partial_text = tokenizer.decode(
                [t for t in gen_ids if t != mask_id],
                skip_special_tokens=True
            ).strip()
            diagnostics["partial_text"] = partial_text
            diagnostics["n_committed_at_remask"] = n_committed
            
            # Retrieve
            if partial_text and len(partial_text) > 5:
                query = f"{question_text} {partial_text[:200]}"
            else:
                query = question_text
            passages = retriever.search(query, top_k=5)
            evidence_text = "\n".join(passages[:3])
            
            # New prefix with evidence
            new_prompt = (
                f"Evidence:\n{evidence_text}\n\n"
                f"Based on the evidence above, answer the question concisely (1-10 words).\n\n"
                f"Question: {question_text}\n\nAnswer:"
            )
            new_prefix_ids = tokenizer.encode(new_prompt, add_special_tokens=False)
            new_n_prefix = len(new_prefix_ids)
            
            # Get confidence of ALL gen positions (including committed)
            all_gen_logits = logits[0, n_prefix:n_prefix + max_new_tokens]
            if temperature > 0:
                all_gen_logits = all_gen_logits / temperature
            all_gen_probs = torch.softmax(all_gen_logits, dim=-1)
            
            # For committed tokens: get confidence of the token that's there
            gen_tokens = x[0, n_prefix:n_prefix + max_new_tokens]
            committed_mask = (gen_tokens != mask_id)
            
            if committed_mask.any():
                committed_positions = committed_mask.nonzero(as_tuple=True)[0]
                committed_tokens = gen_tokens[committed_mask].clone()
                
                # Confidence = probability the model assigns to the committed token
                committed_probs = all_gen_probs[committed_positions]
                committed_conf = committed_probs.gather(
                    1, committed_tokens.unsqueeze(1)
                ).squeeze(1)
                
                diagnostics["confidence_stats"] = {
                    "mean": committed_conf.mean().item(),
                    "std": committed_conf.std().item() if len(committed_conf) > 1 else 0,
                    "min": committed_conf.min().item(),
                    "max": committed_conf.max().item(),
                    "n_committed": len(committed_conf),
                }
                
                # Percentile-based remasking: remask bottom remask_pct% of committed tokens
                n_to_remask = max(1, int(len(committed_conf) * remask_pct))
                _, remask_indices = torch.topk(committed_conf, n_to_remask, largest=False)
                
                keep_mask = torch.ones(len(committed_conf), dtype=torch.bool, device=device)
                keep_mask[remask_indices] = False
                
                diagnostics["n_remasked"] = n_to_remask
                
                kept_tokens = committed_tokens[keep_mask]
                kept_positions = committed_positions[keep_mask]
            else:
                kept_tokens = torch.tensor([], dtype=torch.long, device=device)
                kept_positions = torch.tensor([], dtype=torch.long, device=device)
                diagnostics["n_remasked"] = 0
            
            # Rebuild canvas
            new_canvas = new_prefix_ids + [mask_id] * max_new_tokens
            x_new = torch.tensor([new_canvas], dtype=torch.long, device=device)
            
            for rel_pos, tok in zip(kept_positions, kept_tokens):
                new_pos = new_n_prefix + rel_pos.item()
                if new_pos < len(new_canvas):
                    x_new[0, new_pos] = tok
            
            x = x_new
            attn = torch.ones((1, x.shape[1]), dtype=torch.bool, device=device)
            gen_start = new_n_prefix
            n_prefix = new_n_prefix
            
            remaining = (x[0, gen_start:] == mask_id).sum().item()
            k = max(1, remaining // max(1, steps - step - 1))
            
            print(f"    [CGRRv2 step {step}] committed={n_committed}/{max_new_tokens} "
                  f"remasked={diagnostics['n_remasked']} "
                  f"conf={diagnostics['confidence_stats'].get('mean',0):.4f}±{diagnostics['confidence_stats'].get('std',0):.4f} "
                  f"partial='{partial_text[:60]}'", flush=True)
            continue
        
        # Normal unmasking
        nc = min(k, remaining)
        if step == steps - 1:
            nc = remaining
        _, topk = torch.topk(confidence, min(nc, len(confidence)))
        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)
    
    gen_ids = x[0, gen_start:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, diagnostics


def denoise_baseline(model, tokenizer, prompt_text, max_new_tokens=256, steps=128,
                      temperature=0.1, mask_id=126336):
    """Standard denoising, no retrieval."""
    device = model.device
    prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    
    canvas = prefix_ids + [mask_id] * max_new_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.bool, device=device)
    
    k = max(1, max_new_tokens // steps)
    remaining = max_new_tokens
    
    for step in range(steps):
        if remaining <= 0: break
        mi = (x[0] == mask_id)
        mi[:n_prefix] = False
        if not mi.any(): break
        mp = mi.nonzero(as_tuple=True)[0]
        
        with torch.no_grad():
            logits = model(x, attention_mask=attn).logits
        mask_logits = logits[0, mp]
        if temperature > 0:
            mask_logits = mask_logits / temperature
        probs = torch.softmax(mask_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        x0 = probs.argmax(dim=-1)
        
        nc = min(k, remaining)
        if step == steps - 1: nc = remaining
        _, topk = torch.topk(confidence, min(nc, len(confidence)))
        x[0, mp[topk]] = x0[topk]
        remaining -= len(topk)
    
    gen_ids = x[0, n_prefix:].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_questions", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
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
    
    # Test configurations
    configs = [
        {"name": "baseline", "retriever": None},
        {"name": "retrieve_only", "retriever": "question_only"},
        {"name": "cgrr_25pct_remask30", "remask_at_frac": 0.25, "remask_pct": 0.3},
        {"name": "cgrr_25pct_remask50", "remask_at_frac": 0.25, "remask_pct": 0.5},
        {"name": "cgrr_25pct_remask70", "remask_at_frac": 0.25, "remask_pct": 0.7},
        {"name": "cgrr_10pct_remask50", "remask_at_frac": 0.10, "remask_pct": 0.5},
    ]
    
    all_results = []
    for i, q in enumerate(questions):
        question = q["question"]
        gold_answers = q["golden_answers"]
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(questions)}] Q: {question}", flush=True)
        print(f"  Gold: {gold_answers}", flush=True)
        
        base_prompt = f"Answer the following question concisely (1-10 words).\n\nQuestion: {question}\n\nAnswer:"
        
        result = {"question": question, "gold": gold_answers}
        
        for cfg in configs:
            name = cfg["name"]
            
            if name == "baseline":
                t0 = time.time()
                text = denoise_baseline(model, tokenizer, base_prompt,
                    max_new_tokens=args.max_new_tokens, steps=args.steps, mask_id=MASK_ID)
                elapsed = time.time() - t0
                f1 = max(compute_f1(text, ga) for ga in gold_answers)
                result[name] = {"text": text, "f1": f1, "time": elapsed}
                print(f"  [{name}] '{text[:80]}' F1={f1:.3f}", flush=True)
                
            elif name == "retrieve_only":
                passages = retriever.search(question, top_k=5)
                evidence = "\n".join(passages[:3])
                ret_prompt = f"Evidence:\n{evidence}\n\nBased on the evidence, answer concisely (1-10 words).\n\nQuestion: {question}\n\nAnswer:"
                t0 = time.time()
                text = denoise_baseline(model, tokenizer, ret_prompt,
                    max_new_tokens=args.max_new_tokens, steps=args.steps, mask_id=MASK_ID)
                elapsed = time.time() - t0
                f1 = max(compute_f1(text, ga) for ga in gold_answers)
                result[name] = {"text": text, "f1": f1, "time": elapsed}
                print(f"  [{name}] '{text[:80]}' F1={f1:.3f}", flush=True)
                
            else:
                # CGRR variant
                t0 = time.time()
                text, diag = denoise_with_cgrr(
                    model, tokenizer, base_prompt,
                    max_new_tokens=args.max_new_tokens, steps=args.steps, mask_id=MASK_ID,
                    retriever=retriever, question_text=question,
                    remask_at_frac=cfg["remask_at_frac"], remask_pct=cfg["remask_pct"],
                )
                elapsed = time.time() - t0
                f1 = max(compute_f1(text, ga) for ga in gold_answers)
                result[name] = {"text": text, "f1": f1, "time": elapsed,
                               "n_remasked": diag.get("n_remasked", 0),
                               "partial": diag.get("partial_text", "")[:100],
                               "conf_stats": diag.get("confidence_stats", {})}
                print(f"  [{name}] '{text[:80]}' F1={f1:.3f} "
                      f"remasked={diag.get('n_remasked',0)}", flush=True)
        
        all_results.append(result)
    
    # Summary table
    n = len(all_results)
    print(f"\n{'='*60}", flush=True)
    print(f"CGRR v2 POC | LLaDA | {args.dataset} | N={n}", flush=True)
    print(f"{'='*60}", flush=True)
    
    config_names = [c["name"] for c in configs]
    for name in config_names:
        f1s = [r[name]["f1"] for r in all_results]
        mean_f1 = np.mean(f1s) * 100
        wins = sum(1 for r in all_results if r[name]["f1"] > r["baseline"]["f1"] + 0.01)
        ties = sum(1 for r in all_results if abs(r[name]["f1"] - r["baseline"]["f1"]) <= 0.01)
        print(f"  {name:30s}: F1={mean_f1:5.1f}%  wins={wins} ties={ties} losses={n-wins-ties}", flush=True)
    
    # Per-question table
    print(f"\nPer-question F1:", flush=True)
    header = f"{'Q':>3}"
    for name in config_names:
        header += f" {name[:12]:>12}"
    print(header, flush=True)
    
    for i, r in enumerate(all_results):
        row = f"{i+1:3d}"
        for name in config_names:
            row += f" {r[name]['f1']:12.3f}"
        print(row, flush=True)
    
    out_path = f"/projects/prjs1800/msc-thesis/07-daes/results/cgrr_v2_poc_{args.dataset}_{n}q.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
