"""
ReAct inference for dLLMs (LLaDA / Dream) with E5 dense retrieval over FlashRAG Wiki18 corpus.

Each ReAct turn = one denoising pass:
  Turn 1: [system + question + MASK*N] → think + tool_call
  Turn 2: [... + tool_response + MASK*N] → think + tool_call/answer
  ...until <answer> or max_turns

Retrieval: E5-base-v2 + FAISS IVFPQ over 21M Wiki18 passages.
"""

import argparse, json, os, sys, time, re, string
import numpy as np, torch
import faiss
import sqlite3
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------------------
# Dense Retrieval Tool (replaces DLLM-Searcher's Google Search)
# ---------------------------------------------------------------------------

class DenseRetriever:
    """E5-base-v2 + FAISS over FlashRAG Wiki18 corpus."""
    def __init__(self, index_path, db_path, encoder_name="intfloat/e5-base-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading FAISS index from {index_path}...", flush=True)
        self.index = faiss.read_index(index_path)
        print(f"Index loaded: {self.index.ntotal} vectors", flush=True)
        self.db = sqlite3.connect(db_path)
        self.encoder = SentenceTransformer(encoder_name, device="cpu")
        print("Retriever ready.", flush=True)

    def search(self, queries, top_k=10):
        """Search for multiple queries, return formatted results (like Google Search tool)."""
        if isinstance(queries, str):
            queries = [queries]
        
        all_results = []
        for query in queries:
            emb = self.encoder.encode([f"query: {query}"], normalize_embeddings=True)
            scores, ids = self.index.search(emb.astype(np.float32), top_k)
            
            snippets = []
            for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
                if idx < 0: continue
                cursor = self.db.cursor()
                cursor.execute("SELECT title, contents FROM passages WHERE id = ?", (int(idx) + 1,))
                row = cursor.fetchone()
                if row:
                    title, contents = row
                    snippet = contents[:500]
                    snippets.append(f"{rank+1}. [{title}]\n{snippet}")
            
            result = f"A search for '{query}' found {len(snippets)} results:\n\n## Results\n" + "\n\n".join(snippets)
            all_results.append(result)
        
        return "\n=======\n".join(all_results)


# ---------------------------------------------------------------------------
# dLLM Generation (single turn)
# ---------------------------------------------------------------------------

def dllm_generate_turn(model, tokenizer, messages, max_new_tokens=512, steps=64,
                        temperature=0.1, mask_id=None, ar_shift=False):
    """Generate one ReAct turn via denoising."""
    device = model.device
    if mask_id is None:
        mask_id = tokenizer.mask_token_id or 126336
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    
    # Don't exceed model's max length
    max_len = 8192
    if n_prefix + max_new_tokens > max_len:
        max_new_tokens = max(64, max_len - n_prefix)
    
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
            out = model(x, attention_mask=attn)
        logits = out.logits
        if ar_shift:  # Dream needs this, LLaDA doesn't
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        
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


# ---------------------------------------------------------------------------
# ReAct Loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a search assistant that answers questions by searching for information.

You have access to a search tool. To use it, write:
<tool_call>
{"name": "search", "arguments": {"query": ["your search query"]}}
</tool_call>

After getting search results, reason about them and either search again or give your final answer.

When you have enough information, provide your answer as:
<answer>your concise answer (1-10 words)</answer>

Always search before answering. Think step by step."""


def react_loop(model, tokenizer, retriever, question, max_turns=4,
               max_new_tokens=512, steps=64, temperature=0.1,
               mask_id=None, ar_shift=False):
    """Run ReAct loop: think → search → think → answer."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    
    all_retrieved = []
    
    for turn in range(max_turns):
        t0 = time.time()
        output = dllm_generate_turn(
            model, tokenizer, messages,
            max_new_tokens=max_new_tokens, steps=steps,
            temperature=temperature, mask_id=mask_id, ar_shift=ar_shift,
        )
        gen_time = time.time() - t0
        
        print(f"  Turn {turn+1} ({gen_time:.1f}s): {output[:150]}", flush=True)
        
        # Check for answer
        if "<answer>" in output and "</answer>" in output:
            answer = output.split("<answer>")[1].split("</answer>")[0].strip()
            return answer, turn + 1, all_retrieved
        
        # Check for tool call
        if "<tool_call>" in output and "</tool_call>" in output:
            messages.append({"role": "assistant", "content": output})
            
            # Parse tool call
            try:
                tc_text = output.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                tc = json.loads(tc_text)
                queries = tc.get("arguments", {}).get("query", [])
                if isinstance(queries, str): queries = [queries]
                
                # Call retriever
                t0 = time.time()
                results = retriever.search(queries, top_k=5)
                ret_time = time.time() - t0
                print(f"  Retrieved ({ret_time:.1f}s): {len(queries)} queries", flush=True)
                
                all_retrieved.extend(queries)
                messages.append({"role": "user", "content": f"<tool_response>\n{results}\n</tool_response>"})
            except Exception as e:
                messages.append({"role": "user", "content": f"<tool_response>\nError: {e}\n</tool_response>"})
        else:
            # No tool call and no answer — force answer on next turn
            messages.append({"role": "assistant", "content": output})
            messages.append({"role": "user", "content": "Please provide your final answer using <answer>your answer</answer>"})
    
    # Max turns reached — extract whatever we can
    last = messages[-1]["content"] if messages else ""
    if "<answer>" in last:
        return last.split("<answer>")[1].split("</answer>")[0].strip(), max_turns, all_retrieved
    return output[:100], max_turns, all_retrieved


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_answer(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation); return " ".join(s.split())

def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not pt or not gt: return 0
    common = set(pt) & set(gt)
    if not common: return 0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return 2*p*r/(p+r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada", choices=["llada", "dream"])
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "musique", "2wikimultihopqa"])
    parser.add_argument("--n_questions", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    args = parser.parse_args()
    
    # Load retriever
    retriever = DenseRetriever(
        index_path="/projects/prjs1800/datasets/flashrag/indexes/e5_IVFPQ.index",
        db_path="/projects/prjs1800/datasets/flashrag/wiki18_100w.db",
    )
    
    # Load model
    if args.model == "llada":
        model_name = "GSAI-ML/LLaDA-8B-Instruct"
        mask_id = 126336
        ar_shift = False
    else:
        model_name = "Dream-org/Dream-v0-Instruct-7B"
        mask_id = 151666
        ar_shift = True
    
    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    print("Model loaded.", flush=True)
    
    # Load questions
    ds_path = f"/projects/prjs1800/datasets/flashrag/{args.dataset}/test.jsonl"
    questions = [json.loads(l) for l in open(ds_path)][:args.n_questions]
    print(f"Loaded {len(questions)} questions from {args.dataset}", flush=True)
    
    tag = f"react_{args.model}_{args.dataset}"
    out_path = os.path.join(args.output_dir, f"{tag}_{args.n_questions}q.jsonl")
    open(out_path, "w").close()
    
    sum_f1 = 0
    for i, q in enumerate(questions):
        gold_answers = q["golden_answers"]
        print(f"\n[{i+1}/{len(questions)}] Q: {q['question'][:80]}", flush=True)
        
        t0 = time.time()
        answer, turns, retrieved = react_loop(
            model, tokenizer, retriever, q["question"],
            max_turns=args.max_turns, max_new_tokens=args.max_new_tokens,
            steps=args.steps, mask_id=mask_id, ar_shift=ar_shift,
        )
        elapsed = time.time() - t0
        
        # Compute F1 against best gold answer
        f1 = max(compute_f1(answer, ga) for ga in gold_answers)
        contain = any(ga.lower() in answer.lower() for ga in gold_answers)
        sum_f1 += f1
        
        pred = {
            "id": q["id"], "question": q["question"],
            "gold": gold_answers, "pred": answer,
            "f1": round(f1, 4), "contain": contain,
            "turns": turns, "queries": retrieved,
            "time": round(elapsed, 2),
        }
        with open(out_path, "a") as fw: fw.write(json.dumps(pred) + "\n")
        
        print(f"  Answer: {answer[:80]}", flush=True)
        print(f"  Gold: {gold_answers[0][:80]}", flush=True)
        print(f"  F1={f1:.2f} contain={contain} turns={turns} time={elapsed:.1f}s", flush=True)
    
    n = len(questions)
    cn = sum(1 for l in open(out_path) if json.loads(l)["contain"])
    print(f"\n{'='*60}", flush=True)
    print(f"REACT dLLM | {args.model} | {args.dataset} | N={n}", flush=True)
    print(f"  F1:      {sum_f1/n*100:.1f}%", flush=True)
    print(f"  Contain: {cn}/{n} = {cn/n*100:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
