"""Oracle Anchored-Select Pilot: validate whether ARAM + anchored passage selection can beat ARAM alone.

Uses gold bridge entities from MuSiQue question_decomposition to test the upper bound
of DNMR-Select+ARAM before implementing the full pipeline.

Controls (all on same 50q):
  1. aram_c0          : ARAM decode on original C0 (baseline to beat)
  2. aram_qa0_select  : ARAM seed -> Q+a0 retrieval -> anchored select -> ARAM decode (no bridges)
  3. aram_oracle_anch : ARAM seed -> gold bridge retrieval -> anchored select -> ARAM decode
  4. aram_oracle_full : ARAM seed -> gold bridge retrieval -> full replace top-5 -> ARAM decode

Anchored selection: keep top-K passages from C0 (by retriever score), replace bottom-(5-K) with
best new passages from DNMR pool. Default K=3 (3+2 split).
"""
import argparse, json, math, os, sys, time
import torch
import numpy as np

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, prepare_logits, get_mask_id, decode_answer, compute_f1,
    short_generate, short_user_prompt, QUESTION_FILES,
    spread_generate_shared, aram_generate_shared, build_short_prompt,
    _neg_entropy,
)
from eamd_iterative import extract_bridges_from_answer
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18

MUSIQUE_DEV = "/projects/prjs1800/datasets/musique/musique_ans_v1.0_dev.jsonl"


def load_gold_bridges(arag_questions):
    """Match ARAG questions to MuSiQue dev to get gold bridge entities."""
    with open(MUSIQUE_DEV) as f:
        mq_all = [json.loads(l) for l in f.readlines()]
    mq_by_q = {q["question"]: q for q in mq_all}
    
    bridges = {}
    for aq in arag_questions:
        mq = mq_by_q.get(aq["question"])
        if mq and "question_decomposition" in mq:
            # All intermediate answers except the final one
            bridge_entities = [step["answer"] for step in mq["question_decomposition"][:-1]]
            bridges[aq["question"]] = bridge_entities
        else:
            bridges[aq["question"]] = []
    return bridges


def anchored_select(retriever, question, bridge_queries, initial_passages,
                    initial_scores, n_keep, budget=5, expand_top_k=5):
    """Anchor-preserving passage selection.
    
    Keep top-n_keep from C0 (by retriever score). Fill remaining (budget - n_keep) slots
    from DNMR-retrieved passages, ranked by max similarity to bridge query set.
    De-duplicate against kept C0 passages.
    
    Returns: (selected_passages, metrics_dict)
    """
    n_new = budget - n_keep
    
    # Sort initial passages by score, keep top-n_keep
    sorted_indices = np.argsort(-np.array(initial_scores))
    kept_indices = sorted_indices[:n_keep]
    kept_passages = [initial_passages[i] for i in kept_indices]
    kept_set = set(p[:120] for p in kept_passages)
    
    if n_new <= 0 or not bridge_queries:
        return kept_passages + [initial_passages[i] for i in sorted_indices[n_keep:budget]], {
            "n_kept": n_keep, "n_swapped": 0, "n_candidates": 0,
        }
    
    # Retrieve with bridge queries
    query_texts = ["query: " + bq for bq in bridge_queries]
    results = retriever.retrieve_batch(query_texts, expand_top_k)
    
    # Collect unique new passages (not in kept set)
    candidates = []
    seen = set(kept_set)
    for result_list in results:
        for passage in result_list:
            key = passage[:120]
            if key not in seen:
                candidates.append(passage)
                seen.add(key)
    
    if not candidates:
        # No new passages found, fill from remaining C0
        remaining = [initial_passages[i] for i in sorted_indices[n_keep:]]
        return kept_passages + remaining[:n_new], {
            "n_kept": n_keep, "n_swapped": 0, "n_candidates": 0,
        }
    
    # Score candidates by max similarity to bridge query set
    q_vecs = retriever.model.encode(query_texts, normalize_embeddings=True,
                                     convert_to_numpy=True).astype(np.float32)
    c_vecs = retriever.model.encode(candidates, normalize_embeddings=True,
                                     convert_to_numpy=True).astype(np.float32)
    # Max similarity across all bridge queries for each candidate
    sim_matrix = c_vecs @ q_vecs.T  # (n_cand, n_queries)
    max_sims = sim_matrix.max(axis=1)
    top_cand_idx = np.argsort(-max_sims)[:n_new]
    new_passages = [candidates[i] for i in top_cand_idx]
    
    selected = kept_passages + new_passages
    
    # Compute overlap metrics
    initial_set = set(p[:120] for p in initial_passages)
    n_actually_new = sum(1 for p in new_passages if p[:120] not in initial_set)
    
    return selected, {
        "n_kept": n_keep,
        "n_swapped": len(new_passages),
        "n_actually_new": n_actually_new,
        "n_candidates": len(candidates),
    }


def compute_contain(pred, gold):
    """Check if gold answer is contained in prediction."""
    return 1.0 if gold.lower() in pred.lower() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada", choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_keep", type=int, default=3, help="Passages to keep from C0 (anchor)")
    parser.add_argument("--budget", type=int, default=5, help="Total passage budget")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== Oracle Anchored-Select Pilot ===", flush=True)
    print(f"Model: {args.model}, n_keep={args.n_keep}, budget={args.budget}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model
    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).cuda().eval()

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"  Model loaded in {time.time() - t_start:.1f}s", flush=True)

    # Load retriever and questions
    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    # Load gold bridges
    gold_bridges = load_gold_bridges(questions)
    n_with_bridges = sum(1 for b in gold_bridges.values() if b)
    print(f"  Gold bridges available for {n_with_bridges}/{len(questions)} questions", flush=True)

    # Initial retrieval with scores
    print("Batch initial retrieval...", flush=True)
    query_texts = ["query: " + q["question"] for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.budget)
    
    # Also get retriever scores for anchoring
    all_scores = []
    for qi, q in enumerate(questions):
        q_emb = retriever.model.encode(["query: " + q["question"]], normalize_embeddings=True,
                                        convert_to_numpy=True).astype(np.float32)
        p_embs = retriever.model.encode(all_initial[qi], normalize_embeddings=True,
                                         convert_to_numpy=True).astype(np.float32)
        scores = (q_emb @ p_embs.T)[0].tolist()
        all_scores.append(scores)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    # Methods to evaluate
    methods = ["aram_c0", "aram_qa0_select", "aram_oracle_anchored", "aram_oracle_full_replace"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        initial_scores = all_scores[qi]
        c0_ctx = "\n\n".join(initial)
        bridges = gold_bridges.get(qtext, [])
        rec = {"id": q.get("id", f"dev_{qi}"), "question": qtext, "gold": gold,
               "bridges": bridges}

        # === 1. ARAM on C0 (baseline to beat) ===
        aram_ans, _, aram_extra = aram_generate_shared(
            model, tokenizer, c0_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
        )
        _, _, f1_aram = compute_f1(aram_ans, gold)
        em_aram = 1.0 if aram_ans.strip().lower() == gold.strip().lower() else 0.0
        contain_aram = compute_contain(aram_ans, gold)
        totals["aram_c0"]["f1"] += f1_aram
        totals["aram_c0"]["em"] += em_aram
        totals["aram_c0"]["contain"] += contain_aram
        rec["aram_c0"] = {"answer": aram_ans, "f1": round(f1_aram, 4)}

        # === 2. ARAM(Q+a0 select) — answer-only retrieval, no bridges ===
        # Use ARAM answer as seed for retrieval
        qa0_queries = [qtext + " " + aram_ans]
        sel_qa0, met_qa0 = anchored_select(
            retriever, qtext, qa0_queries, initial, initial_scores,
            n_keep=args.n_keep, budget=args.budget,
        )
        qa0_ctx = "\n\n".join(sel_qa0)
        qa0_ans, _, _ = aram_generate_shared(
            model, tokenizer, qa0_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
        )
        _, _, f1_qa0 = compute_f1(qa0_ans, gold)
        em_qa0 = 1.0 if qa0_ans.strip().lower() == gold.strip().lower() else 0.0
        contain_qa0 = compute_contain(qa0_ans, gold)
        totals["aram_qa0_select"]["f1"] += f1_qa0
        totals["aram_qa0_select"]["em"] += em_qa0
        totals["aram_qa0_select"]["contain"] += contain_qa0
        rec["aram_qa0_select"] = {"answer": qa0_ans, "f1": round(f1_qa0, 4),
                                   "metrics": met_qa0}

        # === 3. ARAM(oracle anchored) — gold bridge queries, anchored selection ===
        if bridges:
            bridge_queries = [qtext + " " + b for b in bridges]
        else:
            bridge_queries = [qtext + " " + aram_ans]  # fallback
        
        sel_oracle, met_oracle = anchored_select(
            retriever, qtext, bridge_queries, initial, initial_scores,
            n_keep=args.n_keep, budget=args.budget,
        )
        oracle_ctx = "\n\n".join(sel_oracle)
        oracle_ans, _, _ = aram_generate_shared(
            model, tokenizer, oracle_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
        )
        _, _, f1_oracle = compute_f1(oracle_ans, gold)
        em_oracle = 1.0 if oracle_ans.strip().lower() == gold.strip().lower() else 0.0
        contain_oracle = compute_contain(oracle_ans, gold)
        totals["aram_oracle_anchored"]["f1"] += f1_oracle
        totals["aram_oracle_anchored"]["em"] += em_oracle
        totals["aram_oracle_anchored"]["contain"] += contain_oracle
        rec["aram_oracle_anchored"] = {"answer": oracle_ans, "f1": round(f1_oracle, 4),
                                        "metrics": met_oracle}

        # === 4. ARAM(oracle full replace) — gold bridge queries, replace all 5 ===
        sel_full, met_full = anchored_select(
            retriever, qtext, bridge_queries, initial, initial_scores,
            n_keep=0, budget=args.budget,  # n_keep=0 = full replace
        )
        full_ctx = "\n\n".join(sel_full)
        full_ans, _, _ = aram_generate_shared(
            model, tokenizer, full_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
        )
        _, _, f1_full = compute_f1(full_ans, gold)
        em_full = 1.0 if full_ans.strip().lower() == gold.strip().lower() else 0.0
        contain_full = compute_contain(full_ans, gold)
        totals["aram_oracle_full_replace"]["f1"] += f1_full
        totals["aram_oracle_full_replace"]["em"] += em_full
        totals["aram_oracle_full_replace"]["contain"] += contain_full
        rec["aram_oracle_full_replace"] = {"answer": full_ans, "f1": round(f1_full, 4),
                                            "metrics": met_full}

        results.append(rec)
        elapsed = time.time() - tq

        if (qi + 1) % 5 == 0 or qi == 0:
            n = qi + 1
            print(f"\n[{n}/{len(questions)}] {elapsed:.1f}s  Q: {qtext[:60]}", flush=True)
            print(f"  Gold: {gold}", flush=True)
            for m in methods:
                avg_f1 = totals[m]["f1"] / n
                avg_em = totals[m]["em"] / n
                avg_c = totals[m]["contain"] / n
                print(f"  {m:30s}  F1={avg_f1:.3f}  EM={avg_em:.3f}  Contain={avg_c:.3f}", flush=True)

    # Final summary
    n = len(questions)
    print(f"\n{'='*80}", flush=True)
    print(f"FINAL RESULTS ({n} questions, model={args.model}, n_keep={args.n_keep})", flush=True)
    print(f"{'='*80}", flush=True)
    for m in methods:
        avg_f1 = totals[m]["f1"] / n
        avg_em = totals[m]["em"] / n
        avg_c = totals[m]["contain"] / n
        print(f"  {m:30s}  F1={avg_f1:.3f}  EM={avg_em:.3f}  Contain={avg_c:.3f}", flush=True)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    summary = {m: {k: round(v / n, 4) for k, v in totals[m].items()} for m in methods}
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved to {args.output}", flush=True)
    print(f"Total time: {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
