"""Baselines runner: SPREAD, ARAM, iSPREAD, iARAM at 1000q scale.

Separate from iDNMR to avoid restarting the main scale run.
Runs on the same questions/retriever/corpus for fair comparison.

Methods:
  spread    - single-query SPREAD (relevance-weighted token ordering)
  aram      - single-query ARAM (SNR guidance, context vs prior)
  ispread   - iterative: SPREAD decode + answer-conditioned bridge extraction
  iaram     - iterative: ARAM decode + answer-conditioned bridge extraction

Run: python -u src/daes/baselines_1k.py --model dream --dataset musique --n_questions 200
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate,
    short_user_prompt, QUESTION_FILES,
    spread_generate_shared, aram_generate_shared,
)
from eamd_iterative import extract_bridges_from_answer
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18


def expand_evidence(retriever, question, seed_answer, bridge_cands, current_passages, expand_top_k=3):
    queries = [f"query: {question} {seed_answer}"]
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if text and len(text) > 1:
            queries.append(f"query: {question} {text}")
    results = retriever.retrieve_batch(queries, expand_top_k)
    existing = set(current_passages)
    new_passages = []
    for result_list in results:
        for passage in result_list:
            if passage not in existing:
                new_passages.append(passage)
                existing.add(passage)
    return list(current_passages) + new_passages, new_passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=200)
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== Baselines Runner ===", flush=True)
    print(f"Model: {args.model}, Dataset: {args.dataset}, Rounds: {args.max_rounds}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"  Model loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    print("Batch initial retrieval...", flush=True)
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["spread", "aram", "ispread", "iaram"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        # --- Timing + metrics ---
        method_stats = {}

        # === SPREAD: single-query ===
        t0 = time.time()
        spread_ans, spread_extra = spread_generate_shared(
            model, tokenizer, old_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
        )
        method_stats["spread"] = {
            "wall_sec": round(time.time() - t0, 3),
            "forward_passes": args.steps + 1,  # decode steps + query embedding
            "retrieval_queries": 0,
            "total_passages": len(initial),
        }

        # === ARAM: single-query ===
        t0 = time.time()
        aram_ans, _, aram_extra = aram_generate_shared(
            model, tokenizer, old_ctx, qtext,
            steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
        )
        method_stats["aram"] = {
            "wall_sec": round(time.time() - t0, 3),
            "forward_passes": args.steps,  # 2 branches batched per step
            "retrieval_queries": 0,
            "total_passages": len(initial),
        }

        # === iSPREAD: iterative ===
        t0 = time.time()
        ispread_passages = list(initial)
        ispread_ctx = old_ctx
        # seed answer for iterative expansion
        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                        steps=16, n_tokens=16, temperature=0.0)
        prev_ispread_ans = seed_ans
        ispread_rounds = 0
        ispread_retrieval_queries = 0
        ispread_new_per_round = []
        for r in range(args.max_rounds):
            bc = extract_bridges_from_answer(
                model, tokenizer, ispread_ctx, qtext, prev_ispread_ans, args.n_candidates
            )
            ispread_passages, new_p = expand_evidence(
                retriever, qtext, prev_ispread_ans, bc,
                ispread_passages, args.expand_top_k
            )
            ispread_retrieval_queries += 1 + len(bc)  # 1 for answer query + 1 per bridge
            ispread_new_per_round.append(len(new_p))
            ispread_ctx = "\n\n".join(ispread_passages)
            ispread_ans, _ = spread_generate_shared(
                model, tokenizer, ispread_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
            )
            ispread_rounds = r + 1
            if len(new_p) == 0 and ispread_ans.strip().lower() == prev_ispread_ans.strip().lower():
                break
            prev_ispread_ans = ispread_ans
        method_stats["ispread"] = {
            "wall_sec": round(time.time() - t0, 3),
            "forward_passes": 16 + ispread_rounds * (1 + 1 + args.steps),  # seed(16) + rounds*(extract+qembed+decode)
            "retrieval_queries": ispread_retrieval_queries,
            "total_passages": len(ispread_passages),
            "rounds": ispread_rounds,
            "new_per_round": ispread_new_per_round,
        }

        # === iARAM: iterative ===
        t0 = time.time()
        iaram_passages = list(initial)
        iaram_ctx = old_ctx
        prev_iaram_ans = seed_ans  # reuse same seed
        iaram_rounds = 0
        iaram_retrieval_queries = 0
        iaram_new_per_round = []
        for r in range(args.max_rounds):
            bc = extract_bridges_from_answer(
                model, tokenizer, iaram_ctx, qtext, prev_iaram_ans, args.n_candidates
            )
            iaram_passages, new_p = expand_evidence(
                retriever, qtext, prev_iaram_ans, bc,
                iaram_passages, args.expand_top_k
            )
            iaram_retrieval_queries += 1 + len(bc)
            iaram_new_per_round.append(len(new_p))
            iaram_ctx = "\n\n".join(iaram_passages)
            iaram_ans, _, _ = aram_generate_shared(
                model, tokenizer, iaram_ctx, qtext,
                steps=args.steps, n_tokens=args.answer_tokens, temperature=0.0,
            )
            iaram_rounds = r + 1
            if len(new_p) == 0 and iaram_ans.strip().lower() == prev_iaram_ans.strip().lower():
                break
            prev_iaram_ans = iaram_ans
        method_stats["iaram"] = {
            "wall_sec": round(time.time() - t0, 3),
            "forward_passes": 16 + iaram_rounds * (1 + args.steps),  # seed(16) + rounds*(extract+decode)
            "retrieval_queries": iaram_retrieval_queries,
            "total_passages": len(iaram_passages),
            "rounds": iaram_rounds,
            "new_per_round": iaram_new_per_round,
        }

        elapsed = time.time() - tq
        row = {"id": q.get("qid") or q.get("id", f"dev_{qi}"),
               "question": qtext, "gold": gold, "elapsed": round(elapsed, 2),
               "method_stats": method_stats}

        for method, ans in [("spread", spread_ans), ("aram", aram_ans),
                            ("ispread", ispread_ans), ("iaram", iaram_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        results.append(row)

        log_this = (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1
        if log_this:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s)", flush=True)
            for m in methods:
                print(f"  {m:16s} {row[m]['answer'][:40]:40s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()} for m in methods}
            # Compute efficiency summary
            eff = {}
            for m in methods:
                walls = [r["method_stats"][m]["wall_sec"] for r in results if m in r.get("method_stats", {})]
                fps = [r["method_stats"][m]["forward_passes"] for r in results if m in r.get("method_stats", {})]
                rcs = [r["method_stats"][m]["retrieval_queries"] for r in results if m in r.get("method_stats", {})]
                tps = [r["method_stats"][m].get("total_passages", 0) for r in results if m in r.get("method_stats", {})]
                rnds = [r["method_stats"][m].get("rounds", 0) for r in results if m in r.get("method_stats", {})]
                if walls:
                    eff[m] = {
                        "mean_wall_sec": round(sum(walls) / len(walls), 3),
                        "mean_forward_passes": round(sum(fps) / len(fps), 1),
                        "mean_retrieval_queries": round(sum(rcs) / len(rcs), 1),
                        "mean_total_passages": round(sum(tps) / len(tps), 1) if tps else 0,
                        "mean_rounds": round(sum(rnds) / len(rnds), 2) if rnds and any(rnds) else 0,
                    }
            with open(args.output, "w") as f:
                json.dump({"summary": summary, "efficiency": eff, "results": results,
                           "config": vars(args),
                           "timing": {"elapsed_sec": round(time.time() - t_start, 1)}}, f, indent=2)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\n{'Method':<16s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 38)
    for m in methods:
        s = summary[m]
        print(f"{m:<16s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")
    print(f"\nTotal: {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, n):.1f}s/q)")


if __name__ == "__main__":
    main()
