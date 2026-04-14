"""Remask hyperparameter sweep — loads model ONCE, sweeps params internally.
Run: python -u src/daes/remask_sweep.py --model dream --n_questions 50
"""
import argparse, json, itertools, time, sys, os
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    load_model_and_tokenizer, Wiki18Retriever, load_questions,
    build_short_pair, decode_short, eamd_remask_shared, decode_answer,
    compute_f1, get_mask_id, expand_evidence, QUESTION_FILES,
)
import torch

SWEEP = {
    "tau": [0.01, 0.05, 0.20, 0.50, 1.00],
    "prior": [0.20, 0.40, 0.60],
    "threshold": [0.05, 0.15, 0.30],
    "seed_mode": ["baseline", "aram"],
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--answer_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/results/eamd_v2_wiki18/remask_sweep")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {args.model} model...")
    model, tokenizer = load_model_and_tokenizer(args.model)
    print(f"Loading retriever...")
    retriever = Wiki18Retriever()
    questions = load_questions(QUESTION_FILES[args.dataset], args.n_questions)
    print(f"Loaded {len(questions)} questions")

    # Pre-compute all baseline/aram seeds and expanded evidence for each question
    # so we only pay retrieval + seed generation cost ONCE
    print("Pre-computing seeds and expanded evidence...")
    precomputed = []
    mask_id = get_mask_id(tokenizer)
    for qi, q in enumerate(questions):
        question_text = q["question"]
        gold = q["answer"]
        # Initial retrieval
        initial_passages = retriever.retrieve_batch([f"query: {question_text}"], args.initial_top_k)[0]
        old_context = "\n\n".join(initial_passages)

        # Baseline seed
        baseline_answer, baseline_tokens, _ = decode_short(
            model, tokenizer, question_text, old_context,
            n_answer=args.answer_tokens, steps=args.steps, temperature=args.temperature
        )

        # ARAM seed
        from eamd_v2_wiki18 import aram_decode
        aram_answer, aram_tokens, _ = aram_decode(
            model, tokenizer, question_text, old_context,
            n_answer=args.answer_tokens, steps=args.steps, temperature=args.temperature
        )

        # Expanded evidence
        expanded_passages = expand_evidence(
            retriever, question_text, baseline_answer, initial_passages,
            model, tokenizer, old_context,
            n_candidates=args.n_candidates, expand_top_k=args.expand_top_k,
            n_answer=args.answer_tokens, steps=args.steps, temperature=args.temperature
        )
        new_context = "\n\n".join(expanded_passages)

        precomputed.append({
            "idx": qi,
            "qid": q.get("id", f"dev_{qi}"),
            "question": question_text,
            "gold": gold,
            "old_context": old_context,
            "new_context": new_context,
            "baseline_tokens": baseline_tokens,
            "baseline_answer": baseline_answer,
            "aram_tokens": aram_tokens,
            "aram_answer": aram_answer,
        })
        if (qi + 1) % 10 == 0:
            print(f"  Pre-computed {qi+1}/{len(questions)}")

    print(f"Pre-computation done. Starting sweep...")

    # Now sweep remask params — only the remask step, no model reload or re-retrieval
    configs = list(itertools.product(
        SWEEP["tau"], SWEEP["prior"], SWEEP["threshold"], SWEEP["seed_mode"]
    ))
    print(f"Total configs: {len(configs)}")

    results = {}
    for ci, (tau, prior, threshold, seed_mode) in enumerate(configs):
        tag = f"tau{tau}_prior{prior}_thresh{threshold}_seed{seed_mode}"
        f1_sum = 0.0
        em_sum = 0.0
        contain_sum = 0.0
        t0 = time.time()

        for pc in precomputed:
            seed_tokens = pc[f"{seed_mode}_tokens"]
            answer, _, meta = eamd_remask_shared(
                model, tokenizer,
                pc["question"], pc["old_context"], pc["new_context"],
                seed_tokens,
                steps=args.steps, temperature=args.temperature,
                tau=tau, remask_prior=prior, remask_cost=0.0, remask_threshold=threshold,
            )
            f1 = compute_f1(answer, pc["gold"])
            f1_sum += f1
            em_sum += float(answer.strip().lower() == pc["gold"].strip().lower())
            contain_sum += float(pc["gold"].strip().lower() in answer.strip().lower())

        n = len(precomputed)
        elapsed = time.time() - t0
        result = {
            "tau": tau, "prior": prior, "threshold": threshold, "seed_mode": seed_mode,
            "f1": round(f1_sum / n, 4),
            "em": round(em_sum / n, 4),
            "contain": round(contain_sum / n, 4),
            "seconds": round(elapsed, 1),
        }
        results[tag] = result
        print(f"[{ci+1}/{len(configs)}] {tag}: F1={result[f1]:.3f} EM={result[em]:.3f} ({elapsed:.1f}s)")

    # Save and print summary
    outfile = os.path.join(args.output_dir, f"remask_sweep_{args.model}_{args.dataset}.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")

    # Print sorted by F1
    print("\n" + "=" * 60)
    print(f"TOP 10 by F1:")
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["f1"])
    for tag, r in sorted_results[:10]:
        print(f"  {tag}: F1={r[f1]:.3f} EM={r[em]:.3f} contain={r[contain]:.3f} ({r[seconds]:.0f}s)")

if __name__ == "__main__":
    main()
