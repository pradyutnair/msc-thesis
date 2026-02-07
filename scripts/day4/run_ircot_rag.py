"""Run IRCoT with FlashRAG — with per-round retrieval tracking.

IRCoT (Trivedi et al., ACL 2023) interleaves retrieval with chain-of-thought:
  think -> search -> think -> search -> answer

This script reimplements the IRCoT loop with explicit per-round tracking of:
  - Retrieved doc IDs and scores at each round
  - Query reformulations (generated thoughts used as retrieval queries)
  - Accumulated retrieval pool growth
  - Per-round retrieval recall against ground-truth supporting docs

The retriever and generator must be loaded simultaneously since IRCoT
interleaves retrieval and generation. On A100-80GB + 128GB RAM this fits:
  - GPU: Qwen2.5-7B (~14GB VRAM)
  - CPU: FAISS flat index (~61GB RAM) + E5-base-v2 (~400MB)
"""

import argparse
import os
import sys
import time
import json

# CRITICAL: Set threading BEFORE any library import for FAISS parallelism
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"

from flashrag.config import Config
from flashrag.utils import get_dataset, get_generator, get_retriever, ircot_pred_parse
from flashrag.prompt import PromptTemplate
from flashrag.evaluator import Evaluator
import faiss


# ── IRCoT prompt (from Trivedi et al. / FlashRAG) ──────────────────────────
IRCOT_INSTRUCTION = (
    "You serve as an intelligent assistant, adept at facilitating users through "
    "complex, multi-hop reasoning across multiple documents. This task is "
    "illustrated through demonstrations, each consisting of a document set "
    "paired with a relevant question and its multi-hop reasoning thoughts. "
    "Your task is to generate one thought for current step, DON\'T generate "
    "the whole thoughts at once! If you reach what you believe to be the "
    'final step, start with "So the answer is:".'
)

IRCOT_EXAMPLE = (
    "Wikipedia Title: Kurram Garhi\n"
    "Kurram Garhi is a small village located near the city of Bannu, which is "
    "the part of Khyber Pakhtunkhwa province of Pakistan. Its population is "
    "approximately 35000.\n\n"
    "Wikipedia Title: 2001\u201302 UEFA Champions League second group stage\n"
    "Eight winners and eight runners-up from the first group stage were drawn "
    "into four groups of four teams, each containing two group winners and two "
    "runners-up.\n\n"
    "Wikipedia Title: Satellite tournament\n"
    "A satellite tournament is either a minor tournament or event on a "
    "competitive sporting tour or one of a group of such tournaments that form "
    "a series played in the same country or region.\n\n"
    "Wikipedia Title: Trojkrsti\n"
    "Trojkrsti is a village in Municipality of Prilep, Republic of Macedonia.\n\n"
    "Wikipedia Title: Telephone numbers in Ascension Island\n"
    "Country Code:+ 247< br> International Call Prefix: 00 Ascension Island "
    "does not share the same country code( +290) with the rest of St Helena.\n\n"
    "Question: Are both Kurram Garhi and Trojkrsti located in the same country?\n"
    "Thought: Kurram Garhi is located in the country of Pakistan. Trojkrsti is "
    "located in the country of Republic of Macedonia. Thus, they are not in the "
    "same country. So the answer is: no.\n\n"
)


def run_ircot_with_tracking(config, test_data, max_iter=5):
    """Run IRCoT with explicit per-round retrieval tracking.

    Returns:
        test_data: Updated dataset with predictions and per-round tracking
        timing: Dict with retrieval/generation timing breakdowns
    """
    # Build IRCoT prompt template (no chat template, following original paper)
    prompt_template = PromptTemplate(
        config=config,
        system_prompt=f"{IRCOT_INSTRUCTION}\n\n{IRCOT_EXAMPLE}",
        user_prompt="{reference}Question: {question}\nThought:",
        reference_template="Wikipedia Title: {title}\n{text}\n\n",
        enable_chat=False,
    )

    # Load retriever and generator simultaneously (required for interleaved IRCoT)
    print("Loading retriever...")
    t_load = time.time()
    retriever = get_retriever(config)
    print(f"Retriever loaded in {time.time() - t_load:.1f}s")

    print("Loading generator (vLLM)...")
    t_load = time.time()
    generator = get_generator(config)
    print(f"Generator loaded in {time.time() - t_load:.1f}s")

    # CRITICAL: vLLM/PyTorch CUDA init resets OMP threads to 1
    # Must explicitly set FAISS threads after generator loading
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "16"))
    print(f"FAISS OMP threads before fix: {faiss.omp_get_max_threads()}")
    faiss.omp_set_num_threads(omp_threads)
    print(f"FAISS OMP threads after fix: {faiss.omp_get_max_threads()}")

    questions = [item.question for item in test_data]
    N = len(questions)
    print(f"\nRunning IRCoT on {N} examples, max_iter={max_iter}")

    # ── Per-item tracking structures ───────────────────────────────────────
    batch_thoughts = {i: [] for i in range(N)}
    doc2score = [{} for _ in range(N)]      # accumulated doc->score
    id2doc = [{} for _ in range(N)]          # accumulated doc_id->doc_object
    per_round_doc_ids = [{} for _ in range(N)]  # round -> list of new doc IDs
    per_round_queries = [{} for _ in range(N)]  # round -> query string
    per_round_thoughts = [{} for _ in range(N)] # round -> generated thought
    n_iterations = [0] * N                       # actual iterations per item

    # ── Round 0: Initial retrieval on original questions ───────────────────
    print("\n--- Round 0: Initial retrieval on original questions ---")
    t0 = time.time()
    results, scoress = retriever.batch_search(questions, return_score=True)
    t_initial_ret = time.time() - t0
    print(f"Initial retrieval: {t_initial_ret:.1f}s ({t_initial_ret/N:.3f}s/query)")

    for i in range(N):
        per_round_queries[i][0] = questions[i]
        per_round_doc_ids[i][0] = [doc["id"] for doc in results[i]]
        for doc, score in zip(results[i], scoress[i]):
            doc2score[i][doc["id"]] = score
            id2doc[i][doc["id"]] = doc

    total_retrieval_time = t_initial_ret
    total_generation_time = 0.0

    # ── Iterative IRCoT loop ──────────────────────────────────────────────
    active_ids = list(range(N))

    for iter_num in range(max_iter):
        if not active_ids:
            print(f"All items terminated before iter {iter_num}")
            break

        print(f"\n--- Iteration {iter_num}: {len(active_ids)} active items ---")

        # Build current retrieval results for each active item (sorted ascending by score)
        current_results = []
        for item_id in active_ids:
            sorted_pairs = sorted(doc2score[item_id].items(), key=lambda x: x[1], reverse=False)
            sorted_docs = [id2doc[item_id][did] for did, _ in sorted_pairs]
            current_results.append(sorted_docs)

        # Build prompts
        input_prompts = [
            prompt_template.get_string(
                question=test_data[item_id].question,
                retrieval_result=current_results[idx],
                previous_gen=" ".join(batch_thoughts[item_id]),
            )
            for idx, item_id in enumerate(active_ids)
        ]

        # Generate one thought per active item (stop at sentence boundary)
        t_gen_start = time.time()
        new_thoughts = generator.generate(input_prompts, stop=[".", "\n"])
        t_gen = time.time() - t_gen_start
        total_generation_time += t_gen
        print(f"  Generation: {t_gen:.1f}s for {len(active_ids)} items")

        # Process generated thoughts
        new_active_ids = []
        retrieval_queries = []
        retrieval_item_ids = []

        for idx, item_id in enumerate(active_ids):
            thought = new_thoughts[idx]
            batch_thoughts[item_id].append(thought)
            per_round_thoughts[item_id][iter_num] = thought
            n_iterations[item_id] = iter_num + 1

            # Store intermediate output (matches FlashRAG format)
            test_data[item_id].update_output(
                f"intermediate_output_iter{iter_num}",
                {"input_prompt": input_prompts[idx], "new_thought": thought},
            )

            # Check termination: did the model produce a final answer?
            if "So the answer is:" not in thought:
                new_active_ids.append(item_id)
                retrieval_queries.append(thought)
                retrieval_item_ids.append(item_id)

        terminated = len(active_ids) - len(new_active_ids)
        print(f"  Terminated: {terminated} items (answered), {len(new_active_ids)} continue")

        # Batch retrieval using generated thoughts as queries
        if retrieval_queries:
            t_ret_start = time.time()
            new_results, new_scoress = retriever.batch_search(
                retrieval_queries, return_score=True
            )
            t_ret = time.time() - t_ret_start
            total_retrieval_time += t_ret
            print(f"  Retrieval: {t_ret:.1f}s for {len(retrieval_queries)} queries")

            # Track new docs per round and merge into accumulated pool
            round_num = iter_num + 1  # round 0 was initial, this is round iter_num+1
            for i, item_id in enumerate(retrieval_item_ids):
                per_round_queries[item_id][round_num] = retrieval_queries[i]
                per_round_doc_ids[item_id][round_num] = [
                    doc["id"] for doc in new_results[i]
                ]

                # Merge new docs into accumulated pool (keep max score)
                for doc, score in zip(new_results[i], new_scoress[i]):
                    did = doc["id"]
                    id2doc[item_id][did] = doc
                    if did in doc2score[item_id]:
                        doc2score[item_id][did] = max(doc2score[item_id][did], score)
                    else:
                        doc2score[item_id][did] = score

            # Count new unique docs added this round
            new_doc_counts = []
            for i, item_id in enumerate(retrieval_item_ids):
                prev_ids = set()
                for r in range(round_num):
                    if r in per_round_doc_ids[item_id]:
                        prev_ids.update(per_round_doc_ids[item_id][r])
                current_ids = set(per_round_doc_ids[item_id].get(round_num, []))
                new_unique = len(current_ids - prev_ids)
                new_doc_counts.append(new_unique)
            if new_doc_counts:
                avg_new = sum(new_doc_counts) / len(new_doc_counts)
                print(f"  Avg new unique docs: {avg_new:.1f}")

        active_ids = new_active_ids

    # ── Finalize: store accumulated results and predictions ────────────────
    print(f"\n=== Finalizing results ===")
    for i in range(N):
        # Store final accumulated retrieval results
        sorted_pairs = sorted(doc2score[i].items(), key=lambda x: x[1], reverse=False)
        final_docs = [id2doc[i][did] for did, _ in sorted_pairs]
        test_data[i].update_output("retrieval_result", final_docs)

        # Store prediction (all accumulated thoughts)
        test_data[i].update_output("pred", " ".join(batch_thoughts[i]))

        # Store per-round tracking data for analysis
        test_data[i].update_output("per_round_doc_ids", per_round_doc_ids[i])
        test_data[i].update_output("per_round_queries", per_round_queries[i])
        test_data[i].update_output("per_round_thoughts", per_round_thoughts[i])
        test_data[i].update_output("n_iterations", n_iterations[i])
        test_data[i].update_output("total_accumulated_docs", len(doc2score[i]))

    # Iteration stats
    avg_iters = sum(n_iterations) / N
    iter_dist = {}
    for n in n_iterations:
        iter_dist[n] = iter_dist.get(n, 0) + 1
    print(f"Average iterations: {avg_iters:.2f}")
    print(f"Iteration distribution: {dict(sorted(iter_dist.items()))}")

    timing = {
        "total_retrieval": total_retrieval_time,
        "total_generation": total_generation_time,
        "total": total_retrieval_time + total_generation_time,
        "initial_retrieval": t_initial_ret,
    }

    return test_data, timing


def main():
    parser = argparse.ArgumentParser(description="Run IRCoT with per-round tracking")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--max_iter", type=int, default=5,
                        help="Max IRCoT iterations (default: 5)")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = Config(config_file_path=args.config)

    print(f"Dataset: {config['dataset_name']}, Split: {config['split']}")
    print(f"Generator: {config['generator_model']}")
    print(f"Retriever: {config['retrieval_method']}, top-k: {config['retrieval_topk']}")
    print(f"Max IRCoT iterations: {args.max_iter}")
    print(f"Save dir: {config['save_dir']}")

    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    # Run IRCoT with tracking
    test_data, timing = run_ircot_with_tracking(config, test_data, max_iter=args.max_iter)

    # Parse predictions (extract answer after "So the answer is:")
    print("\nParsing predictions...")
    test_data = ircot_pred_parse(test_data)

    # Evaluate
    print("Evaluating...")
    evaluator = Evaluator(config)
    eval_result = evaluator.evaluate(test_data)
    print(f"\n{eval_result}")

    # Print timing summary
    print(f"\n=== Timing Summary ===")
    print(f"Initial retrieval: {timing['initial_retrieval']:.1f}s")
    print(f"Total retrieval: {timing['total_retrieval']:.1f}s")
    print(f"Total generation: {timing['total_generation']:.1f}s")
    print(f"Total: {timing['total']:.1f}s")
    print(f"Per example: {timing['total']/len(test_data):.2f}s")
    print(f"\nResults saved to {config['save_dir']}")


if __name__ == "__main__":
    main()
