#!/usr/bin/env python3
"""Sequential IRCoT benchmark runner with per-question timing."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import faiss
import tiktoken

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.qa_benchmark import build_record, write_dataset_artifacts, write_jsonl

# Keep thread config identical to the original IRCoT runner.
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "16")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "16")
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "16")
os.environ["NUMEXPR_NUM_THREADS"] = os.environ.get("NUMEXPR_NUM_THREADS", "16")

from flashrag.config import Config
from flashrag.prompt import PromptTemplate
from flashrag.utils import get_dataset, get_generator, get_retriever


IRCOT_INSTRUCTION = (
    "You serve as an intelligent assistant, adept at facilitating users through "
    "complex, multi-hop reasoning across multiple documents. This task is "
    "illustrated through demonstrations, each consisting of a document set "
    "paired with a relevant question and its multi-hop reasoning thoughts. "
    "Your task is to generate one thought for current step, DON'T generate "
    'the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'
)

IRCOT_EXAMPLE = (
    "Wikipedia Title: Kurram Garhi\n"
    "Kurram Garhi is a small village located near the city of Bannu, which is "
    "the part of Khyber Pakhtunkhwa province of Pakistan. Its population is "
    "approximately 35000.\n\n"
    "Wikipedia Title: 2001–02 UEFA Champions League second group stage\n"
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

FINAL_ANSWER_PREFIX = "So the answer is:"


def doc_to_passage(doc: dict) -> str:
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    if title and text:
        return f'"{title}"\n{text}'
    return text or title


def extract_final_answer(raw_pred: str) -> str:
    if FINAL_ANSWER_PREFIX in raw_pred:
        return raw_pred.split(FINAL_ANSWER_PREFIX, 1)[1].strip()
    return raw_pred.strip()


def count_retrieved_tokens(passages: list[str]) -> int:
    encoder = tiktoken.encoding_for_model("gpt-4o")
    return sum(len(encoder.encode(passage)) for passage in passages if passage)


def run_single_question(item, retriever, generator, prompt_template: PromptTemplate, max_iter: int, dataset_name: str, model_name: str) -> dict:
    question = item.question
    gold_answers = item.golden_answers
    qid = item.id

    doc2score: dict[str, float] = {}
    id2doc: dict[str, dict] = {}
    per_round_doc_ids: dict[int, list[str]] = {}
    per_round_queries: dict[int, str] = {}
    per_round_thoughts: dict[int, str] = {}
    thoughts: list[str] = []

    retrieval_elapsed = 0.0
    generation_elapsed = 0.0
    retrieval_calls = 0
    llm_calls = 0

    t0 = time.time()
    results, scores = retriever.batch_search([question], return_score=True)
    retrieval_elapsed += time.time() - t0
    retrieval_calls += 1
    initial_docs = results[0]
    initial_scores = scores[0]
    per_round_queries[0] = question
    per_round_doc_ids[0] = [doc["id"] for doc in initial_docs]
    for doc, score in zip(initial_docs, initial_scores):
        doc2score[doc["id"]] = score
        id2doc[doc["id"]] = doc

    for iter_num in range(max_iter):
        sorted_pairs = sorted(doc2score.items(), key=lambda x: x[1], reverse=False)
        current_results = [id2doc[doc_id] for doc_id, _ in sorted_pairs]
        prompt = prompt_template.get_string(
            question=question,
            retrieval_result=current_results,
            previous_gen=" ".join(thoughts),
        )

        t_gen = time.time()
        thought = generator.generate([prompt], stop=[".", "\n"])[0]
        generation_elapsed += time.time() - t_gen
        llm_calls += 1
        thoughts.append(thought)
        per_round_thoughts[iter_num] = thought

        if FINAL_ANSWER_PREFIX in thought:
            break

        t_ret = time.time()
        new_results, new_scores = retriever.batch_search([thought], return_score=True)
        retrieval_elapsed += time.time() - t_ret
        retrieval_calls += 1
        new_docs = new_results[0]
        new_doc_scores = new_scores[0]
        round_idx = iter_num + 1
        per_round_queries[round_idx] = thought
        per_round_doc_ids[round_idx] = [doc["id"] for doc in new_docs]
        for doc, score in zip(new_docs, new_doc_scores):
            doc_id = doc["id"]
            id2doc[doc_id] = doc
            if doc_id in doc2score:
                doc2score[doc_id] = max(doc2score[doc_id], score)
            else:
                doc2score[doc_id] = score

    final_pairs = sorted(doc2score.items(), key=lambda x: x[1], reverse=False)
    final_docs = [id2doc[doc_id] for doc_id, _ in final_pairs]
    raw_pred = " ".join(thoughts)
    pred = extract_final_answer(raw_pred)
    initial_passages = [doc_to_passage(doc) for doc in initial_docs]
    final_passages = [doc_to_passage(doc) for doc in final_docs]
    total_elapsed = retrieval_elapsed + generation_elapsed

    return build_record(
        dataset=dataset_name,
        qid=qid,
        method="ircot",
        model=model_name,
        question=question,
        gold_answers=gold_answers,
        pred_answer=pred,
        elapsed_sec_total=total_elapsed,
        elapsed_sec_llm=generation_elapsed,
        elapsed_sec_retrieval=retrieval_elapsed,
        retrieval_calls=retrieval_calls,
        unique_chunks_read=len({doc["id"] for doc in final_docs}),
        total_retrieved_tokens=count_retrieved_tokens(final_passages),
        loops_or_rounds=len(thoughts),
        llm_calls=llm_calls,
        c0_passages=initial_passages,
        final_passages=final_passages,
        extra={
            "raw_pred": raw_pred,
            "per_round_doc_ids": per_round_doc_ids,
            "per_round_queries": per_round_queries,
            "per_round_thoughts": per_round_thoughts,
            "retrieval_result": final_docs,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IRCoT benchmark runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_iter", type=int, default=5)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = Config(config_file_path=args.config)
    dataset_name = str(config["dataset_name"])
    model_name = str(config["generator_model"])

    prompt_template = PromptTemplate(
        config=config,
        system_prompt=f"{IRCOT_INSTRUCTION}\n\n{IRCOT_EXAMPLE}",
        user_prompt="{reference}Question: {question}\nThought:",
        reference_template="Wikipedia Title: {title}\n{text}\n\n",
        enable_chat=False,
    )

    retriever = get_retriever(config)
    generator = get_generator(config)
    faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))

    all_split = get_dataset(config)
    test_data = all_split["test"]
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(test_data)
    if args.limit is not None:
        end_idx = min(end_idx, start_idx + args.limit)

    selected = [test_data[idx] for idx in range(start_idx, end_idx)]
    records = [
        run_single_question(item, retriever, generator, prompt_template, args.max_iter, dataset_name, model_name)
        for item in selected
    ]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["save_dir"]) / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_dir / "per_example.jsonl", records)
    write_dataset_artifacts(records, output_dir)
    print(f"Saved benchmark artifacts to {output_dir}")


if __name__ == "__main__":
    main()
