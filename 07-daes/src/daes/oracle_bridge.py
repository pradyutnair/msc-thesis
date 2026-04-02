"""Oracle bridge retrieval experiment for MuSiQue.

Compares:
1. baseline: decode from initial evidence C0
2. pool: single-round expansion using predicted bridge candidates
3. oracle_bridge: single-round expansion using gold bridge entities from
   MuSiQue question_decomposition

This script reuses the existing wiki18 retriever and short-answer generation
stack from eamd_v2_wiki18.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import dllm
from types import SimpleNamespace

import eamd_v2_wiki18
from eamd_v2_wiki18 import (
    Wiki18Retriever,
    compute_em,
    compute_f1,
    expand_evidence,
    normalize_answer,
    short_generate,
)

MUSIQUE_DEV = "/projects/prjs1800/datasets/musique/musique_full_v1.0_dev.jsonl"


def load_musique(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def unique_passages(passages: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for passage in passages:
        if passage not in seen:
            seen.add(passage)
            out.append(passage)
    return out


def get_gold_bridges(example: dict[str, Any]) -> list[str]:
    """Extract gold bridge entities from MuSiQue decomposition answers.

    Uses all unique decomposition answers except the final answer when it matches
    the overall gold answer. If that removes everything, falls back to all but
    the last decomposition answer.
    """
    gold = normalize_answer(example.get("answer", ""))
    decomposition = example.get("question_decomposition") or []
    answers = []
    seen = set()
    for step in decomposition:
        ans = (step.get("answer") or "").strip()
        if not ans:
            continue
        key = normalize_answer(ans)
        if key in seen:
            continue
        seen.add(key)
        answers.append(ans)

    bridges = [ans for ans in answers if normalize_answer(ans) != gold]
    if not bridges and len(answers) > 1:
        bridges = answers[:-1]
    return bridges


def expand_with_gold_bridges(
    retriever: Wiki18Retriever,
    question: str,
    initial_passages: list[str],
    gold_bridges: list[str],
    expand_top_k: int = 3,
) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    seen = set()
    for bridge in gold_bridges:
        bridge = bridge.strip()
        if len(bridge) <= 1:
            continue
        query = f"{question} {bridge[:100]}"
        if query not in seen:
            seen.add(query)
            queries.append(query)

    all_passages = list(initial_passages)
    if queries:
        for hits in retriever.retrieve_batch(queries, top_k=expand_top_k):
            all_passages.extend(hits)
    return unique_passages(all_passages), queries


def evaluate_answer(pred: str, gold: str) -> dict[str, float]:
    _, _, f1 = compute_f1(pred, gold)
    em = compute_em(pred, gold)
    contain = float(normalize_answer(gold) in normalize_answer(pred))
    return {"f1": round(f1, 4), "em": round(em, 4), "contain": round(contain, 4)}


def load_model_and_tokenizer(model_name: str):
    if model_name == "dream":
        model_args = SimpleNamespace(model_name_or_path="Dream-org/Dream-v0-Instruct-7B")
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        name = "GSAI-ML/LLaDA-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            name, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda().eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="dream", choices=["dream", "llada"])
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()
    print("=== Oracle Bridge Experiment ===", flush=True)
    print(
        f"Model: {args.model}, Questions: {args.n_questions}, Steps: {args.steps}, Tokens: {args.answer_tokens}",
        flush=True,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model, tokenizer = load_model_and_tokenizer(args.model)
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    questions = load_musique(MUSIQUE_DEV)[args.start_idx : args.start_idx + args.n_questions]
    queries = [f"query: {row['question']}" for row in questions]
    initial_batches = retriever.retrieve_batch(queries, args.initial_top_k)

    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in ("baseline", "pool", "oracle_bridge")}
    results = []

    for idx, example in enumerate(questions, start=1):
        qid = example.get("id", f"dev_{idx - 1}")
        question = example["question"]
        gold = example["answer"]
        initial_passages = initial_batches[idx - 1]
        old_context = "\n\n".join(initial_passages)

        baseline_answer, _, baseline_conf = short_generate(
            model,
            tokenizer,
            old_context,
            question,
            steps=args.steps,
            n_tokens=args.answer_tokens,
            temperature=args.temperature,
        )
        baseline_metrics = evaluate_answer(baseline_answer, gold)

        pool_passages, predicted_candidates = expand_evidence(
            retriever,
            question,
            old_context,
            initial_passages,
            [baseline_answer],
            n_candidates=args.n_candidates,
            expand_top_k=args.expand_top_k,
        )
        pool_context = "\n\n".join(pool_passages)
        pool_answer, _, _ = short_generate(
            model,
            tokenizer,
            pool_context,
            question,
            steps=args.steps,
            n_tokens=args.answer_tokens,
            temperature=args.temperature,
        )
        pool_metrics = evaluate_answer(pool_answer, gold)

        gold_bridges = get_gold_bridges(example)
        oracle_passages, oracle_queries = expand_with_gold_bridges(
            retriever,
            question,
            initial_passages,
            gold_bridges,
            expand_top_k=args.expand_top_k,
        )
        oracle_context = "\n\n".join(oracle_passages)
        oracle_answer, _, _ = short_generate(
            model,
            tokenizer,
            oracle_context,
            question,
            steps=args.steps,
            n_tokens=args.answer_tokens,
            temperature=args.temperature,
        )
        oracle_metrics = evaluate_answer(oracle_answer, gold)

        row = {
            "id": qid,
            "question": question,
            "gold": gold,
            "gold_bridges": gold_bridges,
            "predicted_candidates": [cand["text"] for cand in predicted_candidates],
            "baseline": {
                "answer": baseline_answer,
                "avg_conf": round(baseline_conf, 4),
                **baseline_metrics,
            },
            "pool": {
                "answer": pool_answer,
                "n_passages": len(pool_passages),
                **pool_metrics,
            },
            "oracle_bridge": {
                "answer": oracle_answer,
                "n_passages": len(oracle_passages),
                "queries": oracle_queries,
                **oracle_metrics,
            },
        }
        results.append(row)

        for method in ("baseline", "pool", "oracle_bridge"):
            for metric in ("f1", "em", "contain"):
                totals[method][metric] += row[method][metric]

        if idx == 1 or idx % args.log_every == 0 or idx == len(questions):
            print(f"[{idx}/{len(questions)}] {qid}", flush=True)
            print(
                f"  baseline      {baseline_answer[:80]:80s} F1={baseline_metrics['f1']:.3f}",
                flush=True,
            )
            print(
                f"  pool          {pool_answer[:80]:80s} F1={pool_metrics['f1']:.3f}",
                flush=True,
            )
            print(
                f"  oracle_bridge {oracle_answer[:80]:80s} F1={oracle_metrics['f1']:.3f}",
                flush=True,
            )

    n = max(1, len(results))
    summary = {
        method: {metric: round(value / n, 4) for metric, value in totals[method].items()}
        for method in totals
    }

    print("\nMethod           F1     EM  Contain", flush=True)
    print("-----------------------------------", flush=True)
    for method in ("baseline", "pool", "oracle_bridge"):
        s = summary[method]
        print(f"{method:<14s} {s['f1']:>5.3f}  {s['em']:>5.3f}   {s['contain']:>5.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "summary": summary,
                "results": results,
                "config": vars(args),
                "timing": {"total_sec": round(time.time() - t0, 1)},
            },
            f,
            indent=2,
        )
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
