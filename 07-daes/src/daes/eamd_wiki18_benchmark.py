#!/usr/bin/env python3
"""Benchmark-capable LLaDA/Dream wiki18 runner with shared frontier schema."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.qa_benchmark import build_record, write_dataset_artifacts, write_jsonl
import eamd_wiki18_full_llada as base

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm

try:
    import tiktoken
except ImportError:  # pragma: no cover - runtime env fallback
    tiktoken = None


class TimedRetriever:
    def __init__(self, retriever: base.Wiki18Retriever):
        self.retriever = retriever
        self.elapsed_sec = 0.0
        self.retrieval_calls = 0
        self.retrieved_passages: list[str] = []

    def retrieve_batch(self, queries: list[str], top_k: int):
        started = time.time()
        hits = self.retriever.retrieve_batch(queries, top_k=top_k)
        self.elapsed_sec += time.time() - started
        self.retrieval_calls += len(queries)
        for row in hits:
            self.retrieved_passages.extend(row)
        return hits

    def retrieve(self, query: str, top_k: int = 5):
        return self.retrieve_batch([query], top_k=top_k)[0]


class CountModelForwards:
    def __init__(self, model):
        self.model = model
        self.original_forward = model.forward
        self.count = 0

    def __enter__(self):
        def wrapped_forward(*args, **kwargs):
            self.count += 1
            return self.original_forward(*args, **kwargs)

        self.model.forward = wrapped_forward
        return self

    def __exit__(self, exc_type, exc, tb):
        self.model.forward = self.original_forward


def timed_call(model, fn):
    started = time.time()
    with CountModelForwards(model) as counter:
        result = fn()
    return result, time.time() - started, counter.count


def unique_passages(passages: list[str]) -> list[str]:
    seen = set()
    out = []
    for passage in passages:
        key = passage[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(passage)
    return out


def passage_token_count(passages: list[str]) -> int:
    if tiktoken is None:
        return sum(len((passage or "").split()) for passage in passages if passage)
    encoder = tiktoken.encoding_for_model("gpt-4o")
    return sum(len(encoder.encode(passage)) for passage in passages if passage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="llada", choices=["llada", "dream"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dataset", default="musique", choices=sorted(base.QUESTION_FILES))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--questions_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default="baseline,spread,aram,pool,eamd_micro")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--answer_tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=0)
    parser.add_argument("--retriever_encode_batch_size", type=int, default=64)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--micro_pivot_ratio", type=float, default=0.333)
    parser.add_argument("--micro_top_m", type=int, default=2)
    parser.add_argument("--micro_budget_min", type=int, default=1)
    parser.add_argument("--micro_kappa", type=float, default=8.0)
    parser.add_argument("--micro_tau_q", type=float, default=0.30)
    parser.add_argument("--micro_eta", type=float, default=0.5)
    parser.add_argument("--micro_phase1_guidance", default="baseline", choices=["baseline", "aram"])
    parser.add_argument("--neighbor_radius", type=int, default=0)
    parser.add_argument("--pool_seed_mode", default="baseline", choices=["baseline", "aram"])
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    methods = base.parse_methods(args.methods)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    retriever = base.Wiki18Retriever(
        device=os.environ.get("EAMD_RETRIEVER_DEVICE", "cpu"),
        encode_batch_size=args.retriever_encode_batch_size,
        num_threads=max(1, int(os.environ.get("OMP_NUM_THREADS", "16"))),
    )

    model_name = args.model_name
    if model_name is None:
        model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model_type == "dream" else "GSAI-ML/LLaDA-8B-Instruct"

    if args.model_type == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()

    base.MODEL_REF = model
    base.TOKENIZER_REF = tokenizer
    base.MODEL_TYPE_REF = args.model_type

    questions_file = base.resolve_questions_file(args.dataset, args.questions_file)
    all_questions = json.loads(Path(questions_file).read_text())
    end_idx = args.end_idx if args.end_idx is not None else args.start_idx + args.n_questions
    questions = all_questions[args.start_idx:end_idx]
    if not questions:
        raise ValueError("No questions selected.")

    records: list[dict] = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(questions, start=1):
        qid = str(item.get("qid") or item["id"])
        question = item["question"]
        gold_answers = item.get("golden_answers") or [item.get("answer", "")]
        should_log = base.should_log(idx, len(questions), args.log_every)
        if should_log:
            print(f"[{idx}/{len(questions)}] {qid}", flush=True)

        initial_retriever = TimedRetriever(retriever)
        initial_passages = initial_retriever.retrieve(question, top_k=args.initial_top_k)
        old_context = "\n\n".join(initial_passages)
        initial_ret_elapsed = initial_retriever.elapsed_sec

        baseline_answer = None
        baseline_tokens = None
        aram_answer = None
        aram_tokens = None

        if "baseline" in methods or "pool" in methods or "eamd_micro" in methods:
            (baseline_answer, baseline_tokens, baseline_conf), baseline_llm_elapsed, baseline_forwards = timed_call(
                model,
                lambda: base.short_generate(
                    model,
                    tokenizer,
                    old_context,
                    question,
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                    temperature=args.temperature,
                ),
            )
            records.append(
                build_record(
                    dataset=args.dataset,
                    qid=qid,
                    method="baseline",
                    model=model_name,
                    question=question,
                    gold_answers=gold_answers,
                    pred_answer=baseline_answer,
                    elapsed_sec_total=initial_ret_elapsed + baseline_llm_elapsed,
                    elapsed_sec_llm=baseline_llm_elapsed,
                    elapsed_sec_retrieval=initial_ret_elapsed,
                    retrieval_calls=1,
                    unique_chunks_read=len(initial_passages),
                    total_retrieved_tokens=passage_token_count(initial_passages),
                    loops_or_rounds=args.steps,
                    forward_passes=baseline_forwards,
                    denoising_steps=args.steps,
                    c0_passages=initial_passages,
                    final_passages=initial_passages,
                    extra={"avg_conf": baseline_conf},
                )
            )
            if should_log and "baseline" in methods:
                print(f"  baseline:    {baseline_answer}", flush=True)

        if "spread" in methods:
            (spread_answer, spread_stats), spread_llm_elapsed, spread_forwards = timed_call(
                model,
                lambda: base.spread_generate_shared(
                    model,
                    tokenizer,
                    old_context,
                    question,
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                    temperature=args.temperature,
                ),
            )
            records.append(
                build_record(
                    dataset=args.dataset,
                    qid=qid,
                    method="spread",
                    model=model_name,
                    question=question,
                    gold_answers=gold_answers,
                    pred_answer=spread_answer,
                    elapsed_sec_total=initial_ret_elapsed + spread_llm_elapsed,
                    elapsed_sec_llm=spread_llm_elapsed,
                    elapsed_sec_retrieval=initial_ret_elapsed,
                    retrieval_calls=1,
                    unique_chunks_read=len(initial_passages),
                    total_retrieved_tokens=passage_token_count(initial_passages),
                    loops_or_rounds=args.steps,
                    forward_passes=spread_forwards,
                    denoising_steps=args.steps,
                    c0_passages=initial_passages,
                    final_passages=initial_passages,
                    extra={"stats": spread_stats},
                )
            )
            if should_log:
                print(f"  spread:      {spread_answer}", flush=True)

        if "aram" in methods or "pool" in methods or "eamd_remask" in methods:
            (aram_answer, aram_tokens, aram_stats), aram_llm_elapsed, aram_forwards = timed_call(
                model,
                lambda: base.aram_generate_shared(
                    model,
                    tokenizer,
                    old_context,
                    question,
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                    temperature=args.temperature,
                    lambda_max=args.lambda_max,
                    beta=args.beta,
                ),
            )
            if "aram" in methods:
                records.append(
                    build_record(
                        dataset=args.dataset,
                        qid=qid,
                        method="aram",
                        model=model_name,
                        question=question,
                        gold_answers=gold_answers,
                        pred_answer=aram_answer,
                        elapsed_sec_total=initial_ret_elapsed + aram_llm_elapsed,
                        elapsed_sec_llm=aram_llm_elapsed,
                        elapsed_sec_retrieval=initial_ret_elapsed,
                        retrieval_calls=1,
                        unique_chunks_read=len(initial_passages),
                        total_retrieved_tokens=passage_token_count(initial_passages),
                        loops_or_rounds=args.steps,
                        forward_passes=aram_forwards,
                        denoising_steps=args.steps,
                        c0_passages=initial_passages,
                        final_passages=initial_passages,
                        extra={"stats": aram_stats},
                    )
                )
                if should_log:
                    print(f"  aram:        {aram_answer}", flush=True)

        expanded_passages = None
        candidates = []
        expand_ret = None
        expand_elapsed = 0.0
        expand_llm_elapsed = 0.0
        expand_forwards = 0
        if any(name in methods for name in ("pool", "eamd_regen", "eamd_remask")):
            pool_seed = aram_answer if args.pool_seed_mode == "aram" and aram_answer else baseline_answer
            expand_ret = TimedRetriever(retriever)
            (expanded_passages, candidates), expand_elapsed, expand_forwards = timed_call(
                model,
                lambda: base.expand_evidence(
                    expand_ret,
                    question,
                    old_context,
                    initial_passages,
                    [pool_seed] if pool_seed else [],
                    n_candidates=args.n_candidates,
                    expand_top_k=args.expand_top_k,
                ),
            )
            expand_llm_elapsed = max(0.0, expand_elapsed - expand_ret.elapsed_sec)

        if "pool" in methods and expanded_passages is not None and expand_ret is not None:
            (pool_answer, _, pool_conf), pool_llm_elapsed, pool_forwards = timed_call(
                model,
                lambda: base.short_generate(
                    model,
                    tokenizer,
                    "\n\n".join(expanded_passages),
                    question,
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                    temperature=args.temperature,
                ),
            )
            records.append(
                build_record(
                    dataset=args.dataset,
                    qid=qid,
                    method="pool",
                    model=model_name,
                    question=question,
                    gold_answers=gold_answers,
                    pred_answer=pool_answer,
                    elapsed_sec_total=initial_ret_elapsed + expand_elapsed + pool_llm_elapsed,
                    elapsed_sec_llm=expand_llm_elapsed + pool_llm_elapsed,
                    elapsed_sec_retrieval=initial_ret_elapsed + expand_ret.elapsed_sec,
                    retrieval_calls=1 + expand_ret.retrieval_calls,
                    unique_chunks_read=len(expanded_passages),
                    total_retrieved_tokens=passage_token_count(expanded_passages),
                    loops_or_rounds=args.steps,
                    forward_passes=expand_forwards + pool_forwards,
                    denoising_steps=args.steps,
                    c0_passages=initial_passages,
                    final_passages=expanded_passages,
                    extra={"avg_conf": pool_conf, "candidates": candidates},
                )
            )
            if should_log:
                print(f"  pool:        {pool_answer}", flush=True)

        if "eamd_regen" in methods and expanded_passages is not None and expand_ret is not None:
            new_context = "\n\n".join(expanded_passages)
            (regen_answer, regen_stats), regen_llm_elapsed, regen_forwards = timed_call(
                model,
                lambda: base.eamd_regen_shared(
                    model,
                    tokenizer,
                    question,
                    old_context,
                    new_context,
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                    temperature=args.temperature,
                    lambda_max=args.lambda_max,
                    beta=args.beta,
                ),
            )
            records.append(
                build_record(
                    dataset=args.dataset,
                    qid=qid,
                    method="eamd_regen",
                    model=model_name,
                    question=question,
                    gold_answers=gold_answers,
                    pred_answer=regen_answer,
                    elapsed_sec_total=initial_ret_elapsed + expand_elapsed + regen_llm_elapsed,
                    elapsed_sec_llm=expand_llm_elapsed + regen_llm_elapsed,
                    elapsed_sec_retrieval=initial_ret_elapsed + expand_ret.elapsed_sec,
                    retrieval_calls=1 + expand_ret.retrieval_calls,
                    unique_chunks_read=len(expanded_passages),
                    total_retrieved_tokens=passage_token_count(expanded_passages),
                    loops_or_rounds=args.steps,
                    forward_passes=expand_forwards + regen_forwards,
                    denoising_steps=args.steps,
                    c0_passages=initial_passages,
                    final_passages=expanded_passages,
                    extra={"stats": regen_stats, "candidates": candidates},
                )
            )
            if should_log:
                print(f"  eamd_regen:  {regen_answer}", flush=True)

        if "eamd_remask" in methods and expanded_passages is not None and expand_ret is not None:
            new_context = "\n\n".join(expanded_passages)
            seed_tokens = baseline_tokens if args.model_type == "dream" or aram_tokens is None else aram_tokens
            (remask_answer, remask_stats), remask_llm_elapsed, remask_forwards = timed_call(
                model,
                lambda: base.eamd_remask_shared(
                    model,
                    tokenizer,
                    question,
                    old_context,
                    new_context,
                    seed_tokens,
                    steps=args.steps,
                    temperature=args.temperature,
                    lambda_max=args.lambda_max,
                    beta=args.beta,
                ),
            )
            records.append(
                build_record(
                    dataset=args.dataset,
                    qid=qid,
                    method="eamd_remask",
                    model=model_name,
                    question=question,
                    gold_answers=gold_answers,
                    pred_answer=remask_answer,
                    elapsed_sec_total=initial_ret_elapsed + expand_elapsed + remask_llm_elapsed,
                    elapsed_sec_llm=expand_llm_elapsed + remask_llm_elapsed,
                    elapsed_sec_retrieval=initial_ret_elapsed + expand_ret.elapsed_sec,
                    retrieval_calls=1 + expand_ret.retrieval_calls,
                    unique_chunks_read=len(expanded_passages),
                    total_retrieved_tokens=passage_token_count(expanded_passages),
                    loops_or_rounds=args.steps,
                    forward_passes=expand_forwards + remask_forwards,
                    denoising_steps=args.steps,
                    c0_passages=initial_passages,
                    final_passages=expanded_passages,
                    extra={"stats": remask_stats, "candidates": candidates},
                )
            )
            if should_log:
                print(f"  eamd_remask: {remask_answer}", flush=True)

        if "eamd_micro" in methods:
            micro_ret = TimedRetriever(retriever)
            (micro_answer, _, micro_stats), micro_llm_elapsed, micro_forwards = timed_call(
                model,
                lambda: base.eamd_micro_shared(
                    model,
                    tokenizer,
                    micro_ret,
                    question,
                    initial_passages,
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                    temperature=args.temperature,
                    lambda_max=args.lambda_max,
                    beta=args.beta,
                    expand_top_k=args.expand_top_k,
                    pivot_ratio=args.micro_pivot_ratio,
                    top_m=args.micro_top_m,
                    budget_min=args.micro_budget_min,
                    kappa=args.micro_kappa,
                    tau_q=args.micro_tau_q,
                    eta=args.micro_eta,
                    neighbor_radius=args.neighbor_radius,
                    phase1_guidance=args.micro_phase1_guidance,
                ),
            )
            final_passages = unique_passages(initial_passages + micro_ret.retrieved_passages)
            records.append(
                build_record(
                    dataset=args.dataset,
                    qid=qid,
                    method="eamd_micro",
                    model=model_name,
                    question=question,
                    gold_answers=gold_answers,
                    pred_answer=micro_answer,
                    elapsed_sec_total=initial_ret_elapsed + micro_ret.elapsed_sec + micro_llm_elapsed,
                    elapsed_sec_llm=micro_llm_elapsed,
                    elapsed_sec_retrieval=initial_ret_elapsed + micro_ret.elapsed_sec,
                    retrieval_calls=1 + micro_ret.retrieval_calls,
                    unique_chunks_read=len(final_passages),
                    total_retrieved_tokens=passage_token_count(final_passages),
                    loops_or_rounds=args.steps,
                    forward_passes=micro_forwards,
                    denoising_steps=args.steps,
                    c0_passages=initial_passages,
                    final_passages=final_passages,
                    extra={"stats": micro_stats},
                )
            )
            if should_log:
                print(f"  eamd_micro:  {micro_answer}", flush=True)

    write_jsonl(output_dir / "per_example.jsonl", records)
    write_dataset_artifacts(records, output_dir)
    print(f"Saved benchmark artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
