"""Interrupted denoising benchmark for multi-hop QA.

Single-file implementation of:
- baseline RAG (tau=0)
- interrupted denoising at arbitrary schedules
- snapshot pool endpoint (tau=T)
- optional DNMR pool baseline for comparison

Example pilot:
  python -u src/daes/interrupted_denoising.py \
      --models dream,llada \
      --dataset musique \
      --n_questions 50 \
      --schedules baseline,4,8,12,16,24,32 \
      --output results/interrupted_denoising/musique_50q.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import dllm
import eamd_v2_wiki18
from dllm.pipelines.dream.sampler import sample_tokens
from eamd_v2_wiki18 import (
    QUESTION_FILES,
    Wiki18Retriever,
    _neg_entropy,
    build_short_prompt,
    compute_em,
    compute_f1,
    decode_answer,
    extract_candidates_generic,
    get_mask_id,
    normalize_answer,
    prepare_logits,
    unique_passages,
)


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_schedule_spec(text: str, steps: int) -> list[dict[str, Any]]:
    """Parse schedule strings.

    Supported items:
    - baseline
    - pool
    - dnmr_pool
    - 4
    - 8+16
    - 32 (alias for pool when steps=32)
    """
    schedules: list[dict[str, Any]] = []
    seen = set()

    for raw in parse_csv(text):
        key = raw.lower()
        if key == "baseline":
            spec = {"name": "baseline", "kind": "baseline", "interrupt_steps": []}
        elif key in {"pool", str(steps)}:
            spec = {"name": "pool", "kind": "pool", "interrupt_steps": [steps]}
        elif key == "dnmr_pool":
            spec = {"name": "dnmr_pool", "kind": "dnmr_pool", "interrupt_steps": [steps]}
        else:
            parts = [int(part.strip()) for part in raw.split("+") if part.strip()]
            if not parts:
                raise ValueError(f"Invalid schedule: {raw}")
            for tau in parts:
                if tau <= 0 or tau >= steps:
                    raise ValueError(
                        f"Interrupted schedules must use 1 <= tau < steps; got {tau} with steps={steps}"
                    )
            name = "id_tau" + "_".join(str(tau) for tau in parts)
            spec = {"name": name, "kind": "interrupted", "interrupt_steps": parts}

        if spec["name"] in seen:
            continue
        seen.add(spec["name"])
        schedules.append(spec)

    return schedules


def full_attention_mask(length: int, device: torch.device | str):
    if eamd_v2_wiki18.MODEL_TYPE_REF == "dream":
        return "full"
    return torch.ones((1, length), dtype=torch.long, device=device)


def score_answer(pred: str, gold: str) -> dict[str, float]:
    precision, recall, f1 = compute_f1(pred, gold)
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    contain = float(bool(gold_norm) and gold_norm in pred_norm)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "em": float(compute_em(pred, gold)),
        "contain": contain,
    }


def build_query(question: str, snapshot_text: str) -> str:
    snapshot_text = " ".join(snapshot_text.split())
    if snapshot_text:
        return f"query: {question} {snapshot_text}"
    return f"query: {question}"


@torch.inference_mode()
def decode_with_trace(
    model,
    tokenizer,
    context: str,
    question: str,
    *,
    steps: int,
    n_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Run standard denoising decode and keep a per-step entropy trace."""
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    entropy_trajectory: list[dict[str, Any]] = []

    for step in range(steps):
        if remaining <= 0:
            break
        answer_state = x[0, n_prefix:n_prefix + n_tokens]
        masked_local = (answer_state == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        out = model(x, attention_mask=full_attention_mask(x.shape[1], device))
        logits = prepare_logits(out.logits)
        masked_logits = logits[0, masked_local + n_prefix]
        probs = torch.softmax(masked_logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1).mean().item()
        entropy_trajectory.append(
            {
                "step": step + 1,
                "mean_masked_entropy": float(entropy),
                "n_masked": int(len(masked_local)),
            }
        )

        confidence, x0 = sample_tokens(
            masked_logits,
            temperature=temperature,
            neg_entropy=_neg_entropy(),
        )
        n_commit = min(k_per_step, remaining, len(confidence))
        if step == steps - 1:
            n_commit = min(remaining, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= n_commit

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()
    return {
        "answer": decode_answer(tokenizer, answer_tokens),
        "answer_tokens": answer_tokens,
        "entropy_trajectory": entropy_trajectory,
        "n_prefix": n_prefix,
    }


def committed_token_dump(tokenizer, answer_state: torch.Tensor, mask_id: int) -> list[dict[str, Any]]:
    committed = []
    for pos, token_id in enumerate(answer_state.tolist()):
        if token_id == mask_id:
            continue
        committed.append(
            {
                "position": pos,
                "token_id": int(token_id),
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False),
            }
        )
    return committed


def snapshot_from_logits(
    tokenizer,
    logits: torch.Tensor,
    x: torch.Tensor,
    n_prefix: int,
    n_tokens: int,
    *,
    snapshot_mode: str,
) -> dict[str, Any]:
    """Build a retrieval snapshot from pre-computed logits (no extra forward pass)."""
    mask_id = get_mask_id(tokenizer)
    answer_state = x[0, n_prefix:n_prefix + n_tokens].clone()
    masked_local = (answer_state == mask_id).nonzero(as_tuple=True)[0]
    full_tokens = answer_state.clone()
    if len(masked_local) > 0:
        full_tokens[masked_local] = torch.argmax(logits[0, masked_local + n_prefix], dim=-1)

    if snapshot_mode == "committed":
        kept = [tok for tok in answer_state.tolist() if tok != mask_id]
        snapshot_text = tokenizer.decode(kept, skip_special_tokens=True).strip() if kept else ""
    else:
        snapshot_text = decode_answer(tokenizer, full_tokens)

    masked_entropy = None
    if len(masked_local) > 0:
        masked_logits = logits[0, masked_local + n_prefix]
        probs = torch.softmax(masked_logits, dim=-1)
        masked_entropy = float(
            (-(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1).mean()).item()
        )

    return {
        "snapshot_text": snapshot_text,
        "snapshot_tokens": full_tokens,
        "answer_state": answer_state,
        "masked_entropy": masked_entropy,
        "committed_tokens": committed_token_dump(tokenizer, answer_state, mask_id),
    }


def expand_with_snapshot(
    retriever: Wiki18Retriever,
    question: str,
    snapshot_text: str,
    current_passages: list[str],
    *,
    top_k: int,
) -> tuple[list[str], list[str], str]:
    query = build_query(question, snapshot_text)
    hits = retriever.retrieve(query, top_k=top_k)
    existing = set(current_passages)
    new_passages = [passage for passage in hits if passage not in existing]
    return unique_passages(current_passages + new_passages), new_passages, query


def expand_with_dnmr_candidates(
    retriever: Wiki18Retriever,
    question: str,
    seed_answer: str,
    bridge_candidates: list[dict[str, Any]],
    current_passages: list[str],
    *,
    top_k: int,
) -> tuple[list[str], list[str], list[str]]:
    queries = [build_query(question, seed_answer)]
    for cand in bridge_candidates:
        text = cand.get("text", "").strip()
        if text:
            queries.append(build_query(question, text))

    results = retriever.retrieve_batch(queries, top_k=top_k)
    existing = set(current_passages)
    new_passages: list[str] = []
    for hits in results:
        for passage in hits:
            if passage in existing:
                continue
            existing.add(passage)
            new_passages.append(passage)
    return unique_passages(current_passages + new_passages), new_passages, queries


@torch.inference_mode()
def interrupted_generate(
    model,
    tokenizer,
    retriever: Wiki18Retriever,
    *,
    initial_passages: list[str],
    question: str,
    steps: int = 32,
    n_tokens: int = 32,
    temperature: float = 0.0,
    interrupt_steps: list[int] | None = None,
    top_k: int = 3,
    snapshot_mode: str = "full",
) -> dict[str, Any]:
    """Interrupted denoising with fixed interruption schedule.

    At each tau in ``interrupt_steps``, the current canvas is snapshotted,
    retrieval is run with the snapshot, the prompt is rebuilt with the expanded
    context, and committed answer tokens are restored before denoising resumes.
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)
    schedule = sorted({int(step) for step in (interrupt_steps or []) if 0 < int(step) < steps})

    current_passages = unique_passages(list(initial_passages))
    context = "\n\n".join(current_passages)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    x = torch.tensor([prefix_ids + [mask_id] * n_tokens], dtype=torch.long, device=device)

    k_per_step = max(1, math.ceil(n_tokens / steps))
    entropy_trajectory: list[dict[str, Any]] = []
    interruptions: list[dict[str, Any]] = []

    for step in range(steps):
        answer_state = x[0, n_prefix:n_prefix + n_tokens]
        masked_local = (answer_state == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        out = model(x, attention_mask=full_attention_mask(x.shape[1], device))
        logits = prepare_logits(out.logits)
        masked_logits = logits[0, masked_local + n_prefix]
        probs = torch.softmax(masked_logits, dim=-1)
        entropy_trajectory.append(
            {
                "step": step + 1,
                "mean_masked_entropy": float(
                    (-(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1).mean()).item()
                ),
                "n_masked": int(len(masked_local)),
            }
        )

        confidence, x0 = sample_tokens(
            masked_logits,
            temperature=temperature,
            neg_entropy=_neg_entropy(),
        )
        n_commit = min(k_per_step, len(confidence))
        if step == steps - 1:
            n_commit = len(confidence)
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]

        current_tau = step + 1
        if current_tau not in schedule:
            continue

        snap = snapshot_from_logits(
            tokenizer,
            logits,
            x,
            n_prefix,
            n_tokens,
            snapshot_mode=snapshot_mode,
        )
        updated_passages, new_passages, query = expand_with_snapshot(
            retriever,
            question,
            snap["snapshot_text"],
            current_passages,
            top_k=top_k,
        )
        current_passages = updated_passages
        new_context = "\n\n".join(current_passages)
        new_prefix_ids, new_n_prefix = build_short_prompt(tokenizer, new_context, question)
        rebuilt = torch.tensor(
            [new_prefix_ids + [mask_id] * n_tokens],
            dtype=torch.long,
            device=device,
        )
        rebuilt[0, new_n_prefix:new_n_prefix + n_tokens] = snap["answer_state"]
        x = rebuilt
        n_prefix = new_n_prefix

        interruptions.append(
            {
                "tau": current_tau,
                "snapshot_text": snap["snapshot_text"],
                "query": query,
                "n_new_passages": len(new_passages),
                "new_passages": new_passages,
                "masked_entropy": snap["masked_entropy"],
                "committed_tokens": snap["committed_tokens"],
                "n_total_passages": len(current_passages),
            }
        )

    final_answer = decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])
    return {
        "answer": final_answer,
        "snapshots": interruptions,
        "retrieved_passages": [item["new_passages"] for item in interruptions],
        "entropy_trajectory": entropy_trajectory,
        "n_interruptions": len(interruptions),
        "final_passages": current_passages,
    }


@torch.inference_mode()
def snapshot_pool_generate(
    model,
    tokenizer,
    retriever: Wiki18Retriever,
    *,
    initial_passages: list[str],
    question: str,
    steps: int,
    n_tokens: int,
    temperature: float,
    top_k: int,
    snapshot_mode: str,
) -> dict[str, Any]:
    """Endpoint baseline: decode fully, retrieve from final snapshot, regenerate."""
    base_context = "\n\n".join(initial_passages)
    first_pass = decode_with_trace(
        model,
        tokenizer,
        base_context,
        question,
        steps=steps,
        n_tokens=n_tokens,
        temperature=temperature,
    )
    snapshot_text = first_pass["answer"] if snapshot_mode == "full" else first_pass["answer"]
    expanded_passages, new_passages, query = expand_with_snapshot(
        retriever,
        question,
        snapshot_text,
        list(initial_passages),
        top_k=top_k,
    )
    second_context = "\n\n".join(expanded_passages)
    second_pass = decode_with_trace(
        model,
        tokenizer,
        second_context,
        question,
        steps=steps,
        n_tokens=n_tokens,
        temperature=temperature,
    )
    return {
        "answer": second_pass["answer"],
        "snapshots": [
            {
                "tau": steps,
                "snapshot_text": snapshot_text,
                "query": query,
                "n_new_passages": len(new_passages),
                "new_passages": new_passages,
                "committed_tokens": [],
                "n_total_passages": len(expanded_passages),
            }
        ],
        "retrieved_passages": [new_passages],
        "entropy_trajectory": {
            "phase1": first_pass["entropy_trajectory"],
            "phase2": second_pass["entropy_trajectory"],
        },
        "n_interruptions": 1,
        "initial_answer": first_pass["answer"],
        "final_passages": expanded_passages,
    }


@torch.inference_mode()
def dnmr_pool_generate(
    model,
    tokenizer,
    retriever: Wiki18Retriever,
    *,
    initial_passages: list[str],
    question: str,
    steps: int,
    n_tokens: int,
    temperature: float,
    top_k: int,
    n_candidates: int,
    extraction_steps: int,
) -> dict[str, Any]:
    """Existing single-round DNMR-style expansion baseline for direct comparison."""
    base_context = "\n\n".join(initial_passages)
    first_pass = decode_with_trace(
        model,
        tokenizer,
        base_context,
        question,
        steps=steps,
        n_tokens=n_tokens,
        temperature=temperature,
    )
    candidates = extract_candidates_generic(
        model,
        tokenizer,
        base_context,
        question,
        n_candidates=n_candidates,
        extraction_steps=extraction_steps,
    )
    expanded_passages, new_passages, queries = expand_with_dnmr_candidates(
        retriever,
        question,
        first_pass["answer"],
        candidates,
        list(initial_passages),
        top_k=top_k,
    )
    second_context = "\n\n".join(expanded_passages)
    second_pass = decode_with_trace(
        model,
        tokenizer,
        second_context,
        question,
        steps=steps,
        n_tokens=n_tokens,
        temperature=temperature,
    )
    return {
        "answer": second_pass["answer"],
        "snapshots": [
            {
                "tau": steps,
                "snapshot_text": first_pass["answer"],
                "queries": queries,
                "bridge_candidates": candidates,
                "n_new_passages": len(new_passages),
                "new_passages": new_passages,
                "n_total_passages": len(expanded_passages),
            }
        ],
        "retrieved_passages": [new_passages],
        "entropy_trajectory": {
            "phase1": first_pass["entropy_trajectory"],
            "phase2": second_pass["entropy_trajectory"],
        },
        "n_interruptions": 1,
        "initial_answer": first_pass["answer"],
        "final_passages": expanded_passages,
    }


def load_questions(dataset: str, start_idx: int, n_questions: int) -> list[dict[str, Any]]:
    all_questions = json.load(open(QUESTION_FILES[dataset], "r", encoding="utf-8"))
    return all_questions[start_idx:start_idx + n_questions]


def gold_answer(question: dict[str, Any]) -> str:
    return question.get("answer") or (question.get("golden_answers") or [""])[0]


def load_model_and_tokenizer(model_name: str):
    if model_name == "dream":
        ref = "Dream-org/Dream-v0-Instruct-7B"
        model_args = SimpleNamespace(model_name_or_path=ref)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    elif model_name == "llada":
        ref = "GSAI-ML/LLaDA-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(ref, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            ref,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = model_name
    return model, tokenizer


def run_schedule(
    spec: dict[str, Any],
    *,
    model,
    tokenizer,
    retriever: Wiki18Retriever,
    initial_passages: list[str],
    question: str,
    steps: int,
    n_tokens: int,
    temperature: float,
    interrupt_top_k: int,
    snapshot_mode: str,
    n_candidates: int,
    extraction_steps: int,
) -> dict[str, Any]:
    kind = spec["kind"]
    if kind == "baseline":
        payload = decode_with_trace(
            model,
            tokenizer,
            "\n\n".join(initial_passages),
            question,
            steps=steps,
            n_tokens=n_tokens,
            temperature=temperature,
        )
        return {
            "answer": payload["answer"],
            "snapshots": [],
            "retrieved_passages": [],
            "entropy_trajectory": payload["entropy_trajectory"],
            "n_interruptions": 0,
            "final_passages": list(initial_passages),
        }
    if kind == "pool":
        return snapshot_pool_generate(
            model,
            tokenizer,
            retriever,
            initial_passages=initial_passages,
            question=question,
            steps=steps,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=interrupt_top_k,
            snapshot_mode=snapshot_mode,
        )
    if kind == "dnmr_pool":
        return dnmr_pool_generate(
            model,
            tokenizer,
            retriever,
            initial_passages=initial_passages,
            question=question,
            steps=steps,
            n_tokens=n_tokens,
            temperature=temperature,
            top_k=interrupt_top_k,
            n_candidates=n_candidates,
            extraction_steps=extraction_steps,
        )
    return interrupted_generate(
        model,
        tokenizer,
        retriever,
        initial_passages=initial_passages,
        question=question,
        steps=steps,
        n_tokens=n_tokens,
        temperature=temperature,
        interrupt_steps=spec["interrupt_steps"],
        top_k=interrupt_top_k,
        snapshot_mode=snapshot_mode,
    )


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(rows: list[dict[str, Any]], schedule_specs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_rows": len(rows), "models": {}}
    schedule_names = [spec["name"] for spec in schedule_specs]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    for model_name, model_rows in grouped.items():
        model_summary = {"n_questions": len(model_rows), "methods": {}}
        for method_name in schedule_names:
            metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "em": 0.0, "contain": 0.0}
            latency = 0.0
            interruptions = 0.0
            for row in model_rows:
                payload = row["methods"][method_name]
                scores = payload["scores"]
                for key in metrics:
                    metrics[key] += scores[key]
                latency += payload["elapsed_sec"]
                interruptions += payload.get("n_interruptions", 0)

            n = max(1, len(model_rows))
            model_summary["methods"][method_name] = {
                **{key: value / n for key, value in metrics.items()},
                "avg_elapsed_sec": latency / n,
                "avg_interruptions": interruptions / n,
            }
        summary["models"][model_name] = model_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Interrupted denoising benchmark for multi-hop QA")
    parser.add_argument("--models", default="dream,llada")
    parser.add_argument("--dataset", default="musique", choices=list(QUESTION_FILES.keys()))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--interrupt_top_k", type=int, default=3)
    parser.add_argument("--snapshot_mode", choices=["full", "committed"], default="full")
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--schedules", default="baseline,4,8,12,16,24,32")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    schedule_specs = parse_schedule_spec(args.schedules, args.steps)
    questions = load_questions(args.dataset, args.start_idx, args.n_questions)
    retriever = Wiki18Retriever()

    output_path = Path(args.output)
    rows: list[dict[str, Any]] = []
    t0 = time.time()

    print(f"=== Interrupted Denoising Benchmark ===", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Questions: {len(questions)}", flush=True)
    print(f"Schedules: {[spec['name'] for spec in schedule_specs]}", flush=True)

    for model_name in parse_csv(args.models):
        model_t0 = time.time()
        print(f"\nLoading model={model_name}...", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_name)
        print(f"  loaded in {time.time() - model_t0:.1f}s", flush=True)

        query_batch = [f"query: {item['question']}" for item in questions]
        initial_results = retriever.retrieve_batch(query_batch, top_k=args.initial_top_k)

        for idx, (question, initial_passages) in enumerate(zip(questions, initial_results), start=1):
            q_t0 = time.time()
            row = {
                "id": question.get("qid") or question.get("id") or f"q_{args.start_idx + idx - 1}",
                "model": model_name,
                "dataset": args.dataset,
                "question": question["question"],
                "gold": gold_answer(question),
                "initial_passages": initial_passages,
                "methods": {},
            }

            for spec in schedule_specs:
                method_t0 = time.time()
                payload = run_schedule(
                    spec,
                    model=model,
                    tokenizer=tokenizer,
                    retriever=retriever,
                    initial_passages=initial_passages,
                    question=question["question"],
                    steps=args.steps,
                    n_tokens=args.answer_tokens,
                    temperature=args.temperature,
                    interrupt_top_k=args.interrupt_top_k,
                    snapshot_mode=args.snapshot_mode,
                    n_candidates=args.n_candidates,
                    extraction_steps=args.extraction_steps,
                )
                payload["scores"] = score_answer(payload["answer"], row["gold"])
                payload["elapsed_sec"] = round(time.time() - method_t0, 3)
                row["methods"][spec["name"]] = payload

            rows.append(row)

            if idx % args.log_every == 0 or idx == len(questions):
                print(
                    f"[{model_name}] {idx}/{len(questions)} done "
                    f"({time.time() - q_t0:.1f}s last, {time.time() - t0:.1f}s total)",
                    flush=True,
                )
            if len(rows) % args.save_every == 0:
                save_jsonl(output_path, rows)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_jsonl(output_path, rows)
    summary = build_summary(rows, schedule_specs)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved rows to {output_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)
    print(f"Finished in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
