"""LLaDA DNMR MuSiQue 50q pilot aligned to Dream behavior.

Compares:
  - baseline
  - pool_m6_8_hint2_yn
  - pool_v3_bridge
  - pool_v3_bridge_curated
  - pool_v3_full

Run:
python -u dnmr_pool_v2_lean.py --model llada --dataset musique --n_questions 50 \
  --output results/pool_v3/llada_musique_50q.json
"""
import argparse
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.environ.get("DLLM_PATH", "dllm"))
sys.path.insert(0, os.environ.get("DAES_PATH", "src/daes"))

import eamd_v2_wiki18
from eamd_v2_wiki18 import (
    QUESTION_FILES,
    Wiki18Retriever,
    _neg_entropy,
    build_short_prompt,
    decode_answer,
    eamd_regen_shared,
    extract_candidates_agnostic,
    extract_candidates_mixed_posterior,
    get_mask_id,
    prepare_logits,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoModel, AutoTokenizer


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def score(pred: str, gold: str) -> dict:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "em": 0.0, "contain": 0.0}
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "em": 0.0, "contain": 0.0}
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    em = float(normalize_answer(pred) == normalize_answer(gold))
    contain = float(gold.strip().lower() in pred.strip().lower())
    return {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "em": em,
        "contain": contain,
    }


def is_yesno_question(question: str) -> bool:
    q = question.strip().lower()
    return q.startswith(
        (
            "is ",
            "are ",
            "was ",
            "were ",
            "did ",
            "do ",
            "does ",
            "has ",
            "have ",
            "had ",
            "can ",
            "could ",
            "will ",
            "would ",
            "should ",
        )
    )


def is_date_question(question: str) -> bool:
    q = question.strip().lower()
    return (
        "what year" in q
        or "which year" in q
        or "what month" in q
        or "which month" in q
        or q.startswith("when ")
        or "what date" in q
        or "which date" in q
    )


def is_numeric_question(question: str) -> bool:
    q = question.strip().lower()
    return (
        q.startswith("how many ")
        or q.startswith("how much ")
        or "what number" in q
        or "which number" in q
        or "what percentage" in q
        or "how old" in q
    )


def choose_answer_budget(question: str) -> int:
    if is_yesno_question(question):
        return 2
    if is_date_question(question):
        return 4
    if is_numeric_question(question):
        return 3
    return 6


def build_hint_v2(bridge_cands) -> str:
    entities = []
    seen = set()
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        text = re.sub(r"^(?:The answer is:?\s*|The\s+)", "", text, flags=re.IGNORECASE).strip()
        text = text.split("\n")[0].strip().rstrip(".,;:")
        words = text.split()
        if len(words) > 5:
            text = " ".join(words[:5])
        if text and len(text) > 1 and text.lower() not in seen:
            seen.add(text.lower())
            entities.append(text)
    if not entities:
        return ""
    return "Related entities: " + ", ".join(entities[:3]) + "."


def candidate_template_rate(cands) -> float:
    if not cands:
        return 0.0
    templated = 0
    for cand in cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if re.match(r"^\s*(?:the answer is|answer:|short answer:)", text, flags=re.IGNORECASE):
            templated += 1
    return templated / max(1, len(cands))


def candidate_mean_words(cands) -> float:
    if not cands:
        return 0.0
    return sum(len((cand.get("text", "") if isinstance(cand, dict) else str(cand)).split()) for cand in cands) / len(cands)


def answer_word_len(text: str) -> int:
    return len(text.strip().split()) if text.strip() else 0


def contains_gold_in_passages(passages, gold: str) -> bool:
    gold_norm = normalize_answer(gold)
    if not gold_norm:
        return False
    joined = " ".join(normalize_answer(passage) for passage in passages)
    return gold_norm in joined


def classify_failure(question: str, gold: str, pred: str, passages) -> str:
    if contains_gold_in_passages(passages, gold):
        return "answer_stage_failure"
    q = question.lower()
    if any(token in q for token in ("district", "county", "province", "borough", "region", "city", "town", "state", "country", "year", "month", "date")):
        return "wrong_granularity"
    return "nearby_entity_drift"


@torch.inference_mode()
def simple_decode(model, tokenizer, context, question, steps=32, n_tokens=32):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)
    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, masked_local + n_prefix]
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


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


def build_query_specs(question: str, seed_answer: str, bridge_cands, seed_weight: float):
    specs = [{"query": f"query: {question} {seed_answer}", "weight": max(0.0, seed_weight), "kind": "seed"}]
    bridge_mass = sum(max(float(cand.get("init_conf", 0.0)), 0.0) for cand in bridge_cands)
    bridge_budget = max(0.0, 1.0 - seed_weight)
    if bridge_budget <= 0 or bridge_mass <= 0:
        return [{"query": specs[0]["query"], "weight": 1.0, "kind": "seed"}]
    for cand in bridge_cands:
        text = cand.get("text", "").strip()
        mass = max(float(cand.get("init_conf", 0.0)), 0.0)
        if not text or mass <= 0:
            continue
        specs.append(
            {
                "query": f"query: {question} {text}",
                "weight": bridge_budget * mass / bridge_mass,
                "kind": "bridge",
                "bridge_text": text,
            }
        )
    total_weight = sum(spec["weight"] for spec in specs)
    if total_weight <= 0:
        return [{"query": f"query: {question} {seed_answer}", "weight": 1.0, "kind": "seed"}]
    for spec in specs:
        spec["weight"] /= total_weight
    return specs


def expand_evidence_curated(
    retriever,
    question,
    seed_answer,
    bridge_cands,
    current_passages,
    expand_top_k=3,
    budget=8,
    seed_weight=0.35,
):
    query_specs = build_query_specs(question, seed_answer, bridge_cands, seed_weight)
    scored_results = retriever.retrieve_batch_with_scores([spec["query"] for spec in query_specs], expand_top_k)

    existing = set(current_passages)
    candidate_map = {}
    for spec, result in zip(query_specs, scored_results):
        for hit in result["hits"]:
            passage = hit["passage"]
            if passage in existing:
                continue
            utility = spec["weight"] * max(float(hit["score"]), 0.0) / (hit["rank"] + 1)
            if passage not in candidate_map:
                candidate_map[passage] = {
                    "passage": passage,
                    "utility": 0.0,
                    "support_count": 0,
                }
            candidate_map[passage]["utility"] += utility
            candidate_map[passage]["support_count"] += 1

    ranked = sorted(
        candidate_map.values(),
        key=lambda item: (item["utility"], item["support_count"]),
        reverse=True,
    )
    selected = ranked[:budget]
    ordered = sorted(selected, key=lambda item: item["utility"])
    new_passages = [item["passage"] for item in ordered]
    meta = {
        "budget": budget,
        "candidate_pool_size": len(ranked),
        "selected_utilities": [round(item["utility"], 6) for item in selected],
        "query_specs": [
            {
                "kind": spec["kind"],
                "weight": round(spec["weight"], 6),
                "query": spec["query"],
            }
            for spec in query_specs
        ],
    }
    return list(current_passages) + new_passages, new_passages, meta


def init_totals(methods):
    return {
        method: {"f1": 0.0, "precision": 0.0, "recall": 0.0, "em": 0.0, "contain": 0.0}
        for method in methods
    }


def init_diagnostics(methods):
    return {
        method: {
            "questions": 0,
            "answer_words": 0.0,
            "candidate_words": 0.0,
            "candidate_sets": 0,
            "template_rate_sum": 0.0,
            "new_passages": 0.0,
            "total_passages": 0.0,
            "curated_budget_used": 0.0,
            "failure_buckets": Counter(),
        }
        for method in methods
    }


def update_diag(diag_entry, answer, cands, passages, new_passages, curated_budget_used, failure_bucket):
    diag_entry["questions"] += 1
    diag_entry["answer_words"] += answer_word_len(answer)
    if cands is not None:
        diag_entry["candidate_sets"] += 1
        diag_entry["candidate_words"] += candidate_mean_words(cands)
        diag_entry["template_rate_sum"] += candidate_template_rate(cands)
    if passages is not None:
        diag_entry["new_passages"] += float(new_passages)
        diag_entry["total_passages"] += float(len(passages))
        diag_entry["curated_budget_used"] += float(curated_budget_used)
    if failure_bucket:
        diag_entry["failure_buckets"][failure_bucket] += 1


def summarize_totals(totals, n_done):
    return {
        method: {metric: round(value / max(1, n_done), 4) for metric, value in metrics.items()}
        for method, metrics in totals.items()
    }


def summarize_diagnostics(diag):
    out = {}
    for method, stats in diag.items():
        n = max(1, stats["questions"])
        c = max(1, stats["candidate_sets"])
        out[method] = {
            "avg_answer_words": round(stats["answer_words"] / n, 3),
            "avg_candidate_words": round(stats["candidate_words"] / c, 3) if stats["candidate_sets"] else 0.0,
            "template_candidate_rate": round(stats["template_rate_sum"] / c, 4) if stats["candidate_sets"] else 0.0,
            "avg_new_passages": round(stats["new_passages"] / n, 3),
            "avg_total_passages": round(stats["total_passages"] / n, 3),
            "avg_curated_budget_used": round(stats["curated_budget_used"] / n, 3),
            "failure_buckets": dict(stats["failure_buckets"]),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--extraction_steps", type=int, default=12)
    parser.add_argument("--bridge_n_mask", type=int, default=10)
    parser.add_argument("--bridge_steps", type=int, default=16)
    parser.add_argument("--bridge_n_branch", type=int, default=3)
    parser.add_argument("--bridge_n_candidates", type=int, default=4)
    parser.add_argument("--curated_budget", type=int, default=8)
    parser.add_argument("--seed_weight", type=float, default=0.35)
    parser.add_argument("--gamma_cap", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print("=== DNMR Pool v3 MuSiQue Pilot ===", flush=True)
    print(f"Model: {args.model}, Dataset: {args.dataset}, N: {args.n_questions}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available on this node. "
            "Run this on a GPU node, e.g. "
            "`srun --partition=gpu_a100 --gpus=1 --time=00:30:00 --pty bash`, "
            "then activate the venv and rerun."
        )
    if not args.device.startswith("cuda"):
        raise SystemExit("This pilot is intended for GPU execution only. Use --device cuda:0 on a GPU node.")

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(args.device).eval()

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever(device=args.device)
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    print("Batch initial retrieval...", flush=True)
    query_texts = [f"query: {question['question']}" for question in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"Done in {time.time() - t_start:.1f}s", flush=True)

    methods = [
        "baseline",
        "pool_m6_8_hint2_yn",
        "pool_v3_bridge",
        "pool_v3_bridge_curated",
        "pool_v3_full",
    ]
    totals = init_totals(methods)
    diagnostics = init_diagnostics(methods)
    results = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, question in enumerate(questions):
        tq = time.time()
        qtext = question["question"]
        gold = question.get("answer") or (question.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)
        yn = is_yesno_question(qtext)
        v3_tokens = choose_answer_budget(qtext)

        seed_ans = simple_decode(model, tokenizer, old_ctx, qtext, steps=args.steps, n_tokens=32)

        current_cands = extract_candidates_agnostic(
            model,
            tokenizer,
            old_ctx,
            qtext,
            args.n_candidates,
            n_mask=6,
            extraction_steps=args.extraction_steps,
        )
        current_passages, current_new = expand_evidence(
            retriever,
            qtext,
            seed_ans,
            current_cands,
            initial,
            args.expand_top_k,
        )
        current_hint = build_hint_v2(current_cands)
        current_ctx = "\n\n".join(current_passages)
        current_hint_ctx = f"{current_hint}\n\n{current_ctx}" if current_hint else current_ctx
        current_ans = simple_decode(
            model,
            tokenizer,
            current_hint_ctx,
            qtext,
            steps=args.steps,
            n_tokens=2 if yn else 8,
        )

        v3_cands = extract_candidates_mixed_posterior(
            model,
            tokenizer,
            old_ctx,
            qtext,
            n_candidates=args.bridge_n_candidates,
            n_branch=args.bridge_n_branch,
            n_mask=args.bridge_n_mask,
            extraction_steps=args.bridge_steps,
        )
        v3_hint = build_hint_v2(v3_cands)

        v3_passages, v3_new = expand_evidence(
            retriever,
            qtext,
            seed_ans,
            v3_cands,
            initial,
            args.expand_top_k,
        )
        v3_ctx = "\n\n".join(v3_passages)
        v3_hint_ctx = f"{v3_hint}\n\n{v3_ctx}" if v3_hint else v3_ctx
        v3_bridge_ans = simple_decode(
            model,
            tokenizer,
            v3_hint_ctx,
            qtext,
            steps=args.steps,
            n_tokens=v3_tokens,
        )

        curated_passages, curated_new, curated_meta = expand_evidence_curated(
            retriever,
            qtext,
            seed_ans,
            v3_cands,
            initial,
            expand_top_k=args.expand_top_k,
            budget=args.curated_budget,
            seed_weight=args.seed_weight,
        )
        curated_ctx = "\n\n".join(curated_passages)
        curated_hint_ctx = f"{v3_hint}\n\n{curated_ctx}" if v3_hint else curated_ctx
        v3_bridge_curated_ans = simple_decode(
            model,
            tokenizer,
            curated_hint_ctx,
            qtext,
            steps=args.steps,
            n_tokens=v3_tokens,
        )
        v3_full_ans, v3_full_stats = eamd_regen_shared(
            model,
            tokenizer,
            qtext,
            old_ctx,
            curated_hint_ctx,
            steps=args.steps,
            n_tokens=v3_tokens,
            temperature=0.0,
            gamma_cap=args.gamma_cap,
        )

        answers = {
            "baseline": seed_ans,
            "pool_m6_8_hint2_yn": current_ans,
            "pool_v3_bridge": v3_bridge_ans,
            "pool_v3_bridge_curated": v3_bridge_curated_ans,
            "pool_v3_full": v3_full_ans,
        }

        method_context = {
            "baseline": {"cands": None, "passages": initial, "new_passages": 0, "curated_budget_used": 0},
            "pool_m6_8_hint2_yn": {"cands": current_cands, "passages": current_passages, "new_passages": len(current_new), "curated_budget_used": 0},
            "pool_v3_bridge": {"cands": v3_cands, "passages": v3_passages, "new_passages": len(v3_new), "curated_budget_used": 0},
            "pool_v3_bridge_curated": {"cands": v3_cands, "passages": curated_passages, "new_passages": len(curated_new), "curated_budget_used": len(curated_new)},
            "pool_v3_full": {"cands": v3_cands, "passages": curated_passages, "new_passages": len(curated_new), "curated_budget_used": len(curated_new)},
        }

        row = {
            "id": question.get("qid") or question.get("id", f"dev_{qi}"),
            "question": qtext,
            "gold": gold,
            "elapsed": round(time.time() - tq, 2),
            "is_yesno": yn,
            "typed_budget": v3_tokens,
            "current_candidates": [cand.get("text", "")[:60] for cand in current_cands],
            "v3_candidates": [cand.get("text", "")[:60] for cand in v3_cands],
            "current_hint": current_hint,
            "v3_hint": v3_hint,
            "curated_meta": curated_meta,
        }

        for method, answer in answers.items():
            metrics = score(answer, gold)
            metrics["answer"] = answer
            failure_bucket = None
            if metrics["em"] < 1.0:
                failure_bucket = classify_failure(
                    qtext,
                    gold,
                    answer,
                    method_context[method]["passages"],
                )
                metrics["failure_bucket"] = failure_bucket
            row[method] = metrics
            if method == "pool_v3_full":
                row[method]["stats"] = {k: round(v, 6) for k, v in v3_full_stats.items()}
            for metric_name in totals[method]:
                totals[method][metric_name] += metrics[metric_name]
            update_diag(
                diagnostics[method],
                answer,
                method_context[method]["cands"],
                method_context[method]["passages"],
                method_context[method]["new_passages"],
                method_context[method]["curated_budget_used"],
                failure_bucket,
            )

        results.append(row)

        if (qi + 1) % args.log_every == 0 or qi == 0:
            n_done = len(results)
            base_f1 = totals["baseline"]["f1"] / n_done
            cur_f1 = totals["pool_m6_8_hint2_yn"]["f1"] / n_done
            v3_f1 = totals["pool_v3_full"]["f1"] / n_done
            cur_contain = totals["pool_m6_8_hint2_yn"]["contain"] / n_done
            v3_contain = totals["pool_v3_full"]["contain"] / n_done
            print(
                f"[{qi+1}/{len(questions)}] {row['elapsed']:.1f}s | "
                f"base F1={base_f1:.3f} | v2 F1={cur_f1:.3f} C={cur_contain:.1%} | "
                f"v3_full F1={v3_f1:.3f} C={v3_contain:.1%}",
                flush=True,
            )

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "summary": summarize_totals(totals, n_done),
                        "diagnostics": summarize_diagnostics(diagnostics),
                        "results": results,
                        "config": vars(args),
                        "timing": {"elapsed_sec": round(time.time() - t_start, 1)},
                    },
                    f,
                    indent=2,
                )

    n = len(results)
    summary = summarize_totals(totals, n)
    diagnostics_summary = summarize_diagnostics(diagnostics)
    print(f"\n{'Method':<24s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Cont':>8s}")
    print("-" * 60)
    for method in methods:
        stats = summary[method]
        print(
            f"{method:<24s} {stats['f1']:>6.3f} {stats['precision']:>6.3f} "
            f"{stats['recall']:>6.3f} {stats['contain']:>8.3f}",
            flush=True,
        )
    print("\nDiagnostics", flush=True)
    for method in methods:
        print(
            f"{method:<24s} "
            f"cand_w={diagnostics_summary[method]['avg_candidate_words']:.2f} "
            f"templ={diagnostics_summary[method]['template_candidate_rate']:.1%} "
            f"ans_w={diagnostics_summary[method]['avg_answer_words']:.2f}",
            flush=True,
        )
    print(f"\nTotal: {time.time() - t_start:.1f}s ({(time.time() - t_start) / max(1, n):.1f}s/q)", flush=True)


if __name__ == "__main__":
    main()
