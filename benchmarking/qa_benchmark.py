from __future__ import annotations

import csv
import json
import re
import string
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any

CORPUS_NAME = "wiki18_100w"

SOURCE_DATASET_FILES = {
    "hotpotqa": "/projects/prjs1800/external/arag/data/hotpotqa/questions.json",
    "musique": "/projects/prjs1800/external/arag/data/musique/questions.json",
    "2wikimultihopqa": "/projects/prjs1800/external/arag/data/2wikimultihop/questions.json",
    "2wikimultihop": "/projects/prjs1800/external/arag/data/2wikimultihop/questions.json",
}

QUESTION_FILE_HINTS = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wikimultihopqa": "2wikimultihopqa",
    "2wikimultihop": "2wikimultihop",
}


def canonical_dataset_name(name: str | None) -> str:
    raw = (name or "").lower()
    if raw in {"2wiki", "2wikimultihop", "2wikimultihopqa"}:
        return "2wikimultihopqa"
    if raw in {"hotpot", "hotpotqa"}:
        return "hotpotqa"
    if raw in {"musique"}:
        return "musique"
    return raw


def infer_dataset_name(path_hint: str | None) -> str:
    hint = (path_hint or "").lower()
    for dataset, token in QUESTION_FILE_HINTS.items():
        if token in hint:
            return canonical_dataset_name(dataset)
    return ""


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_f1(pred: str, gold: str) -> tuple[float, float, float]:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return precision, recall, 2 * precision * recall / (precision + recall)


def evaluate_prediction(pred: str, gold_answers: list[str]) -> dict[str, Any]:
    answers = gold_answers or [""]
    best = {
        "pred_answer": pred,
        "pred_answer_norm": normalize_answer(pred),
        "gold_answer": answers[0],
        "gold_answers": answers,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "em": 0.0,
        "contain": 0.0,
    }
    for gold in answers:
        precision, recall, f1 = compute_f1(pred, gold)
        em = compute_em(pred, gold)
        contain = float(normalize_answer(gold) in normalize_answer(pred))
        score = (f1, em, contain, precision, recall)
        best_score = (best["f1"], best["em"], best["contain"], best["precision"], best["recall"])
        if score > best_score:
            best.update(
                {
                    "gold_answer": gold,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "em": em,
                    "contain": contain,
                }
            )
    return best


def answer_token_count(text: str) -> int:
    norm = normalize_answer(text)
    return len(norm.split()) if norm else 0


def extract_title_from_passage(passage: str) -> str:
    text = (passage or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip().strip('"')
    return first_line


def _normalize_question(text: str) -> str:
    return " ".join((text or "").strip().split())


def _support_labels_from_item(dataset: str, item: dict[str, Any]) -> list[str]:
    dataset = canonical_dataset_name(dataset)
    evidence = item.get("evidence")
    if dataset == "hotpotqa" and isinstance(evidence, list):
        return sorted({str(entry[0]).strip() for entry in evidence if isinstance(entry, list) and entry})
    if dataset == "2wikimultihopqa" and isinstance(evidence, list):
        labels: set[str] = set()
        for entry in evidence:
            if isinstance(entry, list) and len(entry) >= 3:
                labels.add(str(entry[0]).strip())
                labels.add(str(entry[2]).strip())
        return sorted(labels)
    return []


@lru_cache(maxsize=None)
def load_support_map(dataset: str) -> dict[str, dict[str, Any]]:
    dataset = canonical_dataset_name(dataset)
    source = SOURCE_DATASET_FILES.get(dataset)
    if not source or not Path(source).exists():
        return {}
    rows = json.loads(Path(source).read_text(encoding="utf-8"))
    mapping: dict[str, dict[str, Any]] = {}
    for item in rows:
        question = _normalize_question(item.get("question", ""))
        if not question:
            continue
        mapping[question] = {
            "source_id": item.get("id"),
            "answer": item.get("answer", ""),
            "support_labels": _support_labels_from_item(dataset, item),
        }
    return mapping


def support_info_for_question(dataset: str, question: str) -> dict[str, Any]:
    return load_support_map(dataset).get(_normalize_question(question), {})


def answer_hit_in_passages(passages: list[str], gold_answers: list[str]) -> float:
    normalized_passages = [normalize_answer(p) for p in passages if p]
    if not normalized_passages:
        return 0.0
    for answer in gold_answers:
        norm_answer = normalize_answer(answer)
        if norm_answer and any(norm_answer in passage for passage in normalized_passages):
            return 1.0
    return 0.0


def support_hit_in_passages(passages: list[str], support_labels: list[str] | None) -> float | None:
    labels = [label for label in (support_labels or []) if label]
    if not labels:
        return None
    titles = {normalize_answer(extract_title_from_passage(p)) for p in passages if p}
    label_norm = {normalize_answer(label) for label in labels}
    if not label_norm:
        return None
    return float(label_norm.issubset(titles))


def build_record(
    *,
    dataset: str,
    qid: str,
    method: str,
    model: str,
    question: str,
    gold_answers: list[str],
    pred_answer: str,
    elapsed_sec_total: float,
    elapsed_sec_llm: float,
    elapsed_sec_retrieval: float,
    retrieval_calls: int,
    unique_chunks_read: int,
    total_retrieved_tokens: int,
    loops_or_rounds: int,
    llm_calls: int | None = None,
    forward_passes: int | None = None,
    denoising_steps: int | None = None,
    c0_passages: list[str] | None = None,
    final_passages: list[str] | None = None,
    corpus: str = CORPUS_NAME,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = canonical_dataset_name(dataset)
    c0_passages = c0_passages or []
    final_passages = final_passages or []
    metrics = evaluate_prediction(pred_answer, gold_answers)
    support_info = support_info_for_question(dataset, question)
    support_labels = support_info.get("support_labels") or []
    c0_support_hit = support_hit_in_passages(c0_passages, support_labels)
    final_support_hit = support_hit_in_passages(final_passages, support_labels)
    elapsed_other = max(0.0, elapsed_sec_total - elapsed_sec_llm - elapsed_sec_retrieval)

    record = {
        "dataset": dataset,
        "qid": str(qid),
        "method": method,
        "model": model,
        "corpus": corpus,
        "question": question,
        "gold_answer": metrics["gold_answer"],
        "gold_answers": gold_answers,
        "pred_answer": pred_answer,
        "pred_answer_norm": metrics["pred_answer_norm"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "em": metrics["em"],
        "contain": metrics["contain"],
        "elapsed_sec_total": round(float(elapsed_sec_total), 6),
        "elapsed_sec_llm": round(float(elapsed_sec_llm), 6),
        "elapsed_sec_retrieval": round(float(elapsed_sec_retrieval), 6),
        "elapsed_sec_other": round(float(elapsed_other), 6),
        "retrieval_calls": int(retrieval_calls),
        "unique_chunks_read": int(unique_chunks_read),
        "total_retrieved_tokens": int(total_retrieved_tokens),
        "loops_or_rounds": int(loops_or_rounds),
        "llm_calls": None if llm_calls is None else int(llm_calls),
        "forward_passes": None if forward_passes is None else int(forward_passes),
        "denoising_steps": None if denoising_steps is None else int(denoising_steps),
        "answer_tokens_out": answer_token_count(pred_answer),
        "c0_answer_hit": answer_hit_in_passages(c0_passages, gold_answers),
        "final_answer_hit": answer_hit_in_passages(final_passages, gold_answers),
        "c0_support_hit": c0_support_hit,
        "final_support_hit": final_support_hit,
        "error": error,
    }
    if extra:
        record.update(extra)
    return record


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _is_answer_miss(record: dict[str, Any]) -> bool:
    return float(record.get("c0_answer_hit", 0.0)) < 1.0


def _is_support_miss(record: dict[str, Any]) -> bool:
    value = record.get("c0_support_hit")
    return value is not None and float(value) < 1.0


def _is_c0_miss(record: dict[str, Any]) -> bool:
    return _is_answer_miss(record) or _is_support_miss(record)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["method"])].append(record)
        by_dataset[record["dataset"]].append(record)

    summary_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    hard_subset_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    latency_bucket_rows: list[dict[str, Any]] = []

    for (dataset, method), items in sorted(grouped.items()):
        elapsed = [float(item["elapsed_sec_total"]) for item in items]
        f1s = [float(item["f1"]) for item in items]
        ems = [float(item["em"]) for item in items]
        contains = [float(item["contain"]) for item in items]
        retrieval_calls = [int(item["retrieval_calls"]) for item in items]
        retrieved_tokens = [int(item["total_retrieved_tokens"]) for item in items]
        unique_chunks = [int(item["unique_chunks_read"]) for item in items]
        mean_f1 = mean(f1s)
        mean_calls = mean(retrieval_calls)
        mean_tokens = mean(retrieved_tokens)
        mean_chunks = mean(unique_chunks)

        summary_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(items),
                "f1": round(mean_f1, 6),
                "em": round(mean(ems), 6),
                "contain": round(mean(contains), 6),
                "mean_latency_sec": round(mean(elapsed), 6),
                "mean_retrieval_calls": round(mean_calls, 6),
                "mean_retrieved_tokens": round(mean_tokens, 6),
                "mean_unique_chunks_read": round(mean_chunks, 6),
            }
        )
        latency_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(items),
                "mean_latency_sec": round(mean(elapsed), 6),
                "median_latency_sec": round(_percentile(elapsed, 0.5), 6),
                "p90_latency_sec": round(_percentile(elapsed, 0.9), 6),
                "p95_latency_sec": round(_percentile(elapsed, 0.95), 6),
            }
        )
        frontier_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "f1": round(mean_f1, 6),
                "em": round(mean(ems), 6),
                "contain": round(mean(contains), 6),
                "median_latency_sec": round(_percentile(elapsed, 0.5), 6),
                "p90_latency_sec": round(_percentile(elapsed, 0.9), 6),
                "queries_per_sec": round(len(items) / max(sum(elapsed), 1e-9), 6),
                "mean_retrieval_calls": round(mean_calls, 6),
                "mean_retrieved_tokens": round(mean_tokens, 6),
            }
        )
        retrieval_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n": len(items),
                "f1": round(mean_f1, 6),
                "mean_retrieval_calls": round(mean_calls, 6),
                "mean_retrieved_tokens": round(mean_tokens, 6),
                "mean_unique_chunks_read": round(mean_chunks, 6),
                "f1_per_retrieval_call": round(mean_f1 / max(mean_calls, 1e-9), 6),
                "f1_per_1k_retrieved_tokens": round(mean_f1 / max(mean_tokens / 1000.0, 1e-9), 6),
            }
        )

        subsets = {
            "all": items,
            "c0_miss": [item for item in items if _is_c0_miss(item)],
            "final_hit_after_c0_miss": [
                item for item in items
                if _is_c0_miss(item) and float(item.get("final_answer_hit", 0.0)) >= 1.0
            ],
            "multi_hop_hard": [item for item in items if _is_support_miss(item)],
            "support_recovered": [
                item
                for item in items
                if _is_support_miss(item) and item.get("final_support_hit") is not None and float(item["final_support_hit"]) >= 1.0
            ],
            "partial_only": [
                item for item in items if float(item["f1"]) > 0.0 and float(item["em"]) == 0.0
            ],
        }
        for subset_name, subset_items in subsets.items():
            if not subset_items:
                continue
            hard_subset_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "subset": subset_name,
                    "n": len(subset_items),
                    "f1": round(mean([float(item["f1"]) for item in subset_items]), 6),
                    "em": round(mean([float(item["em"]) for item in subset_items]), 6),
                    "contain": round(mean([float(item["contain"]) for item in subset_items]), 6),
                    "mean_latency_sec": round(mean([float(item["elapsed_sec_total"]) for item in subset_items]), 6),
                }
            )

    for dataset, dataset_items in sorted(by_dataset.items()):
        dataset_latencies = [float(item["elapsed_sec_total"]) for item in dataset_items]
        fast_cutoff = _percentile(dataset_latencies, 1 / 3)
        slow_cutoff = _percentile(dataset_latencies, 2 / 3)
        for (group_dataset, method), items in sorted(grouped.items()):
            if group_dataset != dataset:
                continue
            buckets = {
                "fast": [item for item in items if float(item["elapsed_sec_total"]) <= fast_cutoff],
                "medium": [
                    item
                    for item in items
                    if fast_cutoff < float(item["elapsed_sec_total"]) <= slow_cutoff
                ],
                "slow": [item for item in items if float(item["elapsed_sec_total"]) > slow_cutoff],
            }
            for bucket_name, bucket_items in buckets.items():
                if not bucket_items:
                    continue
                latency_bucket_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "latency_bucket": bucket_name,
                        "bucket_max_sec": round(
                            fast_cutoff if bucket_name == "fast" else slow_cutoff if bucket_name == "medium" else max(dataset_latencies),
                            6,
                        ),
                        "n": len(bucket_items),
                        "f1": round(mean([float(item["f1"]) for item in bucket_items]), 6),
                        "em": round(mean([float(item["em"]) for item in bucket_items]), 6),
                        "contain": round(mean([float(item["contain"]) for item in bucket_items]), 6),
                        "mean_latency_sec": round(mean([float(item["elapsed_sec_total"]) for item in bucket_items]), 6),
                    }
                )

    return {
        "summary_metrics": summary_rows,
        "latency_percentiles": latency_rows,
        "frontier_points": frontier_rows,
        "hard_subset_metrics": hard_subset_rows,
        "retrieval_efficiency": retrieval_rows,
        "latency_bucket_metrics": latency_bucket_rows,
    }


def write_dataset_artifacts(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_example.jsonl", records)
    summaries = summarize_records(records)
    for name, rows in summaries.items():
        _write_csv(output_dir / f"{name}.csv", rows)


def build_pairwise_delta_rows(
    records: list[dict[str, Any]],
    method: str,
    baseline_method: str,
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        indexed[(record["dataset"], record["qid"], record["method"])] = record

    rows: list[dict[str, Any]] = []
    keys = {(dataset, qid) for dataset, qid, meth in indexed if meth == method}
    for dataset, qid in sorted(keys):
        left = indexed.get((dataset, qid, method))
        right = indexed.get((dataset, qid, baseline_method))
        if not left or not right:
            continue
        rows.append(
            {
                "dataset": dataset,
                "qid": qid,
                "method": method,
                "baseline_method": baseline_method,
                "delta_f1": round(float(left["f1"]) - float(right["f1"]), 6),
                "delta_em": round(float(left["em"]) - float(right["em"]), 6),
                "delta_contain": round(float(left["contain"]) - float(right["contain"]), 6),
                "delta_latency_sec": round(float(left["elapsed_sec_total"]) - float(right["elapsed_sec_total"]), 6),
            }
        )
    return rows


def write_pairwise_artifacts(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    comparisons: list[tuple[str, str]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, baseline in comparisons:
        rows = build_pairwise_delta_rows(records, method, baseline)
        if rows:
            _write_csv(output_dir / f"delta_{method}_vs_{baseline}.csv", rows)
