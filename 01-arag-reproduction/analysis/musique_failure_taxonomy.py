#!/usr/bin/env python3
"""MuSiQue failure taxonomy for A-RAG reproduction experiments.

Categories (failed examples only):
1. never_searched_hop2
2. searched_but_missed
3. retrieved_but_couldnt_synthesize
4. decomposition_failure
5. corpus_gap
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

RE_CHUNK_ID = re.compile(r"Chunk ID:\s*(\d+)")

EXPERIMENTS: dict[str, str] = {
    "E1": "qwen25-7b-instruct",
    "E2": "qwen3-8b-vllm",
    "E3": "qwen3-8b-qwen-emb-vllm",
    "E4": "qwen3-30b-e5-deepseekr1",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "who",
    "what",
    "when",
    "where",
    "which",
    "whom",
    "with",
}


def normalize_text(text: str) -> str:
    """Normalize text for lexical matching."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return " ".join(text.split())


def content_tokens(text: str) -> set[str]:
    """Tokenize text, dropping stop words and very short tokens."""
    toks = normalize_text(text).split()
    return {tok for tok in toks if len(tok) > 2 and tok not in STOPWORDS}


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Load jsonl predictions."""
    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def strip_musique_prefix(qid: str) -> str:
    """Convert ARAG qid to MuSiQue dataset id."""
    if qid.startswith("musique_"):
        return qid[len("musique_") :]
    return qid


def load_musique_gold_map(
    musique_data_root: Path,
    needed_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Load only needed MuSiQue gold records from official files."""
    gold_map: dict[str, dict[str, Any]] = {}
    files = [
        musique_data_root / "musique_ans_v1.0_dev.jsonl",
        musique_data_root / "musique_ans_v1.0_test.jsonl",
        musique_data_root / "musique_ans_v1.0_train.jsonl",
    ]
    remaining = set(needed_ids)

    for path in files:
        if not path.exists() or not remaining:
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                qid = item.get("id", "")
                if qid in remaining:
                    gold_map[qid] = item
                    remaining.remove(qid)
        LOGGER.info("Scanned %s; matched so far: %d", path.name, len(gold_map))

    if remaining:
        LOGGER.warning("Missing %d MuSiQue ids in gold map", len(remaining))
    return gold_map


def parse_chunk_texts(chunks_file: Path) -> dict[str, str]:
    """Load chunk text map keyed by chunk id string."""
    raw_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    chunk_map: dict[str, str] = {}
    for idx, raw in enumerate(raw_chunks):
        if isinstance(raw, str):
            if ":" in raw:
                maybe_id, text = raw.split(":", 1)
                maybe_id = maybe_id.strip()
                if maybe_id.isdigit():
                    chunk_map[maybe_id] = text.strip()
                    continue
            chunk_map[str(idx)] = raw
        elif isinstance(raw, dict):
            cid = str(raw.get("id", idx))
            chunk_map[cid] = str(raw.get("text", ""))
        else:
            chunk_map[str(idx)] = str(raw)
    return chunk_map


def collect_retrieved_chunk_ids(pred: dict[str, Any]) -> list[str]:
    """Collect retrieved chunk ids from retrieval_logs + trajectory tool output."""
    ids: list[str] = []

    for log in pred.get("retrieval_logs", []):
        md = log.get("metadata", {})
        for cid in md.get("chunk_ids", []):
            ids.append(str(cid))

    for step in pred.get("trajectory", []):
        result = step.get("tool_result", "")
        if isinstance(result, str):
            ids.extend(RE_CHUNK_ID.findall(result))

    seen: set[str] = set()
    ordered: list[str] = []
    for cid in ids:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)
    return ordered


def collect_query_texts(pred: dict[str, Any]) -> list[str]:
    """Collect query-like strings from retrieval logs and trajectory args."""
    out: list[str] = []

    for log in pred.get("retrieval_logs", []):
        md = log.get("metadata", {})
        if isinstance(md.get("query"), str) and md["query"].strip():
            out.append(md["query"].strip())
        kws = md.get("keywords", [])
        if isinstance(kws, list) and kws:
            out.append(" ".join(str(k) for k in kws))

    for step in pred.get("trajectory", []):
        args = step.get("arguments", {})
        if isinstance(args, dict):
            if isinstance(args.get("query"), str) and args["query"].strip():
                out.append(args["query"].strip())
            kws = args.get("keywords", [])
            if isinstance(kws, list) and kws:
                out.append(" ".join(str(k) for k in kws))

    dedup: list[str] = []
    seen: set[str] = set()
    for q in out:
        if q not in seen:
            seen.add(q)
            dedup.append(q)
    return dedup


def support_paragraphs(gold_item: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract ordered supporting paragraphs from question decomposition."""
    paragraphs = {p.get("idx"): p for p in gold_item.get("paragraphs", [])}
    supports: list[dict[str, Any]] = []
    for step in gold_item.get("question_decomposition", []):
        idx = step.get("paragraph_support_idx")
        if idx in paragraphs:
            supports.append(paragraphs[idx])
    return supports


def paragraph_is_retrieved(
    paragraph_text: str,
    retrieved_chunk_ids: list[str],
    chunk_token_sets: dict[str, set[str]],
    threshold: float,
) -> bool:
    """Heuristic support retrieval check based on token-overlap recall."""
    p_tokens = content_tokens(paragraph_text)
    if not p_tokens:
        return False
    for cid in retrieved_chunk_ids:
        c_tokens = chunk_token_sets.get(cid, set())
        if not c_tokens:
            continue
        overlap = len(p_tokens & c_tokens) / len(p_tokens)
        if overlap >= threshold:
            return True
    return False


def question_overlap_with_decomp(query_texts: list[str], gold_item: dict[str, Any]) -> float:
    """Max overlap of any query with decomposition questions."""
    decomp_q_tokens = [content_tokens(step.get("question", "")) for step in gold_item.get("question_decomposition", [])]
    decomp_q_tokens = [x for x in decomp_q_tokens if x]
    if not query_texts or not decomp_q_tokens:
        return 0.0

    best = 0.0
    for q in query_texts:
        q_tokens = content_tokens(q)
        if not q_tokens:
            continue
        for dq in decomp_q_tokens:
            score = len(q_tokens & dq) / len(dq)
            if score > best:
                best = score
    return best


def classify_failure(
    pred: dict[str, Any],
    gold_item: dict[str, Any],
    chunk_map: dict[str, str],
    chunk_token_sets: dict[str, set[str]],
    corpus_norm_texts: list[str],
    support_overlap_threshold: float,
) -> tuple[str, dict[str, Any]]:
    """Classify one failed prediction into the 5 requested categories."""
    gold_answer = gold_item.get("answer", pred.get("gold_answer", ""))
    gold_answer_norm = normalize_text(gold_answer)

    retrieved_ids = collect_retrieved_chunk_ids(pred)
    query_texts = collect_query_texts(pred)
    retrieval_attempts = len(pred.get("retrieval_logs", []))

    supports = support_paragraphs(gold_item)
    support_matches: list[bool] = []
    for s in supports:
        support_matches.append(
            paragraph_is_retrieved(
                s.get("paragraph_text", ""),
                retrieved_ids,
                chunk_token_sets,
                support_overlap_threshold,
            )
        )

    matched_supports = sum(support_matches)
    total_supports = len(support_matches)
    support_recall = matched_supports / total_supports if total_supports else 0.0

    retrieved_text_norm = " ".join(normalize_text(chunk_map.get(cid, "")) for cid in retrieved_ids)
    answer_in_retrieved = bool(gold_answer_norm) and (gold_answer_norm in retrieved_text_norm)

    answer_in_corpus = False
    if gold_answer_norm:
        answer_in_corpus = any(gold_answer_norm in chunk_norm for chunk_norm in corpus_norm_texts)

    hop1_answer = ""
    if gold_item.get("question_decomposition"):
        hop1_answer = str(gold_item["question_decomposition"][0].get("answer", ""))
    hop1_answer_norm = normalize_text(hop1_answer)
    hop1_found = False
    if total_supports > 0:
        hop1_found = support_matches[0]
    if not hop1_found and hop1_answer_norm:
        hop1_found = hop1_answer_norm in retrieved_text_norm

    decomp_query_overlap = question_overlap_with_decomp(query_texts, gold_item)

    if not answer_in_corpus:
        category = "corpus_gap"
    elif total_supports > 0 and support_recall >= 1.0:
        category = "retrieved_but_couldnt_synthesize"
    elif answer_in_retrieved and total_supports > 0 and support_recall >= 0.5:
        category = "retrieved_but_couldnt_synthesize"
    elif total_supports >= 2 and retrieval_attempts <= 1 and hop1_found:
        category = "never_searched_hop2"
    elif decomp_query_overlap < 0.15 and support_recall < 0.5:
        category = "decomposition_failure"
    elif retrieval_attempts >= 2 and support_recall < 1.0:
        category = "searched_but_missed"
    else:
        category = "decomposition_failure"

    details = {
        "qid": pred.get("qid", ""),
        "retrieval_attempts": retrieval_attempts,
        "retrieved_chunk_count": len(retrieved_ids),
        "support_recall": round(support_recall, 4),
        "matched_supports": matched_supports,
        "total_supports": total_supports,
        "hop1_found": hop1_found,
        "decomp_query_overlap": round(decomp_query_overlap, 4),
        "answer_in_retrieved": answer_in_retrieved,
        "answer_in_corpus": answer_in_corpus,
    }
    return category, details


def analyze_experiment(
    exp_name: str,
    predictions: list[dict[str, Any]],
    gold_map: dict[str, dict[str, Any]],
    chunk_map: dict[str, str],
    chunk_token_sets: dict[str, set[str]],
    corpus_norm_texts: list[str],
    support_overlap_threshold: float,
) -> dict[str, Any]:
    """Run failure taxonomy for one experiment."""
    total = len(predictions)
    failed = [p for p in predictions if float(p.get("llm_accuracy", 0.0)) < 1.0]
    correct = total - len(failed)

    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "never_searched_hop2": [],
        "searched_but_missed": [],
        "retrieved_but_couldnt_synthesize": [],
        "decomposition_failure": [],
        "corpus_gap": [],
    }

    for pred in failed:
        gid = strip_musique_prefix(str(pred.get("qid", "")))
        gold_item = gold_map.get(gid)
        if gold_item is None:
            category = "decomposition_failure"
            details = {
                "qid": pred.get("qid", ""),
                "note": "missing_gold_record",
            }
        else:
            category, details = classify_failure(
                pred,
                gold_item,
                chunk_map,
                chunk_token_sets,
                corpus_norm_texts,
                support_overlap_threshold,
            )

        counts[category] += 1
        if len(examples[category]) < 5:
            examples[category].append(details)

    failure_total = len(failed)
    category_stats: dict[str, dict[str, Any]] = {}
    ordered = [
        "never_searched_hop2",
        "searched_but_missed",
        "retrieved_but_couldnt_synthesize",
        "decomposition_failure",
        "corpus_gap",
    ]
    for name in ordered:
        c = counts[name]
        category_stats[name] = {
            "count": c,
            "pct_of_failures": round(c / failure_total, 4) if failure_total else 0.0,
            "pct_of_total": round(c / total, 4) if total else 0.0,
            "examples": examples[name],
        }

    return {
        "experiment": exp_name,
        "total_samples": total,
        "correct_samples": correct,
        "failed_samples": failure_total,
        "failure_rate": round(failure_total / total, 4) if total else 0.0,
        "categories": category_stats,
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="MuSiQue failure taxonomy for E1-E4")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/projects/prjs1800/msc-thesis/01-arag-reproduction/results"),
    )
    parser.add_argument(
        "--musique-data-root",
        type=Path,
        default=Path("/projects/prjs1800/datasets/musique"),
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=Path("/projects/prjs1800/external/arag/data/musique/chunks.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/projects/prjs1800/msc-thesis/01-arag-reproduction/analysis/musique_failure_taxonomy.json"),
    )
    parser.add_argument(
        "--support-overlap-threshold",
        type=float,
        default=0.35,
        help="Token-overlap recall threshold to mark a support paragraph as retrieved.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    predictions_by_exp: dict[str, list[dict[str, Any]]] = {}
    needed_ids: set[str] = set()

    for exp, folder in EXPERIMENTS.items():
        pred_path = args.results_root / folder / "musique" / "predictions.jsonl"
        rows = load_predictions(pred_path)
        predictions_by_exp[exp] = rows
        for row in rows:
            needed_ids.add(strip_musique_prefix(str(row.get("qid", ""))))
        LOGGER.info("Loaded %s rows for %s", len(rows), exp)

    gold_map = load_musique_gold_map(args.musique_data_root, needed_ids)

    chunk_map = parse_chunk_texts(args.chunks_file)
    chunk_token_sets = {cid: content_tokens(text) for cid, text in chunk_map.items()}
    corpus_norm_texts = [normalize_text(text) for text in chunk_map.values()]

    output: dict[str, Any] = {
        "heuristics": {
            "support_overlap_threshold": args.support_overlap_threshold,
            "correct_if": "llm_accuracy == 1.0",
            "failure_if": "llm_accuracy < 1.0",
            "note": "Heuristic taxonomy using MuSiQue decomposition + ARAG retrieval traces.",
        },
        "experiments": {},
    }

    for exp, rows in predictions_by_exp.items():
        result = analyze_experiment(
            exp,
            rows,
            gold_map,
            chunk_map,
            chunk_token_sets,
            corpus_norm_texts,
            args.support_overlap_threshold,
        )
        output["experiments"][exp] = result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved taxonomy to %s", args.output)

    for exp in EXPERIMENTS:
        r = output["experiments"][exp]
        LOGGER.info(
            "%s failures: %d/%d (%.1f%%)",
            exp,
            r["failed_samples"],
            r["total_samples"],
            100 * r["failure_rate"],
        )
        for cat, cinfo in r["categories"].items():
            LOGGER.info("  %s: %d (%.1f%% of failures)", cat, cinfo["count"], 100 * cinfo["pct_of_failures"])


if __name__ == "__main__":
    main()
