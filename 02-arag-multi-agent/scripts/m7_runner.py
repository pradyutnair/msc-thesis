#!/usr/bin/env python3
"""Batch runner for M7 CORAL multi-agent pipeline.

Usage:
    python scripts/m7_runner.py \
        --config configs/m7v1.yaml \
        --dataset musique \
        --output results/m7v1_pilot10/musique \
        --limit 10 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multi_agent.m7.llm_client import VllmChatClient, token_tracker
from multi_agent.m7.retriever import M7Retriever
from multi_agent.m7.pipeline import M7Pipeline, M7PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DATASET_ALIASES = {
    "hotpotqa": "hotpotqa",
    "2wiki": "2wikimultihop",
    "2wikimultihop": "2wikimultihop",
    "musique": "musique",
}


def load_config(path: str) -> dict[str, Any]:
    """Load YAML config file."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_questions(path: Path, limit: int | None, offset: int = 0) -> list[dict[str, Any]]:
    """Load questions from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions = questions[offset:]
    if limit:
        questions = questions[:limit]
    return questions


def load_completed_qids(predictions_path: Path) -> set:
    """Load QIDs already completed (for resume support)."""
    completed = set()
    if not predictions_path.exists():
        return completed
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                qid = data.get("qid") or data.get("id")
                if qid is not None:
                    completed.add(str(qid))
            except json.JSONDecodeError:
                continue
    return completed


def run_one(
    item: dict[str, Any],
    pipeline: M7Pipeline,
    item_idx: int,
) -> dict[str, Any]:
    """Process one question through the M7 CORAL pipeline."""
    qid = str(item.get("id") or item.get("qid") or item_idx)
    question = item["question"]
    gold = item.get("answer", "")

    token_tracker.reset()
    t0 = time.time()

    try:
        pred_answer, trace = pipeline.answer_question(question)
    except Exception as exc:
        logger.exception("Question %s failed", qid)
        pred_answer = f"Error: {exc}"
        trace = {"error": str(exc)}

    elapsed = time.time() - t0
    tokens = token_tracker.snapshot()

    return {
        "qid": qid,
        "question": question,
        "gold_answer": gold,
        "pred_answer": pred_answer,
        "wall_clock_seconds": round(elapsed, 3),
        "prompt_tokens": tokens["prompt_tokens"],
        "completion_tokens": tokens["completion_tokens"],
        "total_tokens": tokens["total_tokens"],
        "api_calls": tokens["api_calls"],
        "retrieval_rounds": trace.get("retrieval_rounds", 0),
        "num_sub_questions": len(trace.get("sub_questions", [])),
        "question_type": trace.get("question_type", ""),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M7 CORAL Multi-Agent Pipeline Runner")
    parser.add_argument("--config", "-c", required=True, help="YAML config file")
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASET_ALIASES))
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", "-w", type=int, default=4, help="Concurrent question workers")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = DATASET_ALIASES[args.dataset]
    data_root = PROJECT_ROOT / "data" / dataset

    # Resolve paths
    questions_file = data_root / "questions.json"
    index_dir = data_root / "index_e5_base_v2"

    # Output setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    # Load questions
    questions = load_questions(questions_file, args.limit, args.offset)
    completed_qids = load_completed_qids(predictions_path)
    pending = [
        q for q in questions
        if str(q.get("id") or q.get("qid")) not in completed_qids
    ]
    logger.info(
        "Dataset=%s | Total=%d | Completed=%d | Pending=%d",
        dataset, len(questions), len(completed_qids), len(pending),
    )
    if not pending:
        logger.info("All questions already completed!")
        return

    # Build LLM client
    llm_cfg = config.get("llm", {})
    llm = VllmChatClient(
        model=llm_cfg.get("model", os.getenv("ARAG_MODEL", "Qwen3-8B")),
        base_url=llm_cfg.get("base_url", os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1")),
        api_key=llm_cfg.get("api_key", os.getenv("ARAG_API_KEY", "dummy")),
        timeout=llm_cfg.get("timeout", 600),
    )

    # Build retriever
    retriever_cfg = config.get("retriever", {})
    retriever = M7Retriever(
        index_dir=str(index_dir),
        embed_model_name=retriever_cfg.get("embed_model", "intfloat/e5-base-v2"),
        device=retriever_cfg.get("device", "cpu"),
        neighborhood=retriever_cfg.get("neighborhood", 2),
    )

    # Build pipeline
    pipeline_cfg = config.get("pipeline", {})
    prompts_dir = PROJECT_ROOT / "src" / "multi_agent" / "m7" / "prompts"
    m6_prompts_dir = PROJECT_ROOT / "src" / "multi_agent" / "m6" / "prompts"

    pipeline = M7Pipeline(
        llm=llm,
        retriever=retriever,
        decomposer_prompt=str(m6_prompts_dir / pipeline_cfg.get("decomposer_prompt", "decomposer_v29.txt")),
        worker_prompt=str(prompts_dir / pipeline_cfg.get("worker_prompt", "worker.txt")),
        synthesizer_prompt=str(m6_prompts_dir / pipeline_cfg.get("synthesizer_prompt", "synthesizer_v29.txt")),
        top_k=pipeline_cfg.get("top_k", 10),
        max_retries=pipeline_cfg.get("max_retries", 1),
        evidence_pool_top_k=pipeline_cfg.get("evidence_pool_top_k", 5),
    )

    # Run
    t0 = time.time()
    completed = 0
    write_lock = __import__("threading").Lock()

    def _process_and_write(item, idx):
        nonlocal completed
        record = run_one(item, pipeline, idx)
        with write_lock:
            with open(predictions_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            completed += 1
            if completed % 5 == 0 or completed == len(pending):
                elapsed = time.time() - t0
                rate = completed / elapsed * 3600 if elapsed > 0 else 0.0
                logger.info(
                    "Progress %d/%d | %.1f Q/hr | last: '%s' -> '%s'",
                    completed, len(pending), rate,
                    record["question"][:40], record["pred_answer"][:40],
                )
        return record

    logger.info("Starting M7 CORAL pipeline with %d workers", args.workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_process_and_write, item, idx)
            for idx, item in enumerate(pending, start=1)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error("Unhandled error: %s", exc)

    elapsed = time.time() - t0
    logger.info(
        "Batch complete: %d questions in %.1f min (%.1f Q/hr)",
        completed, elapsed / 60,
        completed / elapsed * 3600 if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
