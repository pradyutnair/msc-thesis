#!/usr/bin/env python3
"""Run the escalation pipeline on a dataset and produce predictions.

Supports multiple escalation modes for ablation:
  - learned:            RL-trained escalation policy (our method)
  - heuristic:          rule-based (escalate if structured fails)
  - always_structured:  never escalate (lower bound)
  - always_agentic:     always escalate (upper bound, expensive)

Usage:
    PYTHONPATH=src python -u scripts/run_escalation.py \
        --config configs/adaptive_agency.yaml \
        --questions data/hotpotqa/questions.json \
        --output results/escalation_learned/hotpotqa \
        --mode learned \
        --limit 1000 --concurrent 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from arag.tools.build_tools import build_tools
from multi_agent.escalation_pipeline import EscalationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATASET_DIRS = {"hotpotqa": "hotpotqa", "musique": "musique", "2wiki": "2wikimultihop"}


def normalize(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_em(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)



def create_llm_client(config: Config) -> LLMClient:
    llm_cfg = config.get("llm", {})
    return LLMClient(
        model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-8B"),
        api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
        base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
        temperature=0.0,
        max_tokens=llm_cfg.get("max_tokens", 8192),
    )


def align_data_paths(config: Config, questions_file: str) -> None:
    raw = questions_file.lower()
    for key, dirname in LOCAL_DATASET_DIRS.items():
        if key in raw or dirname in raw:
            data_dir = PROJECT_ROOT / "data" / dirname
            chunks = data_dir / "chunks.json"
            index = data_dir / "index_e5_base_v2"
            if chunks.exists() and not config.get("data.chunks_file"):
                config.set("data.chunks_file", str(chunks))
            if index.exists() and not config.get("data.index_dir"):
                config.set("data.index_dir", str(index))
            break


def create_pipeline(config: Config, mode: str, tools: ToolRegistry = None) -> EscalationPipeline:
    llm = create_llm_client(config)
    if tools is None:
        tools = build_tools(config)
    adaptive_cfg = config.get("adaptive", {})
    structured_cfg = adaptive_cfg.get("structured", {})

    # For learned mode, create a separate LLM client that routes to the
    # escalation LoRA adapter via vLLM's multi-LoRA model name routing.
    escalation_llm = None
    if mode == "learned":
        llm_cfg = config.get("llm", {})
        escalation_model = os.getenv("ESCALATION_MODEL", "escalation")
        escalation_llm = LLMClient(
            model=escalation_model,
            api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
            base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
            temperature=0.0,
            max_tokens=100,
        )

    return EscalationPipeline(
        llm_client=llm,
        worker_llm_client=llm,
        tools=tools,
        escalation_mode=mode,
        escalation_llm_client=escalation_llm,
        worker_max_steps=adaptive_cfg.get("worker_max_steps", 16),
        token_budget=adaptive_cfg.get("token_budget", 300_000),
        structured_retrieval_top_k=structured_cfg.get("retrieval_top_k", 10),
        structured_max_queries=structured_cfg.get("max_queries_per_entity", 6),
        structured_retry_low_confidence=structured_cfg.get("retry_low_confidence", True),
        structured_confidence_threshold=structured_cfg.get("confidence_threshold", 0.6),
    )


async def run_one_question(
    item: dict[str, Any],
    config: Config,
    mode: str,
    semaphore: asyncio.Semaphore,
    shared_tools: ToolRegistry = None,
) -> dict[str, Any]:
    async with semaphore:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold = item.get("answer", item.get("gold_answer", ""))

        pipeline = create_pipeline(config, mode, tools=shared_tools)

        try:
            result = await pipeline.run(question)
        except Exception as exc:
            logger.error("Pipeline error for %s: %s", qid, exc)
            return {
                "qid": qid, "question": question, "gold_answer": gold,
                "pred_answer": f"Error: {exc}", "em": 0.0, "f1": 0.0,
                "total_tokens": 0, "wall_clock_seconds": 0.0,
                "error": str(exc),
            }

        pred = result.pred_answer
        em = compute_em(pred, gold)
        f1 = compute_f1(pred, gold)

        return {
            "qid": qid,
            "question": question,
            "gold_answer": gold,
            "pred_answer": pred,
            "em": em,
            "f1": f1,
            "total_tokens": result.total_tokens,
            "wall_clock_seconds": result.wall_clock_seconds,
            "num_sub_questions": result.num_sub_questions,
            "mode_distribution": result.mode_distribution,
            "mode_tokens": result.mode_tokens,
            "sub_question_details": result.sub_question_details,
        }


async def run_evaluation(
    config: Config,
    questions_file: str,
    output_dir: Path,
    mode: str,
    limit: int | None,
    concurrent: int,
):
    with open(questions_file) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / "predictions.jsonl"

    # Resume support
    completed_qids: set = set()
    if predictions_file.exists():
        with open(predictions_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed_qids.add(d["qid"])
        logger.info("Resuming: %d questions already evaluated", len(completed_qids))

    pending = [q for q in questions if (q.get("qid") or q.get("id")) not in completed_qids]
    logger.info(
        "Evaluating %d questions with mode=%s (%d concurrent)",
        len(pending), mode, concurrent,
    )

    semaphore = asyncio.Semaphore(concurrent)
    t0 = time.monotonic()
    completed = 0

    # Build tools ONCE and share across all questions
    shared_tools = build_tools(config)
    logger.info("Tools loaded: %s", shared_tools.list_tools())

    tasks = [run_one_question(item, config, mode, semaphore, shared_tools=shared_tools) for item in pending]

    em_sum = 0.0
    f1_sum = 0.0
    token_sum = 0

    with open(predictions_file, "a") as fout:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            completed += 1
            em_sum += result.get("em", 0.0)
            f1_sum += result.get("f1", 0.0)
            token_sum += result.get("total_tokens", 0)

            if completed % 20 == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "Progress: %d/%d | EM=%.1f%% | F1=%.1f%% | avg_tokens=%d | %.1f q/hr",
                    completed, len(pending),
                    em_sum / completed * 100,
                    f1_sum / completed * 100,
                    token_sum // completed,
                    completed / elapsed * 3600,
                )

    elapsed = time.monotonic() - t0
    logger.info("Evaluation complete: %d questions in %.1f min", completed, elapsed / 60)

    # Write summary
    total = completed + len(completed_qids)
    with open(predictions_file) as f:
        all_preds = [json.loads(line) for line in f if line.strip()]

    total_em = sum(p.get("em", 0.0) for p in all_preds)
    total_f1 = sum(p.get("f1", 0.0) for p in all_preds)
    total_tok = sum(p.get("total_tokens", 0) for p in all_preds)

    # Count escalation decisions
    n_accept = 0
    n_escalate = 0
    for p in all_preds:
        md = p.get("mode_distribution", {})
        n_accept += md.get("accept", 0)
        n_escalate += md.get("escalate", 0)

    summary = {
        "mode": mode,
        "total": len(all_preds),
        "norm_em": round(total_em / len(all_preds), 4) if all_preds else 0,
        "token_f1": round(total_f1 / len(all_preds), 4) if all_preds else 0,
        "avg_tokens": total_tok // len(all_preds) if all_preds else 0,
        "total_accept": n_accept,
        "total_escalate": n_escalate,
        "escalation_rate": round(
            n_escalate / (n_accept + n_escalate), 4
        ) if (n_accept + n_escalate) > 0 else 0,
    }

    with open(output_dir / "offline_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run escalation pipeline evaluation")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--questions", "-q", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument(
        "--mode", "-m", required=True,
        choices=["learned", "heuristic", "always_structured", "always_agentic"],
    )
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--chunks-file", type=str, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.chunks_file:
        config.set("data.chunks_file", args.chunks_file)
    align_data_paths(config, args.questions)

    asyncio.run(run_evaluation(
        config=config,
        questions_file=args.questions,
        output_dir=Path(args.output),
        mode=args.mode,
        limit=args.limit,
        concurrent=args.concurrent,
    ))


if __name__ == "__main__":
    main()
