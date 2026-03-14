#!/usr/bin/env python3
"""Collect GRPO training trajectories by running the pipeline with temperature sampling.

For each question, samples G decompositions and runs the full pipeline for each,
recording (prompt, completion, reward) tuples for offline GRPO training.

Usage:
    PYTHONPATH=src python -u scripts/collect_trajectories.py \
        --config configs/adaptive_agency.yaml \
        --questions data/hotpotqa/questions.json \
        --output trajectories/hotpotqa \
        --limit 800 --group-size 4 --concurrent 50 \
        --temperature 0.7 --lambda-efficiency 0.1
"""

from __future__ import annotations

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
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.registry import ToolRegistry
from arag.tools.semantic_search import SemanticSearchTool
from multi_agent.adaptive_pipeline import AdaptiveAgencyPipeline

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


def compute_reward(
    pred: str, gold: str, tokens_used: int, token_budget: int,
    lambda_eff: float = 0.1,
) -> float:
    accuracy = compute_em(pred, gold)
    if accuracy == 0.0:
        accuracy = 0.5 * compute_f1(pred, gold)
    efficiency_penalty = tokens_used / token_budget
    return accuracy - lambda_eff * efficiency_penalty


def build_tools(config: Config) -> ToolRegistry:
    data_cfg = config.get("data", {})
    chunks_file = data_cfg.get("chunks_file", "data/chunks.json")
    reg = ToolRegistry()
    reg.register(KeywordSearchTool(chunks_file=chunks_file))
    reg.register(ReadChunkTool(chunks_file=chunks_file))

    index_dir = data_cfg.get("index_dir")
    if index_dir and Path(index_dir).exists():
        emb_cfg = config.get("embedding", {})
        reg.register(SemanticSearchTool(
            chunks_file=chunks_file,
            index_dir=index_dir,
            model_name=emb_cfg.get("model", "intfloat/e5-base-v2"),
            device=emb_cfg.get("device"),
        ))
    return reg


def create_pipeline(config: Config, temperature: float) -> AdaptiveAgencyPipeline:
    llm_cfg = config.get("llm", {})
    client = LLMClient(
        model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-8B"),
        api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
        base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 8192),
    )

    data_cfg = config.get("data", {})
    tools = build_tools(config)
    adaptive_cfg = config.get("adaptive", {})
    structured_cfg = adaptive_cfg.get("structured", {})

    return AdaptiveAgencyPipeline(
        llm_client=client,
        worker_llm_client=client,
        tools=tools,
        worker_max_steps=adaptive_cfg.get("worker_max_steps", 16),
        token_budget=adaptive_cfg.get("token_budget", 300_000),
        wall_clock_timeout=adaptive_cfg.get("wall_clock_timeout", 900.0),
        idle_timeout=adaptive_cfg.get("idle_timeout", 300.0),
        max_actions=adaptive_cfg.get("max_actions", 100),
        enable_consistency_check=adaptive_cfg.get("enable_consistency_check", False),
        max_redecompositions=adaptive_cfg.get("max_redecompositions", 1),
        structured_retrieval_top_k=structured_cfg.get("retrieval_top_k", 10),
        structured_max_queries=structured_cfg.get("max_queries_per_entity", 6),
        structured_retry_low_confidence=structured_cfg.get("retry_low_confidence", True),
        structured_confidence_threshold=structured_cfg.get("confidence_threshold", 0.6),
        decompose_temperature=temperature,
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


async def collect_one_sample(
    item: dict[str, Any],
    config: Config,
    temperature: float,
    token_budget: int,
    lambda_eff: float,
    semaphore: asyncio.Semaphore,
    sample_idx: int,
) -> dict[str, Any]:
    async with semaphore:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold = item.get("answer", item.get("gold_answer", ""))

        pipeline = create_pipeline(config, temperature)
        result = await pipeline.run(question)

        reward = compute_reward(
            result.pred_answer, gold,
            result.total_tokens, token_budget,
            lambda_eff,
        )

        return {
            "qid": qid,
            "sample_idx": sample_idx,
            "question": question,
            "gold_answer": gold,
            "pred_answer": result.pred_answer,
            "decomposition_text": result.decomposition_text,
            "total_tokens": result.total_tokens,
            "wall_clock_seconds": result.wall_clock_seconds,
            "mode_distribution": result.mode_distribution,
            "mode_tokens": result.mode_tokens,
            "em": compute_em(result.pred_answer, gold),
            "f1": compute_f1(result.pred_answer, gold),
            "reward": reward,
            "num_sub_questions": result.num_sub_questions,
            "termination_reason": result.termination_reason,
        }


async def run_collection(
    config: Config,
    questions_file: str,
    output_dir: Path,
    limit: int | None,
    group_size: int,
    concurrent: int,
    temperature: float,
    lambda_eff: float,
):
    with open(questions_file) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_file = output_dir / "trajectories.jsonl"

    completed_qids: set = set()
    if trajectories_file.exists():
        with open(trajectories_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed_qids.add(d["qid"])
        existing_per_q = {}
        with open(trajectories_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    existing_per_q[d["qid"]] = existing_per_q.get(d["qid"], 0) + 1
        fully_done = {qid for qid, count in existing_per_q.items() if count >= group_size}
        logger.info("Resuming: %d questions fully collected", len(fully_done))
    else:
        fully_done = set()

    pending = [q for q in questions if (q.get("qid") or q.get("id")) not in fully_done]
    logger.info("Collecting %d samples each for %d questions (%d concurrent)",
                group_size, len(pending), concurrent)

    adaptive_cfg = config.get("adaptive", {})
    token_budget = adaptive_cfg.get("token_budget", 300_000)

    semaphore = asyncio.Semaphore(concurrent)
    t0 = time.monotonic()
    completed = 0

    tasks = []
    for item in pending:
        for sample_idx in range(group_size):
            tasks.append(collect_one_sample(
                item, config, temperature, token_budget,
                lambda_eff, semaphore, sample_idx,
            ))

    with open(trajectories_file, "a") as fout:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            completed += 1
            if completed % 50 == 0:
                elapsed = time.monotonic() - t0
                logger.info("Progress: %d/%d trajectories (%.1f/hr)",
                            completed, len(tasks), completed / elapsed * 3600)

    elapsed = time.monotonic() - t0
    logger.info("Collection complete: %d trajectories in %.1f min", completed, elapsed / 60)

    with open(trajectories_file) as f:
        all_traj = [json.loads(l) for l in f if l.strip()]

    rewards = [t["reward"] for t in all_traj]
    ems = [t["em"] for t in all_traj]
    tokens = [t["total_tokens"] for t in all_traj]
    logger.info("Stats: avg_reward=%.3f, avg_em=%.3f, avg_tokens=%d",
                sum(rewards) / len(rewards),
                sum(ems) / len(ems),
                sum(tokens) // len(tokens))

    summary = {
        "total_trajectories": len(all_traj),
        "total_questions": len(set(t["qid"] for t in all_traj)),
        "group_size": group_size,
        "temperature": temperature,
        "lambda_efficiency": lambda_eff,
        "avg_reward": sum(rewards) / len(rewards),
        "avg_em": sum(ems) / len(ems),
        "avg_tokens": sum(tokens) / len(tokens),
    }
    with open(output_dir / "collection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect GRPO trajectories")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--questions", "-q", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--group-size", "-g", type=int, default=4)
    parser.add_argument("--concurrent", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lambda-efficiency", type=float, default=0.1)
    parser.add_argument("--chunks-file", type=str, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.chunks_file:
        config.set("data.chunks_file", args.chunks_file)
    align_data_paths(config, args.questions)

    asyncio.run(run_collection(
        config=config,
        questions_file=args.questions,
        output_dir=Path(args.output),
        limit=args.limit,
        group_size=args.group_size,
        concurrent=args.concurrent,
        temperature=args.temperature,
        lambda_eff=args.lambda_efficiency,
    ))


if __name__ == "__main__":
    main()
