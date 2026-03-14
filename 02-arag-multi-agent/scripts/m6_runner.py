#!/usr/bin/env python3
"""Async batch runner for M6 blackboard-coordinated multi-agent pipeline.

Usage:
    python scripts/m6_runner.py \
        --config configs/m6_blackboard.yaml \
        --dataset hotpotqa \
        --output results/m6v21_pilot100/hotpotqa \
        --limit 100 --workers 50
"""

from __future__ import annotations

import asyncio
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

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool

from multi_agent.m6.m6_pipeline import M6Pipeline
from multi_agent.m6.types import M6PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_tools(config: Config) -> ToolRegistry:
    data_cfg = config.get("data", {})
    chunks_file = data_cfg.get("chunks_file", "data/chunks.json")

    reg = ToolRegistry()
    reg.register(KeywordSearchTool(chunks_file=chunks_file))
    reg.register(ReadChunkTool(chunks_file=chunks_file))

    index_dir = data_cfg.get("index_dir")
    if index_dir and Path(index_dir).exists():
        emb_cfg = config.get("embedding", {})
        model_name = emb_cfg.get("model", "intfloat/e5-base-v2")
        logger.info("Loading embedding model: %s", model_name)
        reg.register(
            SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=model_name,
                device=emb_cfg.get("device"),
            )
        )
        logger.info("Embedding model loaded.")
    else:
        logger.warning("Index dir not found (%s), semantic search disabled", index_dir)

    return reg


def result_to_prediction(item: dict[str, Any], result: M6PipelineResult) -> dict[str, Any]:
    qid = item.get("qid") or item.get("id")
    gold = item.get("answer", item.get("gold_answer", ""))
    return {
        "qid": qid,
        "question": result.question,
        "gold_answer": gold,
        "pred_answer": result.pred_answer,
        "num_sub_questions": result.num_sub_questions,
        "num_workers": result.num_workers,
        "total_ticks": result.total_ticks,
        "total_tokens": result.total_tokens,
        "wall_clock_seconds": result.wall_clock_seconds,
        "backtrack_count": result.backtrack_count,
        "verified_count": result.verified_count,
        "failed_count": result.failed_count,
        "entity_registry": result.entity_registry,
        "termination_reason": result.termination_reason,
        "error": result.error,
        "retrieved_chunks": [
            {"chunk_id": ev["source_chunk_id"], "worker": ev.get("retriever_id", ""), "sq_id": ev["sub_question_id"]}
            for ev in getattr(result, "evidence", [])
        ],
        "sub_questions": getattr(result, "sub_question_details", []),
    }


class M6BatchRunner:
    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int | None = None,
        offset: int = 0,
        max_concurrent: int = 5,
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.offset = offset
        self.max_concurrent = max_concurrent

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self._write_lock = asyncio.Lock()
        self.questions = self._load_questions()

    def _load_questions(self) -> list[dict[str, Any]]:
        with open(self.questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        questions = questions[self.offset:]
        if self.limit:
            questions = questions[:self.limit]
        return questions

    def _load_completed_qids(self) -> set:
        completed = set()
        if not self.predictions_file.exists():
            return completed
        with open(self.predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    qid = data.get("qid") or data.get("id")
                    if qid is not None:
                        completed.add(qid)
                except json.JSONDecodeError:
                    continue
        return completed

    async def _append_prediction(self, prediction: dict[str, Any]) -> None:
        async with self._write_lock:
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False, default=str) + "\n")

    def _create_pipeline(self) -> M6Pipeline:
        llm_cfg = self.config.get("llm", {})
        model = llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-8B")
        api_key = llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy")
        base_url = llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1")

        main_client = LLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 8192),
        )

        worker_client = LLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=8192,
            chat_template_kwargs={"enable_thinking": True},
        )

        tools = build_tools(self.config)
        m6_cfg = self.config.get("m6", {})
        prompts_dir = PROJECT_ROOT / "src" / "multi_agent" / "m6" / "prompts"

        return M6Pipeline(
            llm_client=main_client,
            worker_llm_client=worker_client,
            tools=tools,
            decomposer_prompt=str(prompts_dir / "decomposer.txt"),
            synthesizer_prompt=str(prompts_dir / "synthesizer.txt"),
            synthesizer_consistency_prompt=str(prompts_dir / "synthesizer_consistency.txt"),
            worker_plan_prompt=str(prompts_dir / "worker_plan.txt"),
            num_workers=m6_cfg.get("num_workers", 2),
            worker_max_steps=m6_cfg.get("worker_max_steps", 8),
            token_budget=m6_cfg.get("token_budget", 200000),
            wall_clock_timeout=m6_cfg.get("wall_clock_timeout", 300.0),
            idle_timeout=m6_cfg.get("idle_timeout", 30.0),
            max_actions=m6_cfg.get("max_actions", 100),
            enable_consistency_check=m6_cfg.get("enable_consistency_check", True),
            max_redecompositions=m6_cfg.get("max_redecompositions", 1),
        )

    async def _process_one(
        self,
        item: dict[str, Any],
        pipeline: M6Pipeline,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            qid = item.get("qid") or item.get("id")
            question = item.get("question", "")
            gold = item.get("answer", item.get("gold_answer", ""))
            logger.info("Processing %s: %s", qid, question[:60])

            try:
                result = await pipeline.run(question)
                result.qid = str(qid)
                result.gold_answer = gold
            except Exception as exc:
                logger.error("Pipeline error for %s: %s", qid, exc)
                result = M6PipelineResult(
                    qid=str(qid),
                    question=question,
                    gold_answer=gold,
                    pred_answer=f"Error: {exc}",
                    error=str(exc),
                )

            prediction = result_to_prediction(item, result)
            await self._append_prediction(prediction)

            logger.info(
                "Done %s: '%s' (ticks=%d, tokens=%d, %.1fs)",
                qid,
                result.pred_answer[:40],
                result.total_ticks,
                result.total_tokens,
                result.wall_clock_seconds,
            )
            return prediction

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=256)
        )
        completed = self._load_completed_qids()
        pending = [
            q for q in self.questions
            if (q.get("qid") or q.get("id")) not in completed
        ]
        logger.info(
            "Total: %d | Completed: %d | Pending: %d",
            len(self.questions), len(completed), len(pending),
        )

        if not pending:
            logger.info("All questions completed!")
            return

        pipeline = self._create_pipeline()
        semaphore = asyncio.Semaphore(self.max_concurrent)

        t0 = time.monotonic()
        tasks = [self._process_one(item, pipeline, semaphore) for item in pending]

        completed_count = 0
        for coro in asyncio.as_completed(tasks):
            try:
                await coro
                completed_count += 1
                if completed_count % 10 == 0:
                    elapsed = time.monotonic() - t0
                    rate = completed_count / elapsed * 3600
                    logger.info(
                        "Progress: %d/%d (%.0f Q/hr)",
                        completed_count, len(pending), rate,
                    )
            except Exception as exc:
                logger.error("Unhandled error: %s", exc)

        elapsed = time.monotonic() - t0
        logger.info(
            "Batch complete: %d questions in %.1f min (%.1f Q/hr)",
            completed_count,
            elapsed / 60,
            completed_count / elapsed * 3600 if elapsed > 0 else 0,
        )


DATASET_ALIASES = {
    "hotpotqa": "hotpotqa",
    "2wiki": "2wikimultihop",
    "2wikimultihop": "2wikimultihop",
    "musique": "musique",
}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="M6 Multi-Agent Runner")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASET_ALIASES))
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", "-w", type=int, default=5)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    dataset = DATASET_ALIASES[args.dataset]
    data_root = PROJECT_ROOT / "data" / dataset
    config.set("data.chunks_file", str(data_root / "chunks.json"))
    config.set("data.questions_file", str(data_root / "questions.json"))
    config.set("data.index_dir", str(data_root / "index_e5_base_v2"))

    runner = M6BatchRunner(
        config=config,
        questions_file=config.get("data.questions_file"),
        output_dir=args.output,
        limit=args.limit,
        offset=args.offset,
        max_concurrent=args.workers,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
