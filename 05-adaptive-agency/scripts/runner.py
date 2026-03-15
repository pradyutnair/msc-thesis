#!/usr/bin/env python3
"""Async batch runner for the Adaptive Agency pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
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
from multi_agent.types import M6PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATASET_DIRS = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wiki": "2wikimultihop",
}


class AdaptiveBatchRunner:
    """Async batch runner with checkpoint resume."""

    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int | None = None,
        max_concurrent_questions: int = 5,
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.max_concurrent_questions = max_concurrent_questions

        self._align_data_paths_with_questions()

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self._write_lock = asyncio.Lock()
        self.questions = self._load_questions()

    def _infer_dataset_key(self, path: Path) -> str | None:
        raw = str(path).lower()
        if "hotpot" in raw:
            return "hotpotqa"
        if "musique" in raw:
            return "musique"
        if "2wiki" in raw:
            return "2wiki"
        return None

    def _align_data_paths_with_questions(self) -> None:
        self.config.set("data.questions_file", str(self.questions_file))
        dataset_key = self._infer_dataset_key(self.questions_file)
        if dataset_key is None:
            return

        dataset_dir = LOCAL_DATASET_DIRS[dataset_key]
        local_data_dir = PROJECT_ROOT / "data" / dataset_dir
        chunks_candidate = local_data_dir / "chunks.json"
        index_candidate = local_data_dir / "index_e5_base_v2"

        if not chunks_candidate.exists():
            logger.warning("Missing chunks: %s", chunks_candidate)
            return

        if not self.config.get("data.chunks_file"):
            self.config.set("data.chunks_file", str(chunks_candidate))
        logger.info("Using chunks file: %s", self.config.get("data.chunks_file"))
        if index_candidate.exists() and not self.config.get("data.index_dir"):
            self.config.set("data.index_dir", str(index_candidate))
            logger.info("Using index dir: %s", index_candidate)

    def _load_questions(self) -> list[dict[str, Any]]:
        with open(self.questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        if self.limit:
            questions = questions[: self.limit]
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

    def _build_tools(self, data_cfg: dict) -> ToolRegistry:
        from arag.tools.build_tools import build_tools
        return build_tools(self.config)

    def _create_pipeline(self) -> AdaptiveAgencyPipeline:
        llm_cfg = self.config.get("llm", {})
        client = LLMClient(
            model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-8B"),
            api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
            base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            chat_template_kwargs=llm_cfg.get("chat_template_kwargs"),
        )

        data_cfg = self.config.get("data", {})
        tools = self._build_tools(data_cfg)

        adaptive_cfg = self.config.get("adaptive", {})
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
            decompose_temperature=adaptive_cfg.get("decompose_temperature", 0.0),
            force_mode=adaptive_cfg.get("force_mode"),
        )

    def _result_to_prediction(
        self, item: dict[str, Any], result: M6PipelineResult,
    ) -> dict[str, Any]:
        qid = item.get("qid") or item.get("id")
        gold = item.get("answer", item.get("gold_answer", ""))
        return {
            "qid": qid,
            "question": result.question,
            "gold_answer": gold,
            "pred_answer": result.pred_answer,
            "question_type": result.question_type,
            "expected_answer": result.expected_answer,
            "num_sub_questions": result.num_sub_questions,
            "num_workers": result.num_workers,
            "total_ticks": result.total_ticks,
            "total_tokens": result.total_tokens,
            "wall_clock_seconds": result.wall_clock_seconds,
            "backtrack_count": result.backtrack_count,
            "verified_count": result.verified_count,
            "failed_count": result.failed_count,
            "termination_reason": result.termination_reason,
            "mode_distribution": result.mode_distribution,
            "mode_tokens": result.mode_tokens,
            "entity_registry": result.entity_registry,
            "error": result.error,
            "decomposition_text": result.decomposition_text,
        }

    async def _process_one(
        self,
        item: dict[str, Any],
        pipeline: AdaptiveAgencyPipeline,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            qid = item.get("qid") or item.get("id")
            question = item.get("question", "")
            logger.info("Processing %s: %s", qid, question[:60])

            try:
                result = await pipeline.run(question)
            except Exception as exc:
                logger.error("Pipeline error for %s: %s", qid, exc)
                result = M6PipelineResult(
                    question=question,
                    pred_answer=f"Error: {exc}",
                    error=str(exc),
                )

            prediction = self._result_to_prediction(item, result)
            await self._append_prediction(prediction)

            logger.info(
                "Done %s: '%s' (%.1fs, %d tokens, modes=%s)",
                qid,
                result.pred_answer[:40],
                result.wall_clock_seconds,
                result.total_tokens,
                result.mode_distribution,
            )
            return prediction

    async def run(self) -> None:
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
        semaphore = asyncio.Semaphore(self.max_concurrent_questions)

        t0 = time.monotonic()
        tasks = [self._process_one(item, pipeline, semaphore) for item in pending]

        results = []
        completed_count = 0
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed_count += 1
                if completed_count % 10 == 0:
                    elapsed = time.monotonic() - t0
                    rate = completed_count / elapsed * 3600
                    logger.info("Progress: %d/%d (%.0f Q/hr)", completed_count, len(pending), rate)
            except Exception as exc:
                logger.error("Unhandled error: %s", exc)

        elapsed = time.monotonic() - t0
        logger.info(
            "Batch complete: %d questions in %.1f min (%.1f Q/hr)",
            len(results), elapsed / 60,
            len(results) / elapsed * 3600 if elapsed > 0 else 0,
        )

        mode_token_totals: dict[str, int] = {}
        for r in results:
            for mode, tokens in r.get("mode_tokens", {}).items():
                mode_token_totals[mode] = mode_token_totals.get(mode, 0) + tokens
        logger.info("Total tokens by mode: %s", mode_token_totals)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive Agency Batch Runner")
    parser.add_argument("--config", "-c", required=True, help="YAML config path")
    parser.add_argument("--questions", "-q", required=True, help="Questions JSON path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--chunks-file", type=str, default=None)
    parser.add_argument("--index-dir", type=str, default=None)
    parser.add_argument("--concurrent", type=int, default=5)
    parser.add_argument("--force-mode", type=str, default=None,
                        choices=["structured", "agentic", "aggregate", "random"],
                        help="Override planner mode assignment for ablation studies")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.chunks_file:
        config.set("data.chunks_file", args.chunks_file)
    if args.index_dir:
        config.set("data.index_dir", args.index_dir)
    if args.force_mode:
        config.set("adaptive.force_mode", args.force_mode)

    runner = AdaptiveBatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        max_concurrent_questions=args.concurrent,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
