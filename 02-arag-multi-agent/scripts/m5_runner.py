#!/usr/bin/env python3
"""Async batch runner for M5 multi-agent pipeline.

M5 = orchestrator (thinking ON) + E2-style subagents (thinking OFF).
The orchestrator decomposes multi-hop questions via DELEGATE/ANSWER
protocol; each delegation spawns a fresh BaseAgent with raw retrieval
tools (keyword_search, semantic_search, read_chunk).

Usage:
    python scripts/m5_runner.py \
        --config configs/m5_qwen3_8b.yaml \
        --output results/m5_qwen3_8b_pilot100/hotpotqa/ \
        --limit 100 --offset 0 --workers 100
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool
from arag.tools.read_chunk import ReadChunkTool

from multi_agent.m5.m5_pipeline import M5Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_tools(config: Config) -> ToolRegistry:
    """Build tool registry from config (same as E2)."""
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


def result_to_prediction(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Serialize M5Pipeline result to prediction dict."""
    qid = item.get("qid") or item.get("id")
    gold = item.get("answer", item.get("gold_answer", ""))
    return {
        "qid": qid,
        "question": item.get("question", ""),
        "gold_answer": gold,
        "pred_answer": result.get("answer", ""),
        "loops": result.get("loops", 0),
        "findings": result.get("findings", []),
        "total_cost": result.get("total_cost", 0.0),
        "wall_clock_seconds": result.get("wall_clock_seconds", 0.0),
        "error": result.get("error"),
    }


class M5BatchRunner:
    """Async batch runner with checkpoint resume for M5 pipeline."""

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
        # Apply offset then limit
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

    def _create_pipeline(self) -> M5Pipeline:
        """Create M5Pipeline with separate orchestrator/subagent LLM clients."""
        llm_cfg = self.config.get("llm", {})
        model = llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-8B")
        api_key = llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy")
        base_url = llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1")

        # Orchestrator: thinking ON (default for Qwen3)
        orchestrator_client = LLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 8192),
        )

        # Subagent: thinking OFF
        sub_cfg = self.config.get("subagent", {})
        subagent_client = LLMClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=sub_cfg.get("max_tokens", 4096),
            chat_template_kwargs={"enable_thinking": False},
        )

        # Shared tools (same as E2)
        tools = build_tools(self.config)

        # Load prompts
        prompts_dir = PROJECT_ROOT / "src" / "multi_agent" / "m5" / "prompts"
        orch_prompt = (prompts_dir / "orchestrator_v2.txt").read_text(encoding="utf-8")
        sub_prompt = (prompts_dir / "subagent_v2.txt").read_text(encoding="utf-8")

        agent_cfg = self.config.get("agent", {})
        return M5Pipeline(
            orchestrator_llm=orchestrator_client,
            subagent_llm=subagent_client,
            shared_tools=tools,
            orchestrator_prompt=orch_prompt,
            subagent_prompt=sub_prompt,
            max_iterations=agent_cfg.get("max_loops", 10),
            subagent_max_loops=sub_cfg.get("max_loops", 5),
            subagent_max_budget=sub_cfg.get("max_token_budget", 32000),
        )

    async def _process_one(
        self,
        item: dict[str, Any],
        pipeline: M5Pipeline,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            qid = item.get("qid") or item.get("id")
            question = item.get("question", "")
            logger.info("Processing %s: %s", qid, question[:60])

            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(None, pipeline.run, question)
            except Exception as exc:
                logger.error("Pipeline error for %s: %s", qid, exc)
                result = {
                    "answer": f"Error: {exc}",
                    "trajectory": [],
                    "total_cost": 0.0,
                    "loops": 0,
                    "findings": [],
                    "wall_clock_seconds": 0.0,
                    "error": str(exc),
                }

            prediction = result_to_prediction(item, result)
            await self._append_prediction(prediction)

            logger.info(
                "Done %s: '%s' (%d loops, %.1fs)",
                qid,
                result.get("answer", "")[:40],
                result.get("loops", 0),
                result.get("wall_clock_seconds", 0.0),
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

    parser = argparse.ArgumentParser(description="M5 Multi-Agent Pipeline Runner")
    parser.add_argument("--config", "-c", required=True, help="YAML config path")
    parser.add_argument("--dataset", "-d", required=True, choices=list(DATASET_ALIASES), help="Dataset name")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Max questions")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions (for sharding)")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Max concurrent questions")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    # Set data paths from dataset name
    dataset = DATASET_ALIASES[args.dataset]
    data_root = PROJECT_ROOT / "data" / dataset
    config.set("data.chunks_file", str(data_root / "chunks.json"))
    config.set("data.questions_file", str(data_root / "questions.json"))
    config.set("data.index_dir", str(data_root / "index_e5_base_v2"))

    runner = M5BatchRunner(
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
