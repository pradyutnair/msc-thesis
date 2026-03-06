#!/usr/bin/env python3
"""Async batch runner for MA²RAG multi-agent pipeline."""

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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arag.core.config import Config
from arag.core.llm import LLMClient
from multi_agent.pipeline import MultiAgentPipeline
from multi_agent.types import PipelineResult

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


class MultiAgentBatchRunner:
    """Async batch runner with checkpoint resume and concurrent question processing."""

    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int | None = None,
        max_concurrent_questions: int = 5,
        verbose: bool = False,
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.max_concurrent_questions = max_concurrent_questions
        self.verbose = verbose

        # Keep config/tooling aligned with selected dataset.
        self._align_data_paths_with_questions()

        # Output files
        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.decompositions_file = self.output_dir / "decompositions.jsonl"
        self.autonomy_trace_file = self.output_dir / "autonomy_trace.jsonl"
        self.worker_messages_file = self.output_dir / "worker_messages.jsonl"
        self.claim_graph_file = self.output_dir / "claim_graph.jsonl"
        self.cache_analytics_file = self.output_dir / "cache_analytics.json"

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
        """Auto-align retrieval corpus paths to the selected question dataset.

        This avoids accidental cross-dataset runs (e.g., MuSiQue questions with
        Hotpot chunks/index).
        """
        self.config.set("data.questions_file", str(self.questions_file))

        dataset_key = self._infer_dataset_key(self.questions_file)
        if dataset_key is None:
            return

        dataset_dir = LOCAL_DATASET_DIRS[dataset_key]
        local_data_dir = PROJECT_ROOT / "data" / dataset_dir
        chunks_candidate = local_data_dir / "chunks.json"
        index_candidate = local_data_dir / "index_e5_base_v2"

        if not chunks_candidate.exists():
            logger.warning(
                "Could not align chunks path for dataset '%s' (missing %s)",
                dataset_key,
                chunks_candidate,
            )
            return

        configured_chunks = str(self.config.get("data.chunks_file", ""))
        configured_index = str(self.config.get("data.index_dir", ""))

        if configured_chunks != str(chunks_candidate):
            self.config.set("data.chunks_file", str(chunks_candidate))
            logger.info("Using chunks file: %s", chunks_candidate)

        if index_candidate.exists() and configured_index != str(index_candidate):
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

    async def _append_decomposition(self, decomposition: dict[str, Any]) -> None:
        async with self._write_lock:
            with open(self.decompositions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(decomposition, ensure_ascii=False, default=str) + "\n")

    async def _append_autonomy_trace(self, qid: Any, trace: list[dict[str, Any]]) -> None:
        if not trace:
            return
        payload = {"qid": qid, "trace": trace}
        async with self._write_lock:
            with open(self.autonomy_trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    async def _append_worker_messages(self, qid: Any, messages: list[dict[str, Any]]) -> None:
        if not messages:
            return
        payload = {"qid": qid, "messages": messages}
        async with self._write_lock:
            with open(self.worker_messages_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    async def _append_claim_graph(self, qid: Any, claim_graph: list[dict[str, Any]]) -> None:
        if not claim_graph:
            return
        payload = {"qid": qid, "claims": claim_graph}
        async with self._write_lock:
            with open(self.claim_graph_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _create_pipeline(self) -> MultiAgentPipeline:
        """Create a pipeline instance with shared LLM client and tools."""
        llm_cfg = self.config.get("llm", {})
        client = LLMClient(
            model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-30B-A3B"),
            api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
            base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            chat_template_kwargs=llm_cfg.get("chat_template_kwargs"),
        )

        # Build tools from data config
        data_cfg = self.config.get("data", {})
        tools = self._build_tools(data_cfg)

        ma_cfg = self.config.get("multi_agent", {}) or {}
        if ma_cfg.get("enable_m6_litcore", False):
            from multi_agent.m6_pipeline import M6LitCorePipeline
            return M6LitCorePipeline(
                llm_client=client,
                tools=tools,
                config=self.config,
            )

        if ma_cfg.get("enable_sage_v2", False):
            from multi_agent.sage_v2_pipeline import SageV2Pipeline
            return SageV2Pipeline(
                llm_client=client,
                tools=tools,
                config=self.config,
            )

        if ma_cfg.get("enable_sage", False):
            from multi_agent.sage_pipeline import SagePipeline
            return SagePipeline(
                llm_client=client,
                tools=tools,
                config=self.config,
            )

        return MultiAgentPipeline(
            llm_client=client,
            tools=tools,
            config=self.config,
        )

    def _build_tools(self, data_cfg: dict) -> Any:
        """Build tool registry from data config."""
        from arag.tools.registry import ToolRegistry
        from arag.tools.keyword_search import KeywordSearchTool
        from arag.tools.semantic_search import SemanticSearchTool
        from arag.tools.read_chunk import ReadChunkTool
        from arag.tools.finish import FinishTool

        chunks_file = data_cfg.get("chunks_file", "data/chunks.json")
        reg = ToolRegistry()
        reg.register(KeywordSearchTool(chunks_file=chunks_file))
        reg.register(ReadChunkTool(chunks_file=chunks_file))
        reg.register(FinishTool())

        index_dir = data_cfg.get("index_dir")
        if index_dir and Path(index_dir).exists():
            emb_cfg = self.config.get("embedding", {})
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
            logger.info("Embedding model loaded successfully!")
        else:
            logger.warning("Index dir not found at %s, semantic search disabled", index_dir)

        return reg

    def _result_to_prediction(
        self, item: dict[str, Any], result: PipelineResult
    ) -> dict[str, Any]:
        """Serialize PipelineResult to prediction dict."""
        qid = item.get("qid") or item.get("id")
        gold = item.get("answer", item.get("gold_answer", ""))

        agent_results_ser = {}
        for idx, ar in result.agent_results.items():
            agent_results_ser[str(idx)] = {
                "sub_question_index": ar.sub_question_index,
                "answer": ar.answer,
                "evidence_doc_ids": ar.evidence_doc_ids,
                "loops": ar.loops,
                "total_tokens": ar.total_tokens,
                "wall_clock_seconds": ar.wall_clock_seconds,
                "confidence": ar.confidence,
                "unsupported_answer": ar.unsupported_answer,
                "extracted_evidence": ar.extracted_evidence,
                "evidence_count": ar.evidence_count,
                "error": ar.error,
            }

        return {
            "qid": qid,
            "question": result.question,
            "gold_answer": gold,
            "pred_answer": result.final_answer,
            "question_type": result.question_type,
            "num_sub_questions": result.num_sub_questions,
            "num_waves": result.num_waves,
            "pass_id": result.pass_id,
            "retry_trigger_reasons": result.retry_trigger_reasons,
            "verifier_parse_ok": result.verifier_parse_ok,
            "agent_results": agent_results_ser,
            "cache_analytics": result.cache_analytics,
            "total_tokens": result.total_tokens,
            "aggregator_tokens": result.aggregator_tokens,
            "wall_clock_seconds": result.wall_clock_seconds,
            "manager_actions": result.manager_actions,
            "claim_graph": result.claim_graph,
            "autonomy_trace": result.autonomy_trace,
            "worker_messages": result.worker_messages,
            "error": result.error,
        }

    def _result_to_decomposition(
        self, item: dict[str, Any], result: PipelineResult
    ) -> dict[str, Any]:
        """Serialize decomposition plan."""
        qid = item.get("qid") or item.get("id")
        plan = result.decomposition
        if plan is None:
            return {"qid": qid, "question": result.question, "decomposition": None}

        return {
            "qid": qid,
            "question": result.question,
            "question_type": plan.question_type,
            "sub_questions": [
                {
                    "index": sq.index,
                    "text": sq.text,
                    "search_hints": sq.search_hints,
                    "depends_on": sq.depends_on,
                    "placeholder": sq.placeholder,
                }
                for sq in plan.sub_questions
            ],
            "dependency_edges": plan.dependency_edges,
            "parse_retries": plan.parse_retries,
        }

    async def _process_one(
        self,
        item: dict[str, Any],
        pipeline: MultiAgentPipeline,
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        """Process one question through the pipeline."""
        async with semaphore:
            qid = item.get("qid") or item.get("id")
            question = item.get("question", "")

            logger.info("Processing %s: %s", qid, question[:60])

            try:
                result = await pipeline.run(question)
            except Exception as exc:
                logger.error("Pipeline error for %s: %s", qid, exc)
                result = PipelineResult(
                    question=question,
                    final_answer=f"Error: {exc}",
                    error=str(exc),
                )

            prediction = self._result_to_prediction(item, result)
            decomposition = self._result_to_decomposition(item, result)

            await self._append_prediction(prediction)
            await self._append_decomposition(decomposition)
            await self._append_autonomy_trace(qid, result.autonomy_trace)
            await self._append_worker_messages(qid, result.worker_messages)
            await self._append_claim_graph(qid, result.claim_graph)

            logger.info(
                "Done %s: '%s' (%.1fs, %d tokens)",
                qid,
                result.final_answer[:40],
                result.wall_clock_seconds,
                result.total_tokens,
            )

            return prediction

    async def run(self) -> None:
        """Run the batch."""
        completed = self._load_completed_qids()
        pending = [
            q
            for q in self.questions
            if (q.get("qid") or q.get("id")) not in completed
        ]

        logger.info(
            "Total: %d | Completed: %d | Pending: %d",
            len(self.questions),
            len(completed),
            len(pending),
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
                    logger.info(
                        "Progress: %d/%d (%.0f Q/hr)",
                        completed_count,
                        len(pending),
                        rate,
                    )
            except Exception as exc:
                logger.error("Unhandled error: %s", exc)

        all_cache = [r.get("cache_analytics", {}) for r in results if r]
        if all_cache:
            agg_analytics = {
                "total_questions": len(results),
                "mean_doc_cache_hit_rate": sum(c.get("get_hit_rate", 0) for c in all_cache)
                / max(len(all_cache), 1),
                "total_cross_agent_reuses": sum(c.get("cross_agent_reuses", 0) for c in all_cache),
                "total_unique_docs": sum(c.get("unique_docs", 0) for c in all_cache),
                "mean_manager_actions": sum((r.get("manager_actions", 0) if r else 0) for r in results) / max(len(results), 1),
            }
            with open(self.cache_analytics_file, "w") as f:
                json.dump(agg_analytics, f, indent=2)

        elapsed = time.monotonic() - t0
        logger.info(
            "Batch complete: %d questions in %.1f min (%.1f Q/hr)",
            len(results),
            elapsed / 60,
            len(results) / elapsed * 3600 if elapsed > 0 else 0,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MA²RAG Batch Runner")
    parser.add_argument("--config", "-c", required=True, help="YAML config path")
    parser.add_argument("--questions", "-q", required=True, help="Questions JSON path")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--chunks-file", type=str, default=None, help="Override chunks file")
    parser.add_argument("--index-dir", type=str, default=None, help="Override semantic index dir")
    parser.add_argument(
        "--concurrent", type=int, default=5, help="Max concurrent questions"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.chunks_file:
        config.set("data.chunks_file", args.chunks_file)
    if args.index_dir:
        config.set("data.index_dir", args.index_dir)

    runner = MultiAgentBatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        max_concurrent_questions=args.concurrent,
        verbose=args.verbose,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
