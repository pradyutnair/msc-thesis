#!/usr/bin/env python3
"""Benchmark-capable E2 ReAct runner using the existing ARAG agent stack."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arag import BaseAgent, Config, LLMClient, ToolRegistry
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.semantic_search import SemanticSearchTool
from benchmarking.qa_benchmark import build_record, infer_dataset_name

logging.basicConfig(level=logging.ERROR)

CHUNK_ID_RE = re.compile(r"Chunk ID:\s*([^\s(]+)")


def parse_chunk_ids(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for chunk_id in CHUNK_ID_RE.findall(text or ""):
        chunk_id = str(chunk_id)
        if chunk_id not in seen:
            seen.add(chunk_id)
            out.append(chunk_id)
    return out


def extract_chunks_map(tools: ToolRegistry) -> dict[str, str]:
    for tool in getattr(tools, "_tools", {}).values():
        chunks = getattr(tool, "chunks", None)
        if not chunks:
            continue
        out: dict[str, str] = {}
        for raw_id, item in chunks.items():
            cid = str(raw_id)
            if isinstance(item, dict):
                out[cid] = str(item.get("text") or item.get("contents") or "")
            else:
                out[cid] = str(item)
        if out:
            return out
    return {}


class TimedLLMClient(LLMClient):
    def __init__(self, *args, stats: dict[str, Any], **kwargs):
        super().__init__(*args, **kwargs)
        self._stats = stats

    def chat(self, *args, **kwargs) -> dict[str, Any]:
        started = time.time()
        result = super().chat(*args, **kwargs)
        self._stats["elapsed_sec_llm"] += time.time() - started
        self._stats["llm_calls"] += 1
        return result


class TimedTool:
    def __init__(self, tool: Any, stats: dict[str, Any]):
        self._tool = tool
        self._stats = stats

    @property
    def name(self) -> str:
        return self._tool.name

    def get_schema(self) -> dict[str, Any]:
        return self._tool.get_schema()

    def execute(self, context, **kwargs):
        started = time.time()
        result, tool_log = self._tool.execute(context, **kwargs)
        self._stats["elapsed_sec_retrieval"] += time.time() - started
        self._stats["retrieval_calls"] += 1

        if self.name in {"semantic_search", "keyword_search"}:
            chunk_ids = parse_chunk_ids(result)
            if chunk_ids and not self._stats["initial_chunk_ids"]:
                self._stats["initial_chunk_ids"] = list(chunk_ids)
            self._stats["all_chunk_ids"].update(chunk_ids)
        elif self.name == "read_chunk":
            chunk_id = kwargs.get("chunk_id")
            if chunk_id is not None:
                self._stats["all_chunk_ids"].add(str(chunk_id))

        return result, tool_log


class BenchmarkBatchRunner:
    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int | None = None,
        num_workers: int = 1,
        verbose: bool = False,
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose
        self.dataset_name = infer_dataset_name(str(self.questions_file)) or infer_dataset_name(str(output_dir))

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.write_lock = Lock()

        self.questions = self._load_questions()
        self._shared_tools = self._init_shared_tools()
        self._chunks_map = extract_chunks_map(self._shared_tools)

        prompt_candidates = [
            Path("/projects/prjs1800/external/arag/src/arag/agent/prompts/default.txt"),
            Path(__file__).resolve().parents[3] / "external" / "arag" / "src" / "arag" / "agent" / "prompts" / "default.txt",
        ]
        prompt_file = next((path for path in prompt_candidates if path.exists()), None)
        self._system_prompt = prompt_file.read_text() if prompt_file else "You are a helpful assistant."

    def _load_questions(self) -> list[dict[str, Any]]:
        questions = json.loads(self.questions_file.read_text(encoding="utf-8"))
        return questions[: self.limit] if self.limit else questions

    def _load_completed_qids(self) -> set[str]:
        if not self.predictions_file.exists():
            return set()
        completed: set[str] = set()
        with self.predictions_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = row.get("qid")
                if qid:
                    completed.add(str(qid))
        return completed

    def _append_prediction(self, prediction: dict[str, Any]) -> None:
        with self.write_lock:
            with self.predictions_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    def _init_shared_tools(self) -> ToolRegistry:
        data_config = self.config.get("data", {})
        chunks_file = data_config.get("chunks_file", "data/chunks.json")
        index_dir = data_config.get("index_dir", "data/index")

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadChunkTool(chunks_file=chunks_file))

        index_file = Path(index_dir) / "sentence_index.pkl"
        if index_file.exists():
            embedding_config = self.config.get("embedding", {})
            tools.register(
                SemanticSearchTool(
                    chunks_file=chunks_file,
                    index_dir=index_dir,
                    model_name=embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                    device=embedding_config.get("device"),
                )
            )
        return tools

    def _create_agent(self, stats: dict[str, Any]) -> BaseAgent:
        llm_config = self.config.get("llm", {})
        llm = TimedLLMClient(
            model=llm_config.get("model") or os.getenv("ARAG_MODEL", "gpt-4o-mini"),
            api_key=llm_config.get("api_key") or os.getenv("ARAG_API_KEY"),
            base_url=llm_config.get("base_url") or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 1024),
            reasoning_effort=llm_config.get("reasoning_effort"),
            chat_template_kwargs=llm_config.get("chat_template_kwargs"),
            stats=stats,
        )

        wrapped_tools = ToolRegistry()
        for tool in getattr(self._shared_tools, "_tools", {}).values():
            wrapped_tools.register(TimedTool(tool, stats))

        agent_config = self.config.get("agent", {})
        return BaseAgent(
            llm_client=llm,
            tools=wrapped_tools,
            system_prompt=self._system_prompt,
            max_loops=agent_config.get("max_loops", 10),
            max_token_budget=agent_config.get("max_token_budget", 128000),
            verbose=self.verbose,
        )

    def _chunk_texts(self, chunk_ids: list[str]) -> list[str]:
        return [self._chunks_map[cid] for cid in chunk_ids if cid in self._chunks_map]

    def _process_one(self, item: dict[str, Any]) -> dict[str, Any]:
        qid = str(item.get("qid") or item.get("id"))
        question = item.get("question", "")
        gold = item.get("answer", item.get("gold_answer", ""))
        gold_answers = item.get("golden_answers") or ([gold] if gold else [""])
        stats = {
            "elapsed_sec_llm": 0.0,
            "elapsed_sec_retrieval": 0.0,
            "llm_calls": 0,
            "retrieval_calls": 0,
            "initial_chunk_ids": [],
            "all_chunk_ids": set(),
        }
        agent = self._create_agent(stats)
        started = time.time()
        try:
            result = agent.run(question)
            pred_answer = result["answer"]
            error = None
        except Exception as exc:  # pylint: disable=broad-except
            result = {
                "answer": f"Error: {exc}",
                "trajectory": [],
                "total_cost": 0.0,
                "loops": 0,
                "total_retrieved_tokens": 0,
                "retrieval_logs": [],
                "chunks_read_count": 0,
                "chunks_read_ids": [],
            }
            pred_answer = result["answer"]
            error = str(exc)
        elapsed_total = time.time() - started

        final_chunk_ids = sorted({*stats["all_chunk_ids"], *[str(cid) for cid in result.get("chunks_read_ids", [])]})
        c0_passages = self._chunk_texts(list(stats["initial_chunk_ids"]))
        final_passages = self._chunk_texts(final_chunk_ids)
        record = build_record(
            dataset=self.dataset_name,
            qid=qid,
            method="e2_react",
            model=self.config.get("llm", {}).get("model") or os.getenv("ARAG_MODEL", ""),
            question=question,
            gold_answers=gold_answers,
            pred_answer=pred_answer,
            elapsed_sec_total=elapsed_total,
            elapsed_sec_llm=stats["elapsed_sec_llm"],
            elapsed_sec_retrieval=stats["elapsed_sec_retrieval"],
            retrieval_calls=stats["retrieval_calls"],
            unique_chunks_read=len(final_chunk_ids),
            total_retrieved_tokens=result.get("total_retrieved_tokens", 0),
            loops_or_rounds=result.get("loops", 0),
            llm_calls=stats["llm_calls"],
            c0_passages=c0_passages,
            final_passages=final_passages,
            error=error,
            extra={
                "trajectory": result.get("trajectory", []),
                "total_cost": result.get("total_cost", 0.0),
                "retrieval_logs": result.get("retrieval_logs", []),
                "chunks_read_ids": final_chunk_ids,
            },
        )
        return record

    def run(self) -> None:
        completed_qids = self._load_completed_qids()
        pending = [item for item in self.questions if str(item.get("qid") or item.get("id")) not in completed_qids]

        print(f"Total questions: {len(self.questions)}")
        print(f"Completed: {len(completed_qids)}")
        print(f"Pending: {len(pending)}")
        if not pending:
            print("All questions completed.")
            return

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._process_one, item): item.get("qid") or item.get("id") for item in pending}
            with tqdm(total=len(pending), desc="E2 benchmark") as progress:
                for future in as_completed(futures):
                    _ = futures[future]
                    self._append_prediction(future.result())
                    progress.update(1)

        print(f"Saved predictions: {self.predictions_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-capable E2 ReAct runner")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--questions", "-q", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    runner = BenchmarkBatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
    )
    if args.overwrite and runner.predictions_file.exists():
        runner.predictions_file.unlink()
    runner.run()


if __name__ == "__main__":
    main()
