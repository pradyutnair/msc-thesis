#!/usr/bin/env python3
"""Batch runner for M5 orchestrator + subagent tools."""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from tqdm import tqdm

from arag import BaseAgent, Config, LLMClient, ToolRegistry
from arag.tools.finish import FinishTool
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_chunk import ReadChunkTool
from arag.tools.semantic_search import SemanticSearchTool
from multi_agent.m5.subagent_tools import (
    ChunkReaderAgentTool,
    KeywordAgentTool,
    SemanticAgentTool,
)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATASET_DIRS = {
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "2wiki": "2wikimultihop",
}


class M5BatchRunner:
    """Batch runner with checkpoint resume and concurrent execution."""

    def __init__(
        self,
        config: Config,
        questions_file: str,
        output_dir: str,
        limit: int | None = None,
        num_workers: int = 3,
        verbose: bool = False,
        chunks_file: str | None = None,
        index_dir: str | None = None,
    ):
        self.config = config
        self.questions_file = Path(questions_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.write_lock = Lock()

        self._align_data_paths_with_questions(chunks_file=chunks_file, index_dir=index_dir)

        self.questions = self._load_questions()
        self._shared_raw_tools = self._init_shared_raw_tools()

        prompt_file = PROJECT_ROOT / "src" / "multi_agent" / "m5" / "prompts" / "orchestrator.txt"
        self._system_prompt = prompt_file.read_text(encoding="utf-8")

    def _infer_dataset_key(self, path: Path) -> str | None:
        raw = str(path).lower()
        if "hotpot" in raw:
            return "hotpotqa"
        if "musique" in raw:
            return "musique"
        if "2wiki" in raw:
            return "2wiki"
        return None

    def _align_data_paths_with_questions(
        self,
        chunks_file: str | None,
        index_dir: str | None,
    ) -> None:
        self.config.set("data.questions_file", str(self.questions_file))

        if chunks_file:
            self.config.set("data.chunks_file", str(Path(chunks_file)))
        if index_dir:
            self.config.set("data.index_dir", str(Path(index_dir)))
        if chunks_file and index_dir:
            return

        dataset_key = self._infer_dataset_key(self.questions_file)
        if dataset_key is None:
            return

        dataset_dir = LOCAL_DATASET_DIRS[dataset_key]
        local_data_dir = PROJECT_ROOT / "data" / dataset_dir
        chunks_candidate = local_data_dir / "chunks.json"
        index_candidate = local_data_dir / "index_e5_base_v2"

        if chunks_file is None and chunks_candidate.exists():
            self.config.set("data.chunks_file", str(chunks_candidate))

        if index_dir is None and index_candidate.exists():
            self.config.set("data.index_dir", str(index_candidate))

    def _init_shared_raw_tools(self) -> Dict[str, Any]:
        data_cfg = self.config.get("data", {})
        chunks_file = data_cfg.get("chunks_file", "data/chunks.json")
        index_dir = data_cfg.get("index_dir", "data/index")

        raw_keyword = KeywordSearchTool(chunks_file=chunks_file)
        raw_reader = ReadChunkTool(chunks_file=chunks_file)

        raw_semantic = None
        index_file = Path(index_dir) / "sentence_index.pkl"
        if index_file.exists():
            emb_cfg = self.config.get("embedding", {})
            model_name = emb_cfg.get("model", "intfloat/e5-base-v2")
            print(f"Loading embedding model: {model_name}")
            raw_semantic = SemanticSearchTool(
                chunks_file=chunks_file,
                index_dir=index_dir,
                model_name=model_name,
                device=emb_cfg.get("device"),
            )
            print("Embedding model loaded successfully!")
        else:
            print(f"Warning: Index not found at {index_file}, semantic search disabled")

        return {
            "keyword": raw_keyword,
            "reader": raw_reader,
            "semantic": raw_semantic,
        }

    def _load_questions(self) -> List[Dict[str, Any]]:
        with open(self.questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        if self.limit:
            questions = questions[: self.limit]
        return questions

    def _load_completed_qids(self) -> set:
        completed_qids = set()
        if not self.predictions_file.exists():
            return completed_qids

        try:
            with open(self.predictions_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = data.get("qid") or data.get("id")
                    if qid is not None:
                        completed_qids.add(qid)
        except Exception as exc:
            print(f"Warning: failed loading checkpoint: {exc}")

        return completed_qids

    def _append_prediction(self, prediction: Dict[str, Any]) -> None:
        with self.write_lock:
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    def _create_agent(self) -> BaseAgent:
        llm_cfg = self.config.get("llm", {})
        client = LLMClient(
            model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-30B-A3B"),
            api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
            base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 512),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            chat_template_kwargs=llm_cfg.get("chat_template_kwargs"),
        )

        sub_cfg = self.config.get("subagent", {}) or {}
        prompt_dir = PROJECT_ROOT / "src" / "multi_agent" / "m5" / "prompts"

        tools = ToolRegistry()
        tools.register(
            KeywordAgentTool(
                raw_tool=self._shared_raw_tools["keyword"],
                llm_client=client,
                prompt_path=str(prompt_dir / "keyword_extract.txt"),
                max_tokens=int(sub_cfg.get("keyword_max_tokens", 64)),
            )
        )

        if self._shared_raw_tools["semantic"] is not None:
            tools.register(
                SemanticAgentTool(
                    raw_tool=self._shared_raw_tools["semantic"],
                    llm_client=client,
                    prompt_path=str(prompt_dir / "query_formulate.txt"),
                    max_tokens=int(sub_cfg.get("semantic_max_tokens", 128)),
                )
            )

        tools.register(
            ChunkReaderAgentTool(
                raw_tool=self._shared_raw_tools["reader"],
                llm_client=client,
                prompt_path=str(prompt_dir / "extract_evidence.txt"),
                max_tokens=int(sub_cfg.get("chunk_reader_max_tokens", 256)),
            )
        )
        tools.register(FinishTool())

        agent_cfg = self.config.get("agent", {})
        return BaseAgent(
            llm_client=client,
            tools=tools,
            system_prompt=self._system_prompt,
            max_loops=agent_cfg.get("max_loops", 15),
            max_token_budget=agent_cfg.get("max_token_budget", 128000),
            verbose=self.verbose,
        )

    def _process_one(self, item: Dict[str, Any], agent: BaseAgent) -> Dict[str, Any]:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold_answer = item.get("answer", item.get("gold_answer", ""))

        try:
            result = agent.run(question)
            return {
                "qid": qid,
                "question": question,
                "trajectory": result["trajectory"],
                "gold_answer": gold_answer,
                "pred_answer": result["answer"],
                "total_cost": result["total_cost"],
                "loops": result["loops"],
                "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
                "retrieval_logs": result.get("retrieval_logs", []),
                "chunks_read_count": result.get("chunks_read_count", 0),
                "chunks_read_ids": result.get("chunks_read_ids", []),
            }
        except Exception as exc:
            return {
                "qid": qid,
                "question": question,
                "trajectory": [],
                "gold_answer": gold_answer,
                "pred_answer": f"Error: {str(exc)}",
                "total_cost": 0,
                "loops": 0,
                "total_retrieved_tokens": 0,
                "retrieval_logs": [],
                "chunks_read_count": 0,
                "chunks_read_ids": [],
                "error": str(exc),
            }

    def run(self) -> None:
        completed_qids = self._load_completed_qids()
        pending = [
            q
            for q in self.questions
            if (q.get("qid") or q.get("id")) not in completed_qids
        ]

        print(f"Total questions: {len(self.questions)}")
        print(f"Completed: {len(completed_qids)}")
        print(f"Pending: {len(pending)}")

        if not pending:
            print("All questions completed!")
            return

        print(f"Starting with {self.num_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for item in pending:
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent)
                futures[future] = item.get("qid") or item.get("id")

            with tqdm(total=len(pending), desc="Processing") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as exc:
                        print(f"Error processing {qid}: {exc}")
                    pbar.update(1)

        print(f"\nResults saved to: {self.predictions_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="M5 ARAG Batch Runner")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--questions", "-q", required=True, help="Questions JSON path")
    parser.add_argument("--chunks-file", help="Override chunks file path")
    parser.add_argument("--index-dir", help="Override index directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Question limit")
    parser.add_argument("--workers", "-w", type=int, default=3, help="Worker threads")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    runner = M5BatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
        chunks_file=args.chunks_file,
        index_dir=args.index_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()
