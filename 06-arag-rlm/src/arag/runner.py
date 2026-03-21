"""Batch runner for the RLM-style ARAG agent."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from arag.agent import RLMAgent
from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.build_tools import build_tool_bundle


class BatchRunner:
    """Run RLMAgent over a question set with checkpoint resume."""

    def __init__(
        self,
        config: Config,
        questions_file: str | None = None,
        output_dir: str | None = None,
        limit: int | None = None,
    ):
        self.config = config
        data_cfg = config.get("data", {})
        output_cfg = config.get("output", {})

        self.questions_file = Path(questions_file or data_cfg.get("questions_file"))
        self.output_dir = Path(output_dir or output_cfg.get("results_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self._write_lock = Lock()
        self.questions = self._load_questions()
        self.shared_bundle = build_tool_bundle(config)

    def _load_questions(self) -> List[Dict[str, Any]]:
        with open(self.questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        if self.limit is not None:
            questions = questions[: self.limit]
        return questions

    def _load_completed_qids(self) -> set[str]:
        completed: set[str] = set()
        if not self.predictions_file.exists():
            return completed
        with open(self.predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = item.get("qid") or item.get("id")
                if qid is not None:
                    completed.add(str(qid))
        return completed

    def _append_prediction(self, prediction: Dict[str, Any]) -> None:
        with self._write_lock:
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False, default=str) + "\n")

    def _create_llm(self) -> LLMClient:
        llm_cfg = self.config.get("llm", {})
        return LLMClient(
            model=os.getenv("ARAG_MODEL") or llm_cfg.get("model", "Qwen/Qwen3-8B"),
            api_key=os.getenv("ARAG_API_KEY") or llm_cfg.get("api_key", "dummy"),
            base_url=os.getenv("ARAG_BASE_URL") or llm_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            chat_template_kwargs=llm_cfg.get("chat_template_kwargs"),
        )

    def _create_agent(self) -> RLMAgent:
        data_cfg = self.config.get("data", {})
        emb_cfg = self.config.get("embedding", {})
        agent_cfg = self.config.get("agent", {})
        return RLMAgent(
            llm_client=self._create_llm(),
            chunks_dict=self.shared_bundle.get("chunks_dict"),
            embedding_index=self.shared_bundle.get("semantic_tool"),
            chunks_file=data_cfg.get("chunks_file"),
            index_dir=data_cfg.get("index_dir"),
            embedding_model=emb_cfg.get("model"),
            config=self.config,
            depth=0,
            max_depth=agent_cfg.get("max_depth", 1),
            parent_question=None,
            expected_answer_type=None,
            token_budget=agent_cfg.get("max_token_budget", 50000),
            tool_bundle=self.shared_bundle,
        )

    def _process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold_answer = item.get("answer", item.get("gold_answer", ""))
        agent = self._create_agent()
        try:
            result = agent.run(question)
            return {
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": result["answer"],
                "trajectory": result["trajectory"],
                "tokens_used": result.get("tokens_used", 0),
                "total_cost": result.get("total_cost", 0.0),
                "loops": result.get("loops", 0),
                "depth": result.get("depth", 0),
                "delegations_made": result.get("delegations_made", 0),
                "evidence_preview": result.get("evidence_preview", ""),
                "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
                "total_llm_tokens": result.get("total_llm_tokens", 0),
                "retrieval_logs": result.get("retrieval_logs", []),
                "llm_usage_logs": result.get("llm_usage_logs", []),
                "chunks_read_count": result.get("chunks_read_count", 0),
                "chunks_read_ids": result.get("chunks_read_ids", []),
            }
        except Exception as exc:
            return {
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": f"Error: {exc}",
                "trajectory": [],
                "tokens_used": 0,
                "total_cost": 0.0,
                "loops": 0,
                "depth": 0,
                "delegations_made": 0,
                "evidence_preview": "",
                "total_retrieved_tokens": 0,
                "total_llm_tokens": 0,
                "retrieval_logs": [],
                "llm_usage_logs": [],
                "chunks_read_count": 0,
                "chunks_read_ids": [],
                "error": str(exc),
            }

    def run(self) -> Path:
        completed = self._load_completed_qids()
        pending = [
            item for item in self.questions
            if str(item.get("qid") or item.get("id")) not in completed
        ]
        max_workers = int(self.config.get("runner.max_workers", 4))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_one, item) for item in pending]
            for future in as_completed(futures):
                self._append_prediction(future.result())

        return self.predictions_file
