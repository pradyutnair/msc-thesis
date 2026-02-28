#!/usr/bin/env python3
"""M6 runner: M5 orchestrator + decomposer pre-step + verifier post-step."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

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
PROMPT_DIR = PROJECT_ROOT / "src" / "multi_agent" / "m5" / "prompts"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", str(text or "")).strip()


def _safe_parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^?:json)?\s\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


class M6BatchRunner:

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

        self._align_data_paths(chunks_file=chunks_file, index_dir=index_dir)
        self.questions = self._load_questions()
        self._shared_raw_tools = self._init_shared_raw_tools()

        self._system_prompt = (PROMPT_DIR / "orchestrator.txt").read_text(encoding="utf-8")
        self._decompose_prompt = (PROMPT_DIR / "decompose.txt").read_text(encoding="utf-8")
        self._verify_prompt = (PROMPT_DIR / "verify_answer.txt").read_text(encoding="utf-8")

    # ---- data path helpers (same as M5) ----

    def _align_data_paths(self, chunks_file: str | None, index_dir: str | None) -> None:
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
        LOCAL_DIRS = {"hotpotqa": "hotpotqa", "musique": "musique", "2wiki": "2wikimultihop"}
        local_data_dir = PROJECT_ROOT / "data" / LOCAL_DIRS[dataset_key]
        if chunks_file is None and (local_data_dir / "chunks.json").exists():
            self.config.set("data.chunks_file", str(local_data_dir / "chunks.json"))
        if index_dir is None and (local_data_dir / "index_e5_base_v2").exists():
            self.config.set("data.index_dir", str(local_data_dir / "index_e5_base_v2"))

    @staticmethod
    def _infer_dataset_key(path: Path) -> str | None:
        raw = str(path).lower()
        for key in ("hotpot", "musique", "2wiki"):
            if key in raw:
                return {"hotpot": "hotpotqa", "musique": "musique", "2wiki": "2wiki"}[key]
        return None

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
                chunks_file=chunks_file, index_dir=index_dir,
                model_name=model_name, device=emb_cfg.get("device"),
            )
            print("Embedding model loaded successfully!")
        return {"keyword": raw_keyword, "reader": raw_reader, "semantic": raw_semantic}

    def _load_questions(self) -> List[Dict[str, Any]]:
        with open(self.questions_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        return questions[: self.limit] if self.limit else questions

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

    def _append_prediction(self, prediction: Dict[str, Any]) -> None:
        with self.write_lock:
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    # ---- LLM + agent creation ----

    def _make_llm_client(self) -> LLMClient:
        llm_cfg = self.config.get("llm", {})
        return LLMClient(
            model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-30B-A3B"),
            api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
            base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 512),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            chat_template_kwargs=llm_cfg.get("chat_template_kwargs"),
        )

    def _create_agent(self, client: LLMClient) -> BaseAgent:
        sub_cfg = self.config.get("subagent", {}) or {}
        tools = ToolRegistry()
        tools.register(KeywordAgentTool(
            raw_tool=self._shared_raw_tools["keyword"], llm_client=client,
            prompt_path=str(PROMPT_DIR / "keyword_extract.txt"),
            max_tokens=int(sub_cfg.get("keyword_max_tokens", 64)),
        ))
        if self._shared_raw_tools["semantic"] is not None:
            tools.register(SemanticAgentTool(
                raw_tool=self._shared_raw_tools["semantic"], llm_client=client,
                prompt_path=str(PROMPT_DIR / "query_formulate.txt"),
                max_tokens=int(sub_cfg.get("semantic_max_tokens", 128)),
            ))
        tools.register(ChunkReaderAgentTool(
            raw_tool=self._shared_raw_tools["reader"], llm_client=client,
            prompt_path=str(PROMPT_DIR / "extract_evidence.txt"),
            max_tokens=int(sub_cfg.get("chunk_reader_max_tokens", 256)),
        ))
        tools.register(FinishTool())
        agent_cfg = self.config.get("agent", {})
        return BaseAgent(
            llm_client=client, tools=tools, system_prompt=self._system_prompt,
            max_loops=agent_cfg.get("max_loops", 15),
            max_token_budget=agent_cfg.get("max_token_budget", 128000),
            verbose=self.verbose,
        )

    # ---- decompose + verify (the new M6 pieces) ----

    def _decompose(self, client: LLMClient, question: str) -> List[str]:
        prompt = self._decompose_prompt.replace("{question}", question)
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=256,
        )
        raw = _strip_thinking(response["message"].get("content", ""))
        parsed = _safe_parse_json(raw)
        if parsed and isinstance(parsed.get("sub_questions"), list):
            subs = [str(s).strip() for s in parsed["sub_questions"] if str(s).strip()]
            if subs:
                return subs
        return [question]

    def _verify(
        self, client: LLMClient, question: str, answer: str, plan: List[str],
    ) -> Dict[str, Any]:
        prompt = (
            self._verify_prompt
            .replace("{question}", question)
            .replace("{answer}", answer)
            .replace("{plan}", json.dumps(plan))
        )
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=128,
        )
        raw = _strip_thinking(response["message"].get("content", ""))
        parsed = _safe_parse_json(raw)
        if parsed:
            return parsed
        return {"verdict": "accept", "reason": "parse_failed"}

    def _build_augmented_query(self, question: str, sub_questions: List[str]) -> str:
        if len(sub_questions) <= 1:
            return question
        plan_text = "\n".join(f"  Step {i+1}: {sq}" for i, sq in enumerate(sub_questions))
        return (
            f"{question}\n\n"
            f"[Plan — solve these sub-questions in order, then combine for the final answer:]\n"
            f"{plan_text}"
        )

    # ---- main processing ----

    def _process_one(self, item: Dict[str, Any]) -> Dict[str, Any]:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold_answer = item.get("answer", item.get("gold_answer", ""))

        client = self._make_llm_client()
        agent = self._create_agent(client)

        # Phase 1: Decompose
        sub_questions = self._decompose(client, question)
        if self.verbose:
            print(f"Decomposed into {len(sub_questions)} sub-questions: {sub_questions}")

        # Phase 2: Orchestrator (existing M5 loop with augmented query)
        augmented_query = self._build_augmented_query(question, sub_questions)
        result = agent.run(augmented_query)
        answer = result["answer"]

        # Phase 3: Verify
        verification = self._verify(client, question, answer, sub_questions)
        if verification.get("verdict") == "reject":
            fix = verification.get("suggested_fix", "").strip()
            if fix:
                if self.verbose:
                    print(f"Verifier rejected '{answer}', using fix: '{fix}'")
                answer = fix
            else:
                if self.verbose:
                    print(f"Verifier rejected '{answer}' but no fix suggested, keeping original")

        return {
            "qid": qid,
            "question": question,
            "trajectory": result["trajectory"],
            "gold_answer": gold_answer,
            "pred_answer": answer,
            "total_cost": result["total_cost"],
            "loops": result["loops"],
            "total_retrieved_tokens": result.get("total_retrieved_tokens", 0),
            "retrieval_logs": result.get("retrieval_logs", []),
            "chunks_read_count": result.get("chunks_read_count", 0),
            "chunks_read_ids": result.get("chunks_read_ids", []),
            "sub_questions": sub_questions,
            "verification": verification,
        }

    def run(self) -> None:
        completed_qids = self._load_completed_qids()
        pending = [q for q in self.questions if (q.get("qid") or q.get("id")) not in completed_qids]
        print(f"Total questions: {len(self.questions)}")
        print(f"Completed: {len(completed_qids)}")
        print(f"Pending: {len(pending)}")
        if not pending:
            print("All questions completed!")
            return
        print(f"Starting with {self.num_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(self._process_one, item): item.get("qid") or item.get("id") for item in pending}
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
    parser = argparse.ArgumentParser(description="M6 ARAG Batch Runner (decompose + M5 + verify)")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--questions", "-q", required=True)
    parser.add_argument("--chunks-file")
    parser.add_argument("--index-dir")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--workers", "-w", type=int, default=3)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    runner = M6BatchRunner(
        config=config, questions_file=args.questions, output_dir=args.output,
        limit=args.limit, num_workers=args.workers, verbose=args.verbose,
        chunks_file=args.chunks_file, index_dir=args.index_dir,
    )
    runner.run()


if __name__ == "__main__":
    main()