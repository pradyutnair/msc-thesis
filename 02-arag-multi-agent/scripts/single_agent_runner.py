#!/usr/bin/env python3
"""Single-agent ReAct baseline for multi-hop QA.

Fair comparison to M6 multi-agent: same tools, same model, same corpus,
same warm-start, same max_steps budget. One agent, no decomposition.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
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
from arag.tools.read_chunk import ReadChunkTool
from arag.core.context import AgentContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research agent answering a question using search tools.

## Available Tools
- **search_and_read**: Search AND read top chunks in ONE step. Use `method="keyword"` for entity names, `method="semantic"` for conceptual queries. This is your primary tool.
- **keyword_search**: Find chunks by exact keyword matching (returns snippets only).
- **semantic_search**: Find chunks by semantic similarity (returns snippets only).
- **read_chunk**: Read full content of specific chunks by ID.

## Warm-Start Context (initial retrieval)
{warm_start}

## Strategy
1. Start with search_and_read using key entity names from the question.
2. Check the warm-start context above — it may already contain relevant information.
3. If keyword search finds nothing, try search_and_read with method="semantic".
4. For multi-hop questions, solve step by step: find the first entity, then use it to search for the next.
5. Verify evidence mentions the EXACT entity from the question (watch for similar names).
6. Try at least 2 different searches before giving up.

## Answer Format
When you have sufficient evidence, respond with ONLY the answer — a single entity name, date, number, or place.

CORRECT: "Steven Spielberg", "1986", "yes"
WRONG: "The answer is Steven Spielberg"

Just the entity. Nothing else. If you cannot find the answer: unknown"""


def build_tools(config: Config) -> ToolRegistry:
    data_cfg = config.get("data", {})
    chunks_file = data_cfg.get("chunks_file", "data/chunks.json")

    reg = ToolRegistry()
    keyword_tool = KeywordSearchTool(chunks_file=chunks_file)
    reg.register(keyword_tool)
    read_tool = ReadChunkTool(chunks_file=chunks_file)
    reg.register(read_tool)

    semantic_tool = None
    index_dir = data_cfg.get("index_dir")
    if index_dir and Path(index_dir).exists():
        from arag.tools.semantic_search import SemanticSearchTool
        emb_cfg = config.get("embedding", {})
        semantic_tool = SemanticSearchTool(
            chunks_file=chunks_file,
            index_dir=index_dir,
            model_name=emb_cfg.get("model", "intfloat/e5-base-v2"),
            device=emb_cfg.get("device"),
        )
        reg.register(semantic_tool)
        logger.info("Semantic search loaded.")

    from arag.tools.search_and_read import SearchAndReadTool
    reg.register(SearchAndReadTool(keyword_tool, semantic_tool, read_tool))

    return reg


def warm_start(tools: ToolRegistry, question: str) -> str:
    """Keyword search on full question for initial context."""
    keyword_tool = tools.get("keyword_search")
    if keyword_tool is None or not hasattr(keyword_tool, "chunks"):
        return "No warm-start context available."
    keywords = [w for w in question.replace("?", "").split() if len(w) > 2]
    if not keywords:
        return "No warm-start context available."
    scored = []
    for chunk in keyword_tool.chunks:
        text_lower = chunk["text"].lower()
        score = sum(text_lower.count(kw.lower()) * len(kw) for kw in keywords)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "No warm-start context available."
    lines = []
    for _, chunk in scored[:3]:
        cid = chunk.get("id", "?")
        text = chunk.get("text", "")[:400]
        lines.append(f"[{cid}] {text}")
    return "\n".join(lines)


def clean_answer(answer: str) -> str:
    answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
    answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
    answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)
    answer = re.sub(r"\*(.+?)\*", r"\1", answer)
    answer = answer.split("\n")[0].strip()
    answer = answer.strip().strip("\"'`*")
    answer = re.sub(r"\s*[\.,;:!?]+$", "", answer)
    return answer


def solve_question(
    question: str,
    llm: LLMClient,
    tools: ToolRegistry,
    max_steps: int = 16,
) -> dict[str, Any]:
    """Single-agent ReAct loop. Returns prediction dict."""
    t0 = time.monotonic()
    context = AgentContext()
    tool_schemas = tools.get_all_schemas()

    warm = warm_start(tools, question)
    system = SYSTEM_PROMPT.replace("{warm_start}", warm)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    total_tokens = 0
    answer = ""
    steps_used = 0

    for step in range(1, max_steps + 1):
        steps_used = step
        response = llm.chat(messages=messages, tools=tool_schemas, temperature=0.0)
        total_tokens += response.get("input_tokens", 0) + response.get("output_tokens", 0)

        assistant_msg = response["message"]
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls", [])

        if tool_calls:
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                tool_result, _ = tools.execute(func_name, context, **func_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })
        else:
            answer = assistant_msg.get("content", "")
            break

    answer = clean_answer(answer)

    # Clear verbose refusals
    if len(answer) > 60:
        refusal_patterns = [
            "the evidence does not", "does not mention", "not explicitly mentioned",
            "no evidence confirms", "not specified in", "the provided documents",
            "there is no ", "cannot be determined", "not found in the",
        ]
        if any(p in answer.lower() for p in refusal_patterns):
            answer = ""

    # LLM fallback if no answer after all steps
    if not answer or answer.lower() in ("unknown", "error", ""):
        # Extract from conversation: find last tool results
        tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
        if tool_contents:
            ctx = "\n---\n".join(tool_contents[-3:])[:1500]
            try:
                fb = llm.chat(
                    messages=[
                        {"role": "system", "content": "Extract the answer from the context. Reply with ONLY the answer entity."},
                        {"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx}"},
                    ],
                    tools=None, temperature=0.0,
                )
                fb_answer = clean_answer(fb["message"].get("content", ""))
                if fb_answer and len(fb_answer) < 200:
                    answer = fb_answer
                    total_tokens += fb.get("input_tokens", 0) + fb.get("output_tokens", 0)
            except Exception:
                pass

    elapsed = time.monotonic() - t0
    return {
        "pred_answer": answer,
        "total_tokens": total_tokens,
        "steps_used": steps_used,
        "wall_clock_seconds": elapsed,
    }


class SingleAgentRunner:
    def __init__(self, config, questions_file, output_dir, limit=None, offset=0, max_concurrent=5):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.max_concurrent = max_concurrent
        self._write_lock = asyncio.Lock()

        with open(questions_file) as f:
            questions = json.load(f)
        questions = questions[offset:]
        if limit:
            questions = questions[:limit]
        self.questions = questions

    def _load_completed_qids(self) -> set:
        completed = set()
        if not self.predictions_file.exists():
            return completed
        with open(self.predictions_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.add(json.loads(line).get("qid"))
                except json.JSONDecodeError:
                    continue
        return completed

    async def _append(self, pred):
        async with self._write_lock:
            with open(self.predictions_file, "a") as f:
                f.write(json.dumps(pred, ensure_ascii=False, default=str) + "\n")

    async def _process_one(self, item, llm, tools, sem):
        async with sem:
            qid = item.get("qid") or item.get("id")
            question = item["question"]
            gold = item.get("answer", item.get("gold_answer", ""))
            loop = asyncio.get_running_loop()

            try:
                result = await loop.run_in_executor(
                    None, solve_question, question, llm, tools, 16,
                )
            except Exception as exc:
                logger.error("Error %s: %s", qid, exc)
                result = {"pred_answer": f"Error: {exc}", "total_tokens": 0,
                          "steps_used": 0, "wall_clock_seconds": 0}

            pred = {
                "qid": qid,
                "question": question,
                "gold_answer": gold,
                "pred_answer": result["pred_answer"],
                "total_tokens": result["total_tokens"],
                "steps_used": result["steps_used"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            }
            await self._append(pred)
            logger.info("Done %s: '%s' (%d steps, %d tok, %.1fs)",
                        qid, result["pred_answer"][:40], result["steps_used"],
                        result["total_tokens"], result["wall_clock_seconds"])
            return pred

    async def run(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=256))

        completed = self._load_completed_qids()
        pending = [q for q in self.questions if (q.get("qid") or q.get("id")) not in completed]
        logger.info("Total: %d | Completed: %d | Pending: %d",
                     len(self.questions), len(completed), len(pending))
        if not pending:
            return

        llm_cfg = self.config.get("llm", {})
        llm = LLMClient(
            model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-8B"),
            api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
            base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
            temperature=0.0,
            max_tokens=8192,
            chat_template_kwargs={"enable_thinking": True},
        )
        tools = build_tools(self.config)
        sem = asyncio.Semaphore(self.max_concurrent)

        t0 = time.monotonic()
        tasks = [self._process_one(item, llm, tools, sem) for item in pending]
        done = 0
        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            if done % 10 == 0:
                rate = done / (time.monotonic() - t0) * 3600
                logger.info("Progress: %d/%d (%.0f Q/hr)", done, len(pending), rate)

        elapsed = time.monotonic() - t0
        logger.info("Done: %d questions in %.1f min (%.1f Q/hr)",
                     done, elapsed / 60, done / elapsed * 3600 if elapsed > 0 else 0)


DATASET_ALIASES = {
    "hotpotqa": "hotpotqa", "2wiki": "2wikimultihop",
    "2wikimultihop": "2wikimultihop", "musique": "musique",
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Single-Agent ReAct Runner")
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

    runner = SingleAgentRunner(
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
