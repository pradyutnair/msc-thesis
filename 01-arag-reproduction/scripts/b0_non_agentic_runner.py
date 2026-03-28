#!/usr/bin/env python3
"""Non-agentic B0 runner for ARAG index.

Pipeline per question:
1) Embed question + retrieve top-k chunks via semantic search index.
2) Single LLM generation from retrieved context.

Outputs predictions.jsonl compatible with existing eval.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arag import Config, LLMClient
from arag.core.context import AgentContext
from arag.tools.semantic_search import SemanticSearchTool
from benchmarking.qa_benchmark import build_record, infer_dataset_name

CHUNK_ID_RE = re.compile(r"Chunk ID:\s*([^\s(]+)")
REASONING_TAG_RE = re.compile(r"<(think|thnk)>.*?</(think|thnk)>", re.IGNORECASE | re.DOTALL)

SYSTEM_PROMPT = (
    "You are a factual QA assistant. "
    "Answer strictly based on the provided context. "
    "Return only the final answer, concise, with no explanation."
)


def load_questions(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if limit is not None:
        data = data[:limit]
    return data


def strip_reasoning_tags(text: str) -> str:
    if not text:
        return text
    return REASONING_TAG_RE.sub("", text).strip()


def preflight_llm_endpoint(base_url: str, model: str) -> None:
    """Fail fast if target OpenAI-compatible endpoint is unavailable."""
    models_url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(models_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM endpoint not ready at {models_url}: {exc}") from exc

    ids = {str(m.get("id", "")) for m in payload.get("data", []) if isinstance(m, dict)}
    if model and model not in ids:
        raise RuntimeError(
            f"Model '{model}' not present at {models_url}. Available: {sorted(ids)}"
        )


def parse_chunk_ids(tool_result: str) -> List[str]:
    ids = CHUNK_ID_RE.findall(tool_result)
    seen: set[str] = set()
    out: List[str] = []
    for cid in ids:
        if cid not in seen:
            out.append(cid)
            seen.add(cid)
    return out


def build_context_text(chunk_ids: List[str], chunks_map: Dict[str, str]) -> str:
    parts: List[str] = []
    for cid in chunk_ids:
        text = chunks_map.get(cid, "")
        if text:
            parts.append(f"[Chunk {cid}]\n{text}")
    return "\n\n".join(parts)


def get_chunks_map(search_tool: SemanticSearchTool) -> Dict[str, str]:
    chunks_map: Dict[str, str] = {}
    for raw_id, item in search_tool.chunks.items():
        cid = str(raw_id)
        if isinstance(item, dict):
            chunks_map[cid] = str(item.get("text", ""))
        else:
            chunks_map[cid] = str(item)
    return chunks_map


def load_completed_qids(predictions_file: Path) -> set[str]:
    if not predictions_file.exists():
        return set()

    done: set[str] = set()
    with predictions_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            if qid:
                done.add(str(qid))
    return done


def _disable_reasoning_for_model(model_name: str) -> bool:
    m = (model_name or "").lower()
    return "qwen3" in m or "deepseek-r1" in m


def run(args: argparse.Namespace) -> None:
    cfg = Config.from_yaml(args.config)

    data_cfg = cfg.get("data", {})
    emb_cfg = cfg.get("embedding", {})
    llm_cfg = cfg.get("llm", {})

    questions = load_questions(Path(args.questions), args.limit)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "predictions.jsonl"

    if args.overwrite and pred_file.exists():
        pred_file.unlink()

    done_qids = load_completed_qids(pred_file)
    pending = [q for q in questions if str(q.get("qid") or q.get("id")) not in done_qids]

    print(f"Total questions: {len(questions)}")
    print(f"Completed: {len(done_qids)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All questions completed.")
        return

    search_tool = SemanticSearchTool(
        chunks_file=data_cfg.get("chunks_file"),
        index_dir=data_cfg.get("index_dir"),
        model_name=emb_cfg.get("model"),
        device=emb_cfg.get("device"),
    )
    chunks_map = get_chunks_map(search_tool)

    llm_model = os.getenv("ARAG_MODEL") or llm_cfg.get("model")
    llm_api_key = os.getenv("ARAG_API_KEY") or llm_cfg.get("api_key")
    llm_base_url = os.getenv("ARAG_BASE_URL") or llm_cfg.get("base_url")

    if not llm_base_url:
        raise RuntimeError("Missing LLM base URL (set ARAG_BASE_URL or llm.base_url in config).")
    if not llm_model:
        raise RuntimeError("Missing LLM model (set ARAG_MODEL or llm.model in config).")

    chat_kwargs = llm_cfg.get("chat_template_kwargs")
    if not isinstance(chat_kwargs, dict):
        chat_kwargs = {}
    if _disable_reasoning_for_model(llm_model):
        chat_kwargs["enable_thinking"] = False

    preflight_llm_endpoint(llm_base_url, llm_model)

    llm = LLMClient(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        temperature=llm_cfg.get("temperature", 0.0),
        max_tokens=llm_cfg.get("max_tokens", 128),
        reasoning_effort=llm_cfg.get("reasoning_effort"),
        chat_template_kwargs=chat_kwargs,
    )

    top_k = int(args.top_k)
    dataset_name = infer_dataset_name(args.questions) or infer_dataset_name(args.config)
    with pred_file.open("a", encoding="utf-8") as fout:
        for item in tqdm(pending, desc="B0 generation"):
            qid = str(item.get("qid") or item.get("id"))
            question = item.get("question", "")
            gold = item.get("answer", item.get("gold_answer", ""))
            gold_answers = item.get("golden_answers") or ([gold] if gold else [""])

            ctx = AgentContext()
            try:
                total_start = time.time()
                retrieval_start = time.time()
                tool_result, _ = search_tool.execute(ctx, query=question, top_k=top_k)
                retrieval_elapsed = time.time() - retrieval_start
                chunk_ids = parse_chunk_ids(tool_result)
                passages = [chunks_map.get(cid, "") for cid in chunk_ids if chunks_map.get(cid, "")]
                context_text = build_context_text(chunk_ids, chunks_map)

                user_prompt = (
                    f"Question:\n{question}\n\n"
                    f"Retrieved Context (top-{top_k}):\n{context_text}\n\n"
                    "Final answer:"
                )
                llm_start = time.time()
                pred, cost = llm.generate(
                    messages=[{"role": "user", "content": user_prompt}],
                    system=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_tokens=args.max_answer_tokens,
                )
                llm_elapsed = time.time() - llm_start
                pred = strip_reasoning_tags(str(pred))
                total_elapsed = time.time() - total_start

                row = {
                    "qid": qid,
                    "question": question,
                    "trajectory": [
                        {
                            "loop": 1,
                            "tool_name": "semantic_search",
                            "arguments": {"query": question, "top_k": top_k},
                            "tool_result": tool_result,
                            "retrieved_tokens": ctx.total_retrieved_tokens,
                            "chunks_found": len(chunk_ids),
                        }
                    ],
                    "gold_answer": gold,
                    "gold_answers": gold_answers,
                    "pred_answer": pred,
                    "total_cost": float(cost),
                    "loops": 1,
                    "finish_called": True,
                    "total_retrieved_tokens": int(ctx.total_retrieved_tokens),
                    "retrieval_logs": [
                        {
                            "tool_name": log.tool_name,
                            "tokens": log.tokens,
                            "metadata": log.metadata,
                        }
                        for log in ctx.retrieval_logs
                    ],
                    "chunks_read_count": len(chunk_ids),
                    "chunks_read_ids": chunk_ids,
                    **build_record(
                        dataset=dataset_name,
                        qid=qid,
                        method="b0",
                        model=llm_model,
                        question=question,
                        gold_answers=gold_answers,
                        pred_answer=pred,
                        elapsed_sec_total=total_elapsed,
                        elapsed_sec_llm=llm_elapsed,
                        elapsed_sec_retrieval=retrieval_elapsed,
                        retrieval_calls=1,
                        unique_chunks_read=len(chunk_ids),
                        total_retrieved_tokens=int(ctx.total_retrieved_tokens),
                        loops_or_rounds=1,
                        llm_calls=1,
                        c0_passages=passages,
                        final_passages=passages,
                    ),
                }
            except Exception as e:  # pylint: disable=broad-except
                total_elapsed = time.time() - total_start if "total_start" in locals() else 0.0
                retrieval_elapsed = time.time() - retrieval_start if "retrieval_start" in locals() else 0.0
                row = {
                    "qid": qid,
                    "question": question,
                    "trajectory": [],
                    "gold_answer": gold,
                    "gold_answers": gold_answers,
                    "pred_answer": f"Error: {e}",
                    "total_cost": 0.0,
                    "loops": 1,
                    "finish_called": False,
                    "total_retrieved_tokens": int(ctx.total_retrieved_tokens),
                    "retrieval_logs": [
                        {
                            "tool_name": log.tool_name,
                            "tokens": log.tokens,
                            "metadata": log.metadata,
                        }
                        for log in ctx.retrieval_logs
                    ],
                    "chunks_read_count": 0,
                    "chunks_read_ids": [],
                    "error": str(e),
                    **build_record(
                        dataset=dataset_name,
                        qid=qid,
                        method="b0",
                        model=llm_model,
                        question=question,
                        gold_answers=gold_answers,
                        pred_answer=f"Error: {e}",
                        elapsed_sec_total=total_elapsed,
                        elapsed_sec_llm=0.0,
                        elapsed_sec_retrieval=retrieval_elapsed,
                        retrieval_calls=max(1, len(ctx.retrieval_logs)),
                        unique_chunks_read=0,
                        total_retrieved_tokens=int(ctx.total_retrieved_tokens),
                        loops_or_rounds=1,
                        llm_calls=0,
                        c0_passages=[],
                        final_passages=[],
                        error=str(e),
                    ),
                }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"Saved predictions: {pred_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B0 non-agentic ARAG runner")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--questions", required=True, help="questions.json path")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--top-k", type=int, default=5, help="retrieval top-k")
    parser.add_argument("--limit", type=int, default=None, help="optional limit")
    parser.add_argument("--max-answer-tokens", type=int, default=128, help="max generation tokens")
    parser.add_argument("--overwrite", action="store_true", help="remove existing predictions.jsonl before run")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
