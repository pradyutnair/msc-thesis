#!/usr/bin/env python3
"""Collect counterfactual escalation trajectories.

For each question:
  1. Decompose deterministically (temperature=0)
  2. Resolve sub-questions in dependency order
  3. For each sub-question, run BOTH structured and agentic workers
  4. Record per-SQ counterfactual pairs: did structured suffice, or was agentic needed?

This produces training data for the escalation agent.

Usage:
    PYTHONPATH=src python -u scripts/collect_escalation_trajectories.py \
        --config configs/adaptive_agency.yaml \
        --questions data/hotpotqa/questions.json \
        --output trajectories/escalation_hotpotqa \
        --limit 100 --concurrent 20
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arag.core.config import Config
from arag.core.llm import LLMClient
from arag.tools.registry import ToolRegistry
from arag.tools.build_tools import build_tools
from multi_agent.blackboard import Blackboard
from multi_agent.planner_agent import PlannerAgent
from multi_agent.types import (
    EvidenceEntry,
    M6SubQuestion,
    RetrievalMode,
    SubQuestionStatus,
)
from multi_agent.utils import resolve_placeholders
from multi_agent.workers.agentic_worker import AgenticWorker
from multi_agent.workers.structured_worker import StructuredWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATASET_DIRS = {"hotpotqa": "hotpotqa", "musique": "musique", "2wiki": "2wikimultihop"}


def normalize(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def compute_em(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def compute_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)



def create_llm_client(config: Config, temperature: float = 0.0) -> LLMClient:
    llm_cfg = config.get("llm", {})
    return LLMClient(
        model=llm_cfg.get("model") or os.getenv("ARAG_MODEL", "Qwen3-8B"),
        api_key=llm_cfg.get("api_key") or os.getenv("ARAG_API_KEY", "dummy"),
        base_url=llm_cfg.get("base_url") or os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
        temperature=temperature,
        max_tokens=llm_cfg.get("max_tokens", 8192),
    )


def align_data_paths(config: Config, questions_file: str) -> None:
    raw = questions_file.lower()
    for key, dirname in LOCAL_DATASET_DIRS.items():
        if key in raw or dirname in raw:
            data_dir = PROJECT_ROOT / "data" / dirname
            chunks = data_dir / "chunks.json"
            index = data_dir / "index_e5_base_v2"
            if chunks.exists() and not config.get("data.chunks_file"):
                config.set("data.chunks_file", str(chunks))
            if index.exists() and not config.get("data.index_dir"):
                config.set("data.index_dir", str(index))
            break


async def run_structured_on_sq(
    llm_client: LLMClient,
    tools: ToolRegistry,
    sq: M6SubQuestion,
    question: str,
    entity_registry: dict[str, str],
    config: Config,
) -> dict[str, Any]:
    """Run structured worker on a single sub-question, return result dict."""
    adaptive_cfg = config.get("adaptive", {})
    structured_cfg = adaptive_cfg.get("structured", {})

    bb = Blackboard(question=question, token_budget=300_000)
    # Set up a minimal search plan with just this SQ
    sq_copy = M6SubQuestion(
        id=sq.id, text=sq.text, dependencies=[],
        known_entities=list(sq.known_entities),
        unknown_entities=list(sq.unknown_entities),
        search_hints=list(sq.search_hints),
        search_queries=list(sq.search_queries),
        mode=RetrievalMode.STRUCTURED,
        status=SubQuestionStatus.READY,
    )
    await bb.set_search_plan([sq_copy])
    # Inject entity registry from prior resolved SQs
    from multi_agent.types import EntityEntry
    for name, value in entity_registry.items():
        await bb.post_entity(EntityEntry(name=name, value=value, source_evidence_id="prior"))

    worker = StructuredWorker(
        agent_id=f"structured_{sq.id}",
        llm_client=llm_client,
        base_tools=tools,
        assigned_sq_id=sq.id,
        retrieval_top_k=structured_cfg.get("retrieval_top_k", 10),
        max_queries_per_entity=structured_cfg.get("max_queries_per_entity", 6),
        retry_low_confidence=structured_cfg.get("retry_low_confidence", True),
        confidence_threshold=structured_cfg.get("confidence_threshold", 0.6),
    )

    t0 = time.monotonic()
    try:
        tokens = await asyncio.wait_for(worker.tick(bb), timeout=180.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Structured worker timeout/error on SQ-%d: %s", sq.id, e)
        return {
            "answer": "", "tokens": 0, "evidence_count": 0,
            "wall_clock": time.monotonic() - t0, "error": str(e),
        }

    snapshot = await bb.get_snapshot()
    sq_data = snapshot["sub_questions"][0] if snapshot["sub_questions"] else {}
    answer = sq_data.get("answer", "") or ""
    evidence_count = len([e for e in snapshot["evidence"] if e["sub_question_id"] == sq.id])

    return {
        "answer": answer,
        "tokens": snapshot["tokens_used"],
        "evidence_count": evidence_count,
        "wall_clock": time.monotonic() - t0,
        "status": sq_data.get("status", "unknown"),
    }


async def run_agentic_on_sq(
    llm_client: LLMClient,
    tools: ToolRegistry,
    sq: M6SubQuestion,
    question: str,
    entity_registry: dict[str, str],
    config: Config,
) -> dict[str, Any]:
    """Run agentic worker on a single sub-question, return result dict."""
    adaptive_cfg = config.get("adaptive", {})

    bb = Blackboard(question=question, token_budget=300_000)
    sq_copy = M6SubQuestion(
        id=sq.id, text=sq.text, dependencies=[],
        known_entities=list(sq.known_entities),
        unknown_entities=list(sq.unknown_entities),
        search_hints=list(sq.search_hints),
        search_queries=list(sq.search_queries),
        mode=RetrievalMode.AGENTIC,
        status=SubQuestionStatus.READY,
    )
    await bb.set_search_plan([sq_copy])
    from multi_agent.types import EntityEntry
    for name, value in entity_registry.items():
        await bb.post_entity(EntityEntry(name=name, value=value, source_evidence_id="prior"))

    worker = AgenticWorker(
        agent_id=f"agentic_{sq.id}",
        llm_client=llm_client,
        tools=tools,
        assigned_sq_id=sq.id,
        max_steps=adaptive_cfg.get("worker_max_steps", 16),
    )

    t0 = time.monotonic()
    try:
        tokens = await asyncio.wait_for(worker.tick(bb), timeout=300.0)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("Agentic worker timeout/error on SQ-%d: %s", sq.id, e)
        return {
            "answer": "", "tokens": 0, "evidence_count": 0,
            "wall_clock": time.monotonic() - t0, "error": str(e),
        }

    snapshot = await bb.get_snapshot()
    sq_data = snapshot["sub_questions"][0] if snapshot["sub_questions"] else {}
    answer = sq_data.get("answer", "") or ""
    evidence_count = len([e for e in snapshot["evidence"] if e["sub_question_id"] == sq.id])

    return {
        "answer": answer,
        "tokens": snapshot["tokens_used"],
        "evidence_count": evidence_count,
        "wall_clock": time.monotonic() - t0,
        "status": sq_data.get("status", "unknown"),
    }


def compute_escalation_label(
    structured: dict, agentic: dict, gold_answer: str,
) -> dict[str, Any]:
    """Determine if escalation was needed and compute per-SQ metrics."""
    s_answer = structured.get("answer", "")
    a_answer = agentic.get("answer", "")

    # We use the agentic answer as pseudo-gold for the sub-question
    # if we don't have per-SQ gold. For final-answer questions, we use gold_answer.
    s_usable = bool(s_answer) and s_answer.lower() not in ("unknown", "error", "")
    a_usable = bool(a_answer) and a_answer.lower() not in ("unknown", "error", "")

    # Compare answers: if both produce the same answer, structured sufficed
    s_norm = normalize(s_answer)
    a_norm = normalize(a_answer)
    answers_match = s_norm == a_norm and s_usable

    # Token cost
    s_tokens = structured.get("tokens", 0)
    a_tokens = agentic.get("tokens", 0)
    token_savings = a_tokens - s_tokens

    # Determine label
    if s_usable and (answers_match or not a_usable):
        label = "ACCEPT"  # structured was enough
    elif a_usable and not s_usable:
        label = "ESCALATE"  # only agentic found an answer
    elif a_usable and s_usable and not answers_match:
        # Both found answers but different — hard to tell without gold
        # Default to ACCEPT (structured found something)
        label = "ACCEPT"
    else:
        label = "BOTH_FAILED"

    return {
        "label": label,
        "structured_usable": s_usable,
        "agentic_usable": a_usable,
        "answers_match": answers_match,
        "token_savings_if_accept": token_savings,
    }


async def collect_one_question(
    item: dict[str, Any],
    config: Config,
    semaphore: asyncio.Semaphore,
    shared_tools: ToolRegistry = None,
) -> dict[str, Any] | None:
    """Collect counterfactual trajectories for one question."""
    async with semaphore:
        qid = item.get("qid") or item.get("id")
        question = item.get("question", "")
        gold = item.get("answer", item.get("gold_answer", ""))

        llm = create_llm_client(config, temperature=0.0)
        tools = shared_tools if shared_tools is not None else build_tools(config)
        adaptive_cfg = config.get("adaptive", {})

        # Step 1: Deterministic decomposition
        bb = Blackboard(question=question, token_budget=300_000)
        planner = PlannerAgent(
            llm_client=llm,
            max_redecompositions=0,
            decompose_temperature=0.0,
        )
        try:
            await asyncio.wait_for(planner.decompose_first(bb), timeout=120.0)
        except asyncio.TimeoutError:
            logger.warning("Decomposition timed out for %s", qid)
            return None
        except Exception as e:
            import traceback
            logger.warning("Decomposition failed for %s: %s -- %s", qid, e, traceback.format_exc())
            return None

        sub_questions = bb.search_plan
        if not sub_questions:
            logger.warning("Empty search plan for %s", qid)
            return None

        decomposition_text = planner.last_decomposition_text

        # Step 2: Process sub-questions in dependency order
        # Simple topological sort: process SQs with no unresolved deps first
        entity_registry: dict[str, str] = {}
        sq_results: list[dict[str, Any]] = []
        resolved_ids: set[int] = set()

        # Build dependency-ordered list
        remaining = list(sub_questions)
        ordered: list[M6SubQuestion] = []
        while remaining:
            batch = [sq for sq in remaining if all(d in resolved_ids for d in sq.dependencies)]
            if not batch:
                # Cycle or unresolvable — just add remaining
                ordered.extend(remaining)
                break
            ordered.extend(batch)
            for sq in batch:
                resolved_ids.add(sq.id)
                remaining.remove(sq)

        resolved_ids.clear()

        for sq in ordered:
            # Resolve placeholder text using already-resolved entities
            resolved_text = resolve_placeholders(sq.text, entity_registry)

            # Skip aggregate SQs — they don't retrieve, so no escalation decision
            if sq.mode == RetrievalMode.AGGREGATE:
                sq_results.append({
                    "sq_id": sq.id,
                    "sq_text": sq.text,
                    "resolved_text": resolved_text,
                    "original_mode": sq.mode.value,
                    "skipped": True,
                    "reason": "aggregate_mode",
                })
                resolved_ids.add(sq.id)
                continue

            # Run BOTH strategies concurrently
            structured_task = run_structured_on_sq(
                llm, tools, sq, question, entity_registry, config,
            )
            agentic_task = run_agentic_on_sq(
                llm, tools, sq, question, entity_registry, config,
            )
            structured_result, agentic_result = await asyncio.gather(
                structured_task, agentic_task,
            )

            # Compute escalation label
            esc = compute_escalation_label(structured_result, agentic_result, gold)

            sq_results.append({
                "sq_id": sq.id,
                "sq_text": sq.text,
                "resolved_text": resolved_text,
                "original_mode": sq.mode.value,
                "dependencies": sq.dependencies,
                "structured": structured_result,
                "agentic": agentic_result,
                "escalation": esc,
                "skipped": False,
            })

            # Update entity registry with the best available answer for downstream SQs
            best_answer = ""
            if esc["label"] == "ACCEPT" and structured_result.get("answer"):
                best_answer = structured_result["answer"]
            elif agentic_result.get("answer"):
                best_answer = agentic_result["answer"]
            elif structured_result.get("answer"):
                best_answer = structured_result["answer"]

            if best_answer:
                entity_registry[f"answer_{sq.id}"] = best_answer
            resolved_ids.add(sq.id)

        # Compute question-level stats
        total_structured_tokens = sum(
            r.get("structured", {}).get("tokens", 0)
            for r in sq_results if not r.get("skipped")
        )
        total_agentic_tokens = sum(
            r.get("agentic", {}).get("tokens", 0)
            for r in sq_results if not r.get("skipped")
        )
        escalation_labels = [
            r["escalation"]["label"]
            for r in sq_results if not r.get("skipped")
        ]

        return {
            "qid": qid,
            "question": question,
            "gold_answer": gold,
            "decomposition_text": decomposition_text,
            "num_sub_questions": len(sub_questions),
            "sub_question_results": sq_results,
            "total_structured_tokens": total_structured_tokens,
            "total_agentic_tokens": total_agentic_tokens,
            "escalation_summary": {
                "ACCEPT": escalation_labels.count("ACCEPT"),
                "ESCALATE": escalation_labels.count("ESCALATE"),
                "BOTH_FAILED": escalation_labels.count("BOTH_FAILED"),
            },
            "potential_token_savings": total_agentic_tokens - total_structured_tokens,
        }


async def run_collection(
    config: Config,
    questions_file: str,
    output_dir: Path,
    limit: int | None,
    concurrent: int,
):
    with open(questions_file) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_file = output_dir / "escalation_trajectories.jsonl"

    # Resume support
    completed_qids: set = set()
    if trajectories_file.exists():
        with open(trajectories_file) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    completed_qids.add(d["qid"])
        logger.info("Resuming: %d questions already collected", len(completed_qids))

    pending = [q for q in questions if (q.get("qid") or q.get("id")) not in completed_qids]
    logger.info("Collecting escalation trajectories for %d questions (%d concurrent)",
                len(pending), concurrent)

    semaphore = asyncio.Semaphore(concurrent)
    t0 = time.monotonic()
    completed = 0

    # Build tools ONCE and share across all questions
    shared_tools = build_tools(config)
    logger.info("Tools loaded: %s", shared_tools.list_tools())

    tasks = [collect_one_question(item, config, semaphore, shared_tools=shared_tools) for item in pending]

    with open(trajectories_file, "a") as fout:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                continue
            fout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
            fout.flush()
            completed += 1
            if completed % 10 == 0:
                elapsed = time.monotonic() - t0
                esc = result["escalation_summary"]
                logger.info(
                    "Progress: %d/%d | last: ACCEPT=%d ESCALATE=%d FAILED=%d | %.1f q/hr",
                    completed, len(pending),
                    esc["ACCEPT"], esc["ESCALATE"], esc["BOTH_FAILED"],
                    completed / elapsed * 3600,
                )

    elapsed = time.monotonic() - t0
    logger.info("Collection complete: %d questions in %.1f min", completed, elapsed / 60)

    # Summary statistics
    with open(trajectories_file) as f:
        all_traj = [json.loads(l) for l in f if l.strip()]

    total_accept = sum(t["escalation_summary"]["ACCEPT"] for t in all_traj)
    total_escalate = sum(t["escalation_summary"]["ESCALATE"] for t in all_traj)
    total_failed = sum(t["escalation_summary"]["BOTH_FAILED"] for t in all_traj)
    total_sqs = total_accept + total_escalate + total_failed
    total_s_tokens = sum(t["total_structured_tokens"] for t in all_traj)
    total_a_tokens = sum(t["total_agentic_tokens"] for t in all_traj)

    summary = {
        "total_questions": len(all_traj),
        "total_sub_questions": total_sqs,
        "escalation_distribution": {
            "ACCEPT": total_accept,
            "ESCALATE": total_escalate,
            "BOTH_FAILED": total_failed,
        },
        "accept_rate": total_accept / total_sqs if total_sqs else 0,
        "escalate_rate": total_escalate / total_sqs if total_sqs else 0,
        "total_structured_tokens": total_s_tokens,
        "total_agentic_tokens": total_a_tokens,
        "potential_savings_pct": (
            (total_a_tokens - total_s_tokens) / total_a_tokens * 100
            if total_a_tokens else 0
        ),
    }
    with open(output_dir / "escalation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary: %s", json.dumps(summary, indent=2))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect escalation trajectories")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--questions", "-q", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--concurrent", type=int, default=20)
    parser.add_argument("--chunks-file", type=str, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.chunks_file:
        config.set("data.chunks_file", args.chunks_file)
    align_data_paths(config, args.questions)

    asyncio.run(run_collection(
        config=config,
        questions_file=args.questions,
        output_dir=Path(args.output),
        limit=args.limit,
        concurrent=args.concurrent,
    ))


if __name__ == "__main__":
    main()
