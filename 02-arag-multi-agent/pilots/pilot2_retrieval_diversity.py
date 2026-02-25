#!/usr/bin/env python3
"""
Pilot 2 — Retrieval Diversity

For 50 questions, compares chunk coverage between:
  - Strategy A: 1 query (model generates its own query in E4 style — from trajectory)
  - Strategy B: 3 diverse queries (keyword variant, entity-focused, paraphrase)

Measures:
  - Total unique chunks retrieved
  - Overlap with E4 correct-answer questions (as proxy for relevant chunks)
  - Unique chunks per strategy not found by the other
"""

import os
import json
import re
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external/arag/src"))
from arag import LLMClient
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.semantic_search import SemanticSearchTool


PLANNER_PROMPT = """You are a search query planner for a multi-hop QA system.

Given the question below, generate 3 DIVERSE search queries to maximize evidence coverage:
1. A keyword-focused query (specific named entities, dates, titles)
2. An entity-relationship query (the key relationship being asked about)
3. A paraphrase query (restate the question differently)

Question: {question}

Output ONLY a JSON array of 3 query strings, nothing else.
Example: ["query 1", "query 2", "query 3"]"""


def generate_diverse_queries(llm_client: LLMClient, question: str) -> list[str]:
    """Use LLM to generate 3 diverse search queries."""
    prompt = PLANNER_PROMPT.format(question=question)
    try:
        response = llm_client.chat([{"role": "user", "content": prompt}])
        # Strip thinking tags
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Parse JSON array
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and len(queries) >= 1:
                return queries[:3]
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")
    # Fallback: use question as-is
    return [question]


def extract_e4_first_query(trajectory: list) -> str | None:
    """Extract the first search query used by E4 for this question."""
    for step in trajectory:
        if step.get("tool_name") == "keyword_search":
            args = step.get("arguments", {})
            keywords = args.get("keywords", [])
            if keywords:
                return " ".join(keywords)
        elif step.get("tool_name") == "semantic_search":
            args = step.get("arguments", {})
            return args.get("query", "")
    return None


def retrieve_chunks(
    keyword_tool: KeywordSearchTool,
    semantic_tool: SemanticSearchTool,
    query: str,
    top_k: int = 5,
) -> set[str]:
    """Retrieve chunk IDs using both keyword and semantic search for a query."""
    chunk_ids = set()

    try:
        kw_result = keyword_tool.run(keywords=query.split()[:6], top_k=top_k)
        for match in re.finditer(r"Chunk ID: (\d+)", str(kw_result)):
            chunk_ids.add(match.group(1))
    except Exception as e:
        logger.debug(f"Keyword search error: {e}")

    try:
        sem_result = semantic_tool.run(query=query, top_k=top_k)
        for match in re.finditer(r"Chunk ID: (\d+)", str(sem_result)):
            chunk_ids.add(match.group(1))
    except Exception as e:
        logger.debug(f"Semantic search error: {e}")

    return chunk_ids


def main():
    parser = argparse.ArgumentParser(description="Pilot 2: Retrieval diversity")
    parser.add_argument("--e4-predictions", required=True, help="E4 predictions.jsonl")
    parser.add_argument("--chunks-file", required=True, help="Chunks JSON file")
    parser.add_argument("--index-dir", required=True, help="FAISS index directory")
    parser.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="Qwen3-30B-A3B")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load E4 predictions (mix correct + incorrect for diversity analysis)
    logger.info("Loading E4 predictions...")
    preds = []
    with open(args.e4_predictions) as f:
        for line in f:
            preds.append(json.loads(line))

    # Take first N samples with trajectory
    samples = [p for p in preds if p.get("trajectory")][:args.n_samples]
    logger.info(f"Using {len(samples)} questions")

    # Init tools
    logger.info("Loading retrieval tools...")
    from arag.tools.keyword_search import KeywordSearchTool
    from arag.tools.semantic_search import SemanticSearchTool
    from arag import Config

    keyword_tool = KeywordSearchTool(chunks_file=args.chunks_file)
    semantic_tool = SemanticSearchTool(
        chunks_file=args.chunks_file,
        index_dir=args.index_dir,
        model_name=args.embedding_model,
        device="cpu",
    )

    # Init LLM for query generation (nothink is fine for planning)
    llm_client = LLMClient(
        model=args.llm_model,
        api_key=os.getenv("ARAG_API_KEY", "dummy"),
        base_url=args.llm_base_url,
        temperature=0.0,
        max_tokens=256,
        chat_template_kwargs={"enable_thinking": False},
    )

    results = []
    for i, p in enumerate(samples):
        q = p["question"]
        gold_acc = p.get("llm_accuracy", 0)

        # Strategy A: E4's first query
        e4_query = extract_e4_first_query(p.get("trajectory", []))
        if not e4_query:
            e4_query = q

        # Strategy B: 3 diverse queries
        diverse_queries = generate_diverse_queries(llm_client, q)
        logger.info(f"[{i+1}/{len(samples)}] Q: {q[:60]}")
        logger.info(f"  E4 query: {e4_query[:60]}")
        logger.info(f"  Diverse queries: {diverse_queries}")

        # Retrieve for A (single query)
        chunks_a = retrieve_chunks(keyword_tool, semantic_tool, e4_query)

        # Retrieve for B (union of 3 diverse queries)
        chunks_b = set()
        for dq in diverse_queries:
            chunks_b |= retrieve_chunks(keyword_tool, semantic_tool, dq)

        # Metrics
        overlap = chunks_a & chunks_b
        a_only = chunks_a - chunks_b
        b_only = chunks_b - chunks_a

        result = {
            "qid": p["qid"],
            "question": q,
            "e4_correct": bool(gold_acc),
            "e4_query": e4_query,
            "diverse_queries": diverse_queries,
            "strategy_a_chunks": len(chunks_a),
            "strategy_b_chunks": len(chunks_b),
            "overlap": len(overlap),
            "a_only": len(a_only),
            "b_only": len(b_only),
            "b_gain_over_a": len(b_only),
        }
        results.append(result)
        logger.info(f"  A: {len(chunks_a)} chunks | B: {len(chunks_b)} chunks | B gain: +{len(b_only)}")

    # Summary stats
    n = len(results)
    avg_a = sum(r["strategy_a_chunks"] for r in results) / n
    avg_b = sum(r["strategy_b_chunks"] for r in results) / n
    avg_gain = sum(r["b_gain_over_a"] for r in results) / n
    correct = [r for r in results if r["e4_correct"]]
    wrong = [r for r in results if not r["e4_correct"]]

    summary = {
        "n_samples": n,
        "avg_chunks_strategy_a_single": avg_a,
        "avg_chunks_strategy_b_diverse": avg_b,
        "avg_new_chunks_from_diversity": avg_gain,
        "pct_increase": (avg_b - avg_a) / avg_a * 100 if avg_a > 0 else 0,
        "correct_questions": {
            "n": len(correct),
            "avg_a": sum(r["strategy_a_chunks"] for r in correct) / max(1, len(correct)),
            "avg_b": sum(r["strategy_b_chunks"] for r in correct) / max(1, len(correct)),
        },
        "wrong_questions": {
            "n": len(wrong),
            "avg_a": sum(r["strategy_a_chunks"] for r in wrong) / max(1, len(wrong)),
            "avg_b": sum(r["strategy_b_chunks"] for r in wrong) / max(1, len(wrong)),
        },
    }

    logger.info(f"\n{'='*60}")
    logger.info("PILOT 2 SUMMARY")
    logger.info(f"  Strategy A (1 query):       avg {avg_a:.1f} chunks")
    logger.info(f"  Strategy B (3 diverse):     avg {avg_b:.1f} chunks")
    logger.info(f"  Avg new chunks from diversity: +{avg_gain:.1f}")
    logger.info(f"  Diversity gain: +{summary['pct_increase']:.1f}%")
    logger.info(f"  Correct questions: A={summary['correct_questions']['avg_a']:.1f}, B={summary['correct_questions']['avg_b']:.1f}")
    logger.info(f"  Wrong questions:   A={summary['wrong_questions']['avg_a']:.1f}, B={summary['wrong_questions']['avg_b']:.1f}")

    with open(output_dir / "pilot2_results.json", "w") as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)
    with open(output_dir / "pilot2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
