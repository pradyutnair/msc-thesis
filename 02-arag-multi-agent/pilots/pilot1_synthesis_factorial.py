#!/usr/bin/env python3
"""
Pilot 1 — Synthesis Method Factorial (2x2)

Takes 50 E4 MuSiQue failures where evidence was retrieved.
Re-runs ONLY the synthesis step with 4 conditions:
  A: flat evidence pool  + thinking OFF  (E4 baseline synthesis)
  B: flat evidence pool  + thinking ON
  C: structured triples  + thinking OFF
  D: structured triples  + thinking ON

Reports accuracy per condition to isolate reasoning vs structure.
"""

import os
import json
import re
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── LLM client ──────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "external/arag/src"))
from arag import LLMClient


def make_client(thinking: bool, base_url: str, model: str) -> LLMClient:
    return LLMClient(
        model=model,
        api_key=os.getenv("ARAG_API_KEY", "dummy"),
        base_url=base_url,
        temperature=0.0,
        max_tokens=4096 if thinking else 512,
        chat_template_kwargs={"enable_thinking": thinking},
    )


# ─── Evidence extraction ─────────────────────────────────────────────────────

def extract_chunks_from_trajectory(trajectory: list) -> list[dict]:
    """Extract chunks from E4 trajectory (read_chunk + search result snippets)."""
    chunks = []
    seen_ids = set()

    for step in trajectory:
        tool = step.get("tool_name", "")
        result = step.get("tool_result", "")

        if tool == "read_chunk":
            # Parse full chunk blocks: [Chunk N]\n---\n<text>
            for match in re.finditer(
                r"\[Chunk (\d+)\]\s*-+\s*(.*?)(?=\[Chunk \d+\]|\Z)",
                result, re.DOTALL
            ):
                cid, text = match.group(1), match.group(2).strip()
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append({"id": cid, "text": text, "source": "read"})

        elif tool in ("keyword_search", "semantic_search"):
            # Extract short snippet lines from search results
            for match in re.finditer(r"Chunk ID: (\d+)[^\n]*\n.*?Matched: (.*?)(?=Chunk ID:|\Z)", result, re.DOTALL):
                cid, snippet = match.group(1), match.group(2).strip()[:400]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    chunks.append({"id": cid, "text": snippet, "source": "search_snippet"})

    return chunks


# ─── Prompt builders ─────────────────────────────────────────────────────────

FLAT_SYNTHESIS_PROMPT = """You are an expert question answering system. Answer the question using ONLY the provided evidence.
Give a concise, direct answer. Do not explain or add context.

Question: {question}

Evidence:
{evidence}

Answer (one phrase or sentence only):"""


STRUCTURED_SYNTHESIS_PROMPT = """You are an expert question answering system. Answer the question using ONLY the provided evidence.

The evidence has been structured as a fact chain to help you trace multi-hop reasoning:

Question: {question}

Structured Evidence:
{evidence}

Instructions:
1. Identify the key entities and relationships in the chain
2. Trace the reasoning path step by step
3. Give a concise, direct answer

Answer (one phrase or sentence only):"""


def build_flat_evidence(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        text = c["text"].replace("\n", " ").strip()
        lines.append(f"[Doc {i}] {text}")
    return "\n".join(lines)


def build_structured_evidence(chunks: list[dict], question: str) -> str:
    """Format chunks as structured fact triples + chain hints."""
    lines = []

    # Extract entity mentions to build a chain
    all_text = " ".join(c["text"] for c in chunks)

    lines.append("=== Retrieved Facts ===")
    for i, c in enumerate(chunks, 1):
        text = c["text"].replace("\n", " ").strip()
        lines.append(f"\n[Fact {i}]")
        lines.append(f"Source: Document {c['id']}")
        lines.append(f"Content: {text}")

    lines.append("\n=== Evidence Chain ===")
    lines.append("Trace the multi-hop path through the above facts to answer:")
    lines.append(f"  Step 1: Find intermediate entity relevant to the question")
    lines.append(f"  Step 2: Use intermediate entity to find the final answer")
    lines.append(f"  Target: {question}")

    return "\n".join(lines)


# ─── Evaluation ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """Judge whether the predicted answer is correct.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer correct? Reply with only "yes" or "no"."""


def judge_answer(judge_client: LLMClient, question: str, gold: str, pred: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
    resp = judge_client.chat([{"role": "user", "content": prompt}])
    return "yes" in resp.lower()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_condition(
    label: str,
    questions: list[dict],
    thinking: bool,
    structured: bool,
    llm_client: LLMClient,
    judge_client: LLMClient,
) -> dict:
    logger.info(f"Running condition {label} (thinking={thinking}, structured={structured})")
    results = []

    for i, item in enumerate(questions):
        q = item["question"]
        gold = item["gold_answer"]
        chunks = item["chunks"]

        if not chunks:
            results.append({"qid": item["qid"], "correct": False, "pred": "", "note": "no_chunks"})
            continue

        # Build prompt
        if structured:
            evidence = build_structured_evidence(chunks, q)
            prompt = STRUCTURED_SYNTHESIS_PROMPT.format(question=q, evidence=evidence)
        else:
            evidence = build_flat_evidence(chunks)
            prompt = FLAT_SYNTHESIS_PROMPT.format(question=q, evidence=evidence)

        try:
            pred = llm_client.chat([{"role": "user", "content": prompt}])
            # Strip thinking tags if present
            pred = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL).strip()
            # Take first line as answer
            pred = pred.split("\n")[0].strip()

            correct = judge_answer(judge_client, q, gold, pred)
            results.append({"qid": item["qid"], "correct": correct, "pred": pred})

            if (i + 1) % 10 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                logger.info(f"  [{label}] {i+1}/{len(questions)}: running acc={acc:.3f}")

        except Exception as e:
            logger.warning(f"  [{label}] Error on {item['qid']}: {e}")
            results.append({"qid": item["qid"], "correct": False, "pred": "", "error": str(e)})

    n = len(results)
    correct = sum(r["correct"] for r in results)
    return {
        "condition": label,
        "thinking": thinking,
        "structured": structured,
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n > 0 else 0.0,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Pilot 1: Synthesis factorial")
    parser.add_argument("--e4-predictions", required=True, help="E4 MuSiQue predictions.jsonl")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of failure samples")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1", help="LLM base URL")
    parser.add_argument("--llm-model", default="Qwen3-30B-A3B", help="Generator model name")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:9000/v1", help="Judge base URL")
    parser.add_argument("--judge-model", default="DeepSeek-R1-Distill-Qwen-32B", help="Judge model name")
    parser.add_argument("--conditions", default="ABCD", help="Which conditions to run (e.g. AB, ABCD)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load E4 failures with trajectory
    logger.info("Loading E4 predictions...")
    preds = []
    with open(args.e4_predictions) as f:
        for line in f:
            preds.append(json.loads(line))

    failures = [p for p in preds if p.get("llm_accuracy", 1) == 0 and len(p.get("trajectory", [])) >= 2]
    logger.info(f"Found {len(failures)} failures with trajectory (need {args.n_samples})")
    failures = failures[:args.n_samples]

    # Extract chunks for each failure
    samples = []
    for p in failures:
        chunks = extract_chunks_from_trajectory(p.get("trajectory", []))
        samples.append({
            "qid": p["qid"],
            "question": p["question"],
            "gold_answer": p["gold_answer"],
            "chunks": chunks,
            "e4_pred": p.get("pred_answer", ""),
        })

    chunk_counts = [len(s["chunks"]) for s in samples]
    logger.info(f"Extracted chunks: avg={sum(chunk_counts)/len(chunk_counts):.1f}, "
                f"min={min(chunk_counts)}, max={max(chunk_counts)}")

    # Condition definitions
    condition_map = {
        "A": (False, False, "flat + nothink (E4 baseline)"),
        "B": (False, True,  "flat + thinking"),
        "C": (True,  False, "structured + nothink"),
        "D": (True,  True,  "structured + thinking"),
    }

    # Note: For simplicity, use same endpoint for both thinking ON and OFF
    # (vLLM handles enable_thinking at request level via chat_template_kwargs)
    judge_client = LLMClient(
        model=args.judge_model,
        api_key=os.getenv("ARAG_API_KEY", "dummy"),
        base_url=args.judge_base_url,
        temperature=0.0,
        max_tokens=16,
    )

    all_results = []
    for cond in args.conditions:
        if cond not in condition_map:
            logger.warning(f"Unknown condition {cond}, skipping")
            continue
        thinking, structured, desc = condition_map[cond]
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition {cond}: {desc}")

        llm_client = make_client(thinking, args.llm_base_url, args.llm_model)
        result = run_condition(cond, samples, thinking, structured, llm_client, judge_client)
        result["description"] = desc
        all_results.append(result)
        logger.info(f"Condition {cond} accuracy: {result['accuracy']:.3f} ({result['correct']}/{result['n']})")

        # Save intermediate results
        with open(output_dir / f"condition_{cond}_results.json", "w") as f:
            json.dump(result, f, indent=2)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PILOT 1 SUMMARY")
    logger.info(f"{'='*60}")
    summary = []
    for r in all_results:
        logger.info(f"  {r['condition']}: {r['description']}")
        logger.info(f"     Accuracy: {r['accuracy']:.3f} ({r['correct']}/{r['n']})")
        summary.append({
            "condition": r["condition"],
            "description": r["description"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "n": r["n"],
        })

    with open(output_dir / "pilot1_summary.json", "w") as f:
        json.dump({"samples": args.n_samples, "conditions": summary}, f, indent=2)
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
