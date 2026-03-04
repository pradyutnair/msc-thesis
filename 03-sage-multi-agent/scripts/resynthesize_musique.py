#!/usr/bin/env python3
"""Re-synthesize wrong/empty MuSiQue predictions using agent findings.

Only re-synthesizes predictions where LLM judge marked them wrong (llm_accuracy=0)
or where the prediction was empty. Keeps correct predictions unchanged.

Usage:
    python -u resynthesize_musique.py \
        --predictions results/sage_v7_1000/musique/predictions.jsonl \
        --output results/sage_v7_resynth/musique/predictions.jsonl \
        --base-url http://127.0.0.1:8000/v1 \
        --concurrent 32
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RESYNTH_PROMPT = """You must answer a multi-hop question by chaining intermediate research findings.

## Question
{question}

## Question Type
{question_type}

## Research Findings (from multiple agents)
{agent_findings}

## Previous Answer Attempt
{prev_answer}

## Instructions

This is a {question_type} question requiring you to chain multiple facts together.

For BRIDGE questions: Agent findings form a reasoning chain. Agent 0's answer feeds into Agent 1's search, and so on. The FINAL answer is typically derived from the LAST agent's finding.
- Example: Q: "Where was the director of Film X born?" → Agent 0 finds director is "John Smith" → Agent 1 finds "John Smith was born in London" → Answer: "London"

For COMPARISON questions: Compare the findings from different agents to determine the answer.

CRITICAL RULES:
1. ALWAYS give a concrete, specific answer. Never say "cannot be determined" or "insufficient evidence".
2. If agents found partial information, use your best judgment to chain the reasoning.
3. Answer in 1-8 words maximum.
4. If the previous answer attempt was wrong, try a DIFFERENT approach to chaining the findings.
5. Look for the KEY ENTITY that directly answers the question - don't repeat intermediate entities.

Think step by step, then give your final answer.

REASONING:
"""


async def call_llm(session, base_url, prompt, semaphore, api_key="dummy"):
    """Make async LLM call."""
    async with semaphore:
        url = f"{base_url}/chat/completions"
        payload = {
            "model": "Qwen3-30B-A3B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,  # slight temperature for diversity
            "max_tokens": 512,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        log.warning(f"LLM call failed (attempt {attempt+1}): {resp.status} {text[:200]}")
                        await asyncio.sleep(2)
                        continue
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
            except Exception as e:
                log.warning(f"LLM call error (attempt {attempt+1}): {e}")
                await asyncio.sleep(2)

        return None


def extract_answer(raw):
    """Extract final answer from LLM response."""
    import re

    if not raw:
        return ""

    # Remove think tags
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)

    # Look for FINAL ANSWER marker
    m = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        # Clean up
        answer = re.sub(r"</?(?:answer|response|result)[^>]*>", "", answer, flags=re.IGNORECASE).strip()
        if answer:
            return answer

    # Look for "Answer:" or "The answer is"
    m = re.search(r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)", raw, re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        if answer:
            return answer

    # Take last non-empty line
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if lines:
        # Check if last line looks like an answer (short)
        last = lines[-1]
        if len(last) < 200:
            return last

    return raw.strip()[:200]


def format_agent_findings(agent_results):
    """Format agent results into readable findings."""
    lines = []
    for tid in sorted(agent_results.keys(), key=lambda x: int(x) if x.isdigit() else 100):
        ar = agent_results[tid]
        answer = ar.get("answer", "").strip()
        if answer:
            lines.append(f"- Agent {tid}: {answer}")
        else:
            lines.append(f"- Agent {tid}: (no findings)")
    return "\n".join(lines) if lines else "No agent findings available."


async def resynthesize_one(session, base_url, pred, semaphore):
    """Re-synthesize one prediction."""
    question = pred["question"]
    qtype = pred.get("question_type", "bridge")
    agent_results = pred.get("agent_results", {})
    prev_answer = pred.get("pred_answer", "").strip()

    findings = format_agent_findings(agent_results)

    prompt = RESYNTH_PROMPT.format(
        question=question,
        question_type=qtype,
        agent_findings=findings,
        prev_answer=prev_answer if prev_answer else "(empty - no answer produced)",
    )

    # Add the answer extraction instruction
    prompt += "\n\nNow give your final answer:\nFINAL ANSWER:"

    raw = await call_llm(session, base_url, prompt, semaphore)
    if raw:
        answer = extract_answer(raw)
        return answer
    return prev_answer  # fallback to original


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--concurrent", type=int, default=32)
    args = parser.parse_args()

    # Load predictions
    preds = []
    with open(args.predictions) as f:
        for line in f:
            preds.append(json.loads(line))
    log.info(f"Loaded {len(preds)} predictions")

    # Separate correct vs wrong/empty
    correct = [p for p in preds if p.get("llm_accuracy") == 1.0]
    wrong = [p for p in preds if p.get("llm_accuracy") != 1.0]
    empty = [p for p in preds if not p["pred_answer"].strip()]

    log.info(f"Correct (keeping): {len(correct)}")
    log.info(f"Wrong/empty (re-synthesizing): {len(wrong)}")
    log.info(f"Of which empty: {len(empty)}")

    # Re-synthesize wrong/empty predictions
    semaphore = asyncio.Semaphore(args.concurrent)
    connector = aiohttp.TCPConnector(limit=args.concurrent + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for pred in wrong:
            tasks.append(resynthesize_one(session, args.base_url, pred, semaphore))

        log.info(f"Starting {len(tasks)} re-synthesis calls with concurrency {args.concurrent}...")
        t0 = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - t0
        log.info(f"Re-synthesis done in {elapsed:.1f}s")

    # Merge results
    resynth_count = 0
    changed_count = 0
    still_empty = 0
    for pred, new_answer in zip(wrong, results):
        if isinstance(new_answer, Exception):
            log.warning(f"Error for {pred['qid']}: {new_answer}")
            new_answer = pred["pred_answer"]

        old_answer = pred["pred_answer"].strip()
        new_answer = new_answer.strip() if new_answer else ""

        if new_answer and new_answer != old_answer:
            pred["pred_answer_original"] = old_answer
            pred["pred_answer"] = new_answer
            pred["resynthesized"] = True
            changed_count += 1
        if not new_answer:
            still_empty += 1
        resynth_count += 1

    log.info(f"Re-synthesized: {resynth_count}")
    log.info(f"Changed answers: {changed_count}")
    log.info(f"Still empty: {still_empty}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine correct + modified wrong, sorted by original order
    all_preds = {p["qid"]: p for p in correct}
    for p in wrong:
        all_preds[p["qid"]] = p

    # Write in original order
    qid_order = [p["qid"] for p in preds]
    with open(output_path, "w") as f:
        for qid in qid_order:
            f.write(json.dumps(all_preds[qid], ensure_ascii=False, default=str) + "\n")

    log.info(f"Written {len(qid_order)} predictions to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
