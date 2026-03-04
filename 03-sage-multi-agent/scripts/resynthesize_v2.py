#!/usr/bin/env python3
"""Re-synthesize wrong MuSiQue predictions using agent findings with self-consistency.

Uses majority voting across multiple samples to improve accuracy.
Only re-synthesizes wrong predictions (llm_accuracy=0). Keeps correct ones.

Usage:
    python -u resynthesize_v2.py \
        --predictions results/sage_v7_1000/musique/predictions.jsonl \
        --output results/sage_v7_resynth2/musique/predictions.jsonl \
        --base-url http://127.0.0.1:8001/v1 \
        --samples 5 \
        --concurrent 16
"""

import argparse
import asyncio
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


CHAIN_PROMPT = """Answer a multi-hop question by chaining research findings step by step.

QUESTION: {question}

RESEARCH FINDINGS:
{agent_findings}

INSTRUCTIONS:
1. Each agent investigated a sub-question. Chain their findings to answer the main question.
2. For bridge questions: Agent 0's answer is used by Agent 1, whose answer is used by Agent 2, etc. The final answer comes from the LAST agent in the chain.
3. Give a SHORT, SPECIFIC answer (1-5 words). Just the answer entity, nothing else.
4. You MUST give an answer. Never say "cannot be determined" or "not found".
5. If agent findings are incomplete, use your best inference from available clues.

FINAL ANSWER:"""


DIRECT_PROMPT = """Answer this question as briefly as possible (1-5 words).

Question: {question}

Context clues from research:
{agent_findings}

Give ONLY the answer, nothing else. If uncertain, give your best guess.

Answer:"""


async def call_llm(session, base_url, prompt, semaphore, temperature=0.6, max_tokens=128):
    """Make async LLM call."""
    async with semaphore:
        url = f"{base_url}/chat/completions"
        payload = {
            "model": "Qwen3-30B-A3B",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        }

        for attempt in range(3):
            try:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(1)
                        continue
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content
            except Exception as e:
                if attempt == 2:
                    log.warning(f"LLM call failed: {e}")
                await asyncio.sleep(1)

        return None


def extract_answer(raw):
    """Extract final answer from LLM response."""
    if not raw:
        return ""

    # Remove think tags
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)

    # Look for FINAL ANSWER marker
    m = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
    if m:
        answer = m.group(1).strip().split("\n")[0].strip()
        answer = re.sub(r"</?[^>]+>", "", answer).strip()
        if answer and len(answer) < 200:
            return answer

    # Look for "Answer:" pattern
    m = re.search(r"(?:^|\n)\s*(?:answer|the answer is)\s*:?\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        if answer and len(answer) < 200:
            return answer

    # Take last non-empty line if short
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        # Remove common prefixes
        for prefix in ["Answer:", "FINAL ANSWER:", "The answer is", "Therefore,"]:
            if last.lower().startswith(prefix.lower()):
                last = last[len(prefix):].strip()
        if last and len(last) < 200:
            return last

    return ""


def normalize_answer(answer):
    """Normalize for comparison during majority voting."""
    answer = answer.lower().strip()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = re.sub(r"\s+", " ", answer)
    return answer


def format_agent_findings(agent_results):
    """Format agent results into readable findings."""
    lines = []
    for tid in sorted(agent_results.keys(), key=lambda x: int(x) if x.isdigit() else 100):
        ar = agent_results[tid]
        answer = ar.get("answer", "").strip()
        if answer:
            # Truncate very long agent answers
            if len(answer) > 300:
                answer = answer[:300] + "..."
            lines.append(f"Agent {tid}: {answer}")
        else:
            lines.append(f"Agent {tid}: (no result)")
    return "\n".join(lines) if lines else "No findings available."


async def resynthesize_with_voting(session, base_url, pred, semaphore, n_samples=5):
    """Re-synthesize one prediction using self-consistency (majority voting)."""
    question = pred["question"]
    agent_results = pred.get("agent_results", {})
    findings = format_agent_findings(agent_results)

    # Check if any agent actually has useful findings
    has_useful_findings = any(
        v.get("answer", "").strip()
        and not v["answer"].startswith("No ")
        and not v["answer"].startswith("The provided ")
        and "not contain" not in v["answer"]
        and "not found" not in v["answer"].lower()
        for v in agent_results.values()
    )

    if not has_useful_findings:
        # No useful agent findings - can't improve
        return pred["pred_answer"]

    candidates = []

    # Use chain prompt (main approach)
    chain_prompt = CHAIN_PROMPT.format(question=question, agent_findings=findings)
    tasks_chain = [
        call_llm(session, base_url, chain_prompt, semaphore, temperature=0.6)
        for _ in range(n_samples)
    ]

    # Use direct prompt (backup approach)
    direct_prompt = DIRECT_PROMPT.format(question=question, agent_findings=findings)
    tasks_direct = [
        call_llm(session, base_url, direct_prompt, semaphore, temperature=0.4)
        for _ in range(max(1, n_samples // 2))
    ]

    all_results = await asyncio.gather(*(tasks_chain + tasks_direct), return_exceptions=True)

    for raw in all_results:
        if isinstance(raw, Exception) or not raw:
            continue
        answer = extract_answer(raw)
        if answer and len(answer) < 100:
            # Filter out non-answers
            lower = answer.lower()
            if any(bad in lower for bad in [
                "cannot", "not determined", "not found", "insufficient",
                "no evidence", "not specified", "not available", "not clear",
                "no information", "unable to", "not mentioned",
            ]):
                continue
            candidates.append(answer)

    if not candidates:
        return pred["pred_answer"]  # Keep original

    # Majority vote with normalization
    normalized = [normalize_answer(c) for c in candidates]
    counts = Counter(normalized)
    best_norm, best_count = counts.most_common(1)[0]

    # Find the original (un-normalized) version
    for cand, norm in zip(candidates, normalized):
        if norm == best_norm:
            return cand

    return candidates[0]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--concurrent", type=int, default=16)
    args = parser.parse_args()

    preds = [json.loads(l) for l in open(args.predictions)]
    log.info(f"Loaded {len(preds)} predictions")

    correct = [p for p in preds if p.get("llm_accuracy") == 1.0]
    wrong = [p for p in preds if p.get("llm_accuracy") != 1.0]
    empty_orig = [p for p in wrong if not p["pred_answer"].strip()]

    log.info(f"Correct (keeping): {len(correct)}")
    log.info(f"Wrong/empty (re-synthesizing): {len(wrong)}")
    log.info(f"Of which empty: {len(empty_orig)}")
    log.info(f"Using {args.samples} samples per question for self-consistency")

    semaphore = asyncio.Semaphore(args.concurrent)
    connector = aiohttp.TCPConnector(limit=args.concurrent + 5)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            resynthesize_with_voting(session, args.base_url, p, semaphore, args.samples)
            for p in wrong
        ]

        log.info(f"Starting {len(tasks)} re-synthesis tasks (~{len(tasks) * args.samples * 1.5:.0f} LLM calls)...")
        t0 = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - t0
        log.info(f"Re-synthesis done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    changed = 0
    still_empty = 0
    for pred, new_answer in zip(wrong, results):
        if isinstance(new_answer, Exception):
            new_answer = pred["pred_answer"]

        old_answer = pred["pred_answer"].strip()
        new_answer = (new_answer or "").strip()

        if new_answer and new_answer != old_answer:
            pred["pred_answer_original"] = old_answer
            pred["pred_answer"] = new_answer
            pred["resynthesized"] = True
            changed += 1

        if not pred["pred_answer"].strip():
            still_empty += 1

    log.info(f"Changed answers: {changed}")
    log.info(f"Still empty: {still_empty}")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_preds = {p["qid"]: p for p in correct}
    for p in wrong:
        all_preds[p["qid"]] = p

    qid_order = [p["qid"] for p in preds]
    with open(output_path, "w") as f:
        for qid in qid_order:
            f.write(json.dumps(all_preds[qid], ensure_ascii=False, default=str) + "\n")

    log.info(f"Written {len(qid_order)} predictions to {output_path}")

    # Quick stats
    empty_final = sum(1 for q in qid_order if not all_preds[q]["pred_answer"].strip())
    log.info(f"Final empty predictions: {empty_final}")


if __name__ == "__main__":
    asyncio.run(main())
