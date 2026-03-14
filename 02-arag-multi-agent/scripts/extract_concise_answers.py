#!/usr/bin/env python3
"""Extract concise answer entities from verbose E2 predictions using an LLM.

Reads predictions.jsonl, sends each (question, verbose_answer) to an LLM
to extract ONLY the core answer entity, saves new predictions.jsonl.

Usage:
    python scripts/extract_concise_answers.py \
        --input predictions.jsonl \
        --output results/e2_concise/hotpotqa/ \
        --workers 50
"""

import argparse
import json
import os
import re
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm
from arag import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REASONING_TAG_RE = re.compile(
    r"<(?:think|thnk)(?:\s[^>]*)?>.*?</(?:think|thnk)>",
    flags=re.IGNORECASE | re.DOTALL,
)

EXTRACT_PROMPT = """Given a question and a verbose answer, extract ONLY the core answer entity.

Rules:
- Output ONLY the final answer: a name, entity, number, date, place, or yes/no.
- Do NOT include any explanation, reasoning, or context.
- If the verbose answer mentions multiple possibilities, pick the one it commits to as the main answer.
- If the answer is a person, output their full name.
- If the answer is yes/no, output just "yes" or "no".
- Maximum 5 words.

Question: {question}
Verbose answer: {verbose_answer}

Core answer (entity only):"""


def strip_reasoning(text):
    if not text:
        return ""
    text = REASONING_TAG_RE.sub("", text)
    text = re.sub(r"<(?:think|thnk)(?:\s[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:think|thnk)>", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_one(llm, question, verbose_answer):
    cleaned = strip_reasoning(verbose_answer)
    if not cleaned:
        return ""

    prompt = EXTRACT_PROMPT.format(question=question, verbose_answer=cleaned[:2000])
    try:
        response, _ = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        result = strip_reasoning(str(response)).strip().strip("\"'`*")
        # Take first line only
        result = result.split("\n")[0].strip()
        # Strip common prefixes
        result = re.sub(r"^(?:the (?:answer|core answer) is\s+)", "", result, flags=re.IGNORECASE)
        result = re.sub(r"^(?:answer:\s*)", "", result, flags=re.IGNORECASE)
        result = result.strip().strip("\"'`*.")
        return result
    except Exception as e:
        logger.warning("LLM error: %s", e)
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input predictions.jsonl")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "predictions.jsonl"

    # Load predictions
    preds = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    logger.info("Loaded %d predictions from %s", len(preds), args.input)

    llm = LLMClient(
        model=os.getenv("ARAG_MODEL", "Qwen3-8B"),
        api_key=os.getenv("ARAG_API_KEY", "dummy"),
        base_url=os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
        temperature=0.0,
        max_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )

    results = [None] * len(preds)
    write_lock = Lock()

    def process(idx):
        p = preds[idx]
        question = p.get("question", "")
        verbose = p.get("pred_answer", "")
        concise = extract_one(llm, question, verbose)

        new_pred = {
            "qid": p.get("qid", ""),
            "question": question,
            "gold_answer": p.get("gold_answer", p.get("answer", "")),
            "pred_answer": concise,
            "pred_answer_verbose": verbose,
            "total_cost": p.get("total_cost", 0),
            "loops": p.get("loops", 0),
            "total_retrieved_tokens": p.get("total_retrieved_tokens", 0),
            "trajectory": p.get("trajectory", []),
            "chunks_read_count": p.get("chunks_read_count", 0),
            "chunks_read_ids": p.get("chunks_read_ids", []),
        }
        return idx, new_pred

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, i): i for i in range(len(preds))}
        pbar = tqdm(total=len(preds), desc="Extracting")

        for future in as_completed(futures):
            idx, new_pred = future.result()
            results[idx] = new_pred
            pbar.update(1)
        pbar.close()

    # Write results in order
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick stats
    concise_lens = [len(r["pred_answer"]) for r in results]
    verbose_lens = [len(strip_reasoning(r["pred_answer_verbose"])) for r in results]
    empty = sum(1 for r in results if not r["pred_answer"])
    logger.info(
        "Done: %d predictions, %d empty, avg len: %d -> %d chars",
        len(results), empty,
        sum(verbose_lens) // len(verbose_lens),
        sum(concise_lens) // len(concise_lens),
    )
    logger.info("Saved to %s", out_file)


if __name__ == "__main__":
    main()
