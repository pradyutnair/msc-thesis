#!/usr/bin/env python3
"""Runner for Submodular RAG pipeline."""

import argparse
import json
import logging
import os
import re
import string
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
# Also add M6 project for LLM client reuse
sys.path.insert(0, str(Path(__file__).parent.parent / "02-arag-multi-agent" / "src"))

from arag.core.llm import LLMClient
from pipeline import SubmodPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def norm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(pred, gold):
    pt, gt = set(norm(pred).split()), set(norm(gold).split())
    if not pt or not gt:
        return 0.0
    c = pt & gt
    if not c:
        return 0.0
    p = len(c) / len(pt)
    r = len(c) / len(gt)
    return 2 * p * r / (p + r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, choices=["hotpotqa", "2wiki", "musique"])
    parser.add_argument("--limit", "-l", type=int, default=10)
    parser.add_argument("--output", "-o", default="results/smoke")
    parser.add_argument("--budget", type=int, default=15, help="Retrieval budget per SQ")
    parser.add_argument("--select-k", type=int, default=5, help="Submodular selection size")
    args = parser.parse_args()

    dataset_map = {"hotpotqa": "hotpotqa", "2wiki": "2wikimultihop", "musique": "musique"}
    ds_name = dataset_map[args.dataset]
    data_dir = Path(__file__).parent / "data" / ds_name

    chunks_file = str(data_dir / "chunks.json")
    questions_file = str(data_dir / "questions.json")
    index_dir = str(data_dir / "index_e5_base_v2")

    output_dir = Path(args.output) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / "predictions.jsonl"

    # LLM client (uses env vars from SLURM job)
    llm = LLMClient(
        model=os.getenv("ARAG_MODEL", "Qwen3-8B"),
        api_key=os.getenv("ARAG_API_KEY", "dummy"),
        base_url=os.getenv("ARAG_BASE_URL", "http://127.0.0.1:8000/v1"),
        temperature=0.0,
        max_tokens=8192,
        chat_template_kwargs={"enable_thinking": True},
    )

    logger.info("Loading pipeline...")
    pipeline = SubmodPipeline(
        llm_client=llm,
        chunks_file=chunks_file,
        index_dir=index_dir,
        model_name="intfloat/e5-base-v2",
        device="cpu",
        retrieval_budget=args.budget,
        selection_k=args.select_k,
    )

    questions = json.load(open(questions_file, encoding="utf-8"))[:args.limit]
    logger.info("Running %d questions from %s", len(questions), args.dataset)

    t0 = time.monotonic()
    correct = 0

    with open(pred_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(questions):
            qid = item.get("qid") or item.get("id")
            question = item["question"]
            gold = item.get("answer", item.get("gold_answer", ""))

            logger.info("[%d/%d] %s: %s", i + 1, len(questions), qid, question[:60])

            try:
                result = pipeline.run(question)
                pred = result["pred_answer"]
            except Exception as exc:
                logger.error("Error on %s: %s", qid, exc)
                pred = f"Error: {exc}"
                result = {}

            is_correct = norm(pred) == norm(gold)
            if is_correct:
                correct += 1

            prediction = {
                "qid": qid,
                "question": question,
                "gold_answer": gold,
                "pred_answer": pred,
                **{k: v for k, v in result.items() if k != "pred_answer"},
            }
            f.write(json.dumps(prediction, ensure_ascii=False, default=str) + "\n")
            f.flush()

            status = "OK" if is_correct else "XX"
            logger.info("  %s PRED='%s' GOLD='%s'", status, pred[:40], gold[:40])

    elapsed = time.monotonic() - t0
    em = correct / len(questions) * 100
    f1_avg = sum(
        token_f1(
            json.loads(line).get("pred_answer", ""),
            json.loads(line).get("gold_answer", ""),
        )
        for line in open(pred_file)
    ) / len(questions) * 100

    print(f"\n=== SubmodRAG {args.dataset} (n={len(questions)}): EM={em:.1f}% F1={f1_avg:.1f}% ({elapsed:.1f}s) ===")
    for line in open(pred_file):
        p = json.loads(line)
        pred = str(p.get("pred_answer", ""))[:50]
        gold = str(p.get("gold_answer", ""))[:50]
        status = "OK" if norm(pred) == norm(gold) else "XX"
        print(f"  {status} PRED: {pred:45s} GOLD: {gold}")


if __name__ == "__main__":
    main()
