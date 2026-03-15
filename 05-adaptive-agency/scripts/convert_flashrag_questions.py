#!/usr/bin/env python3
"""Convert FlashRAG dataset format to our questions.json format.

FlashRAG format (JSONL):
    {"id": "...", "question": "...", "golden_answers": ["..."], "type": "..."}

Our format (JSON array):
    [{"id": "...", "question": "...", "answer": "...", "source": "...", "question_type": "..."}]

Usage:
    python scripts/convert_flashrag_questions.py \
        --input /scratch-shared/pnair/flashrag/datasets/hotpotqa/dev.jsonl \
        --output data/hotpotqa/questions.json \
        --source hotpotqa

    # Or convert all three at once:
    python scripts/convert_flashrag_questions.py --all \
        --flashrag-dir /scratch-shared/pnair/flashrag/datasets \
        --data-dir data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Mapping from FlashRAG directory names to our directory names
DATASET_MAP = {
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihop",
    "musique": "musique",
}


def convert_one(input_path: Path, output_path: Path, source: str, limit: int | None = None) -> int:
    """Convert a single FlashRAG JSONL to our JSON format."""
    questions = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # FlashRAG uses "golden_answers" (list) or "answer" (string)
            answers = obj.get("golden_answers", [])
            if not answers:
                ans = obj.get("answer", "")
                answers = [ans] if ans else []

            # Use the first golden answer
            answer = answers[0] if answers else ""

            question_type = obj.get("type", "bridge")

            questions.append({
                "id": str(obj.get("id", len(questions))),
                "source": source,
                "question": obj["question"],
                "answer": answer,
                "question_type": question_type,
            })

            if limit and len(questions) >= limit:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    logger.info("Converted %d questions: %s -> %s", len(questions), input_path, output_path)
    return len(questions)


def main():
    parser = argparse.ArgumentParser(description="Convert FlashRAG questions")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--source", type=str, help="Source dataset name")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions")
    parser.add_argument("--all", action="store_true", help="Convert all datasets")
    parser.add_argument(
        "--flashrag-dir",
        default="/scratch-shared/pnair/flashrag/datasets",
        help="FlashRAG datasets directory",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Output data directory",
    )
    args = parser.parse_args()

    if args.all:
        flashrag_dir = Path(args.flashrag_dir)
        data_dir = Path(args.data_dir)
        total = 0

        for flashrag_name, our_name in DATASET_MAP.items():
            input_path = flashrag_dir / flashrag_name / "dev.jsonl"
            if not input_path.exists():
                # Fall back to test.jsonl (e.g., HotPotQA only has test split)
                input_path = flashrag_dir / flashrag_name / "test.jsonl"
            if not input_path.exists():
                logger.warning("Not found: %s (tried dev.jsonl and test.jsonl)", flashrag_dir / flashrag_name)
                continue
            output_path = data_dir / our_name / "questions.json"
            n = convert_one(input_path, output_path, our_name, args.limit)
            total += n

        logger.info("Total: %d questions across %d datasets", total, len(DATASET_MAP))
    else:
        if not args.input or not args.output:
            parser.error("--input and --output required (or use --all)")
        convert_one(
            Path(args.input), Path(args.output),
            args.source or "unknown", args.limit,
        )


if __name__ == "__main__":
    main()
