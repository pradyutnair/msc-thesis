#!/usr/bin/env python3
"""Prepare A-RAG question JSON from FlashRAG-style JSON/JSONL files.

Why this exists:
- rag_test already ships question files, so you usually do not need this.
- Use this when you want to swap in a different split/file and keep A-RAG format.

Expected output format (A-RAG batch_runner input):
[
  {"qid": "...", "question": "...", "answer": "..."}
]

Docs:
- A-RAG batch runner: /projects/prjs1800/external/arag/scripts/batch_runner.py
- FlashRAG datasets (local): /projects/prjs1800/datasets/flashrag/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator

DATASET_DEFAULTS: Dict[str, str] = {
    "hotpotqa": "/projects/prjs1800/datasets/flashrag/hotpotqa/test.jsonl",
    "musique": "/projects/prjs1800/datasets/flashrag/musique/dev.jsonl",
    "2wikimultihopqa": "/projects/prjs1800/datasets/flashrag/2wikimultihopqa/test.jsonl",
}


def read_json_or_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield records from .jsonl or list-based .json files."""
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in JSON file: {path}")
    for row in data:
        if isinstance(row, dict):
            yield row


def normalize_item(idx: int, row: Dict[str, Any]) -> Dict[str, str]:
    """Normalize one source row into A-RAG question schema."""
    qid = row.get("qid") or row.get("id") or row.get("_id") or str(idx)
    question = row.get("question") or row.get("query") or ""

    answer: Any = (
        row.get("golden_answers")
        or row.get("answer")
        or row.get("gold_answer")
        or row.get("answers")
        or ""
    )
    if isinstance(answer, list):
        answer = answer[0] if answer else ""

    return {"qid": str(qid), "question": str(question), "answer": str(answer)}


def convert_rows(rows: Iterable[Dict[str, Any]], limit: int | None = None) -> list[Dict[str, str]]:
    """Convert an iterable of source rows into normalized A-RAG rows."""
    out: list[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        out.append(normalize_item(idx, row))
        if limit is not None and len(out) >= limit:
            break
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare A-RAG questions JSON")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS.keys()), required=True)
    parser.add_argument("--input", type=str, default=None, help="Override source file path")
    parser.add_argument("--output", type=str, required=True, help="Output .json path")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input or DATASET_DEFAULTS[args.dataset])
    out_path = Path(args.output)

    rows = convert_rows(read_json_or_jsonl(in_path), limit=args.limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} questions -> {out_path}")


if __name__ == "__main__":
    main()
