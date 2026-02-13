#!/usr/bin/env python3
"""
Prepare A-RAG question files from local datasets.

Docs:
- A-RAG expected question format: external/arag/scripts/batch_runner.py
- MuSiQue/HotpotQA/2Wiki source files in /projects/prjs1800/datasets and /projects/prjs1800/datasets/flashrag
"""
import argparse
import json
from pathlib import Path

DATASET_DEFAULTS = {
    "hotpotqa": "/projects/prjs1800/datasets/flashrag/hotpotqa/test.jsonl",
    # flashrag currently has train/dev for musique; use dev unless you add test
    "musique": "/projects/prjs1800/datasets/flashrag/musique/dev.jsonl",
    "2wikimultihopqa": "/projects/prjs1800/datasets/flashrag/2wikimultihopqa/test.jsonl",
}

def read_json_or_jsonl(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        for x in data:
            yield x

def normalize_item(i, row):
    qid = row.get("qid") or row.get("id") or row.get("_id") or str(i)
    question = row.get("question") or row.get("query") or ""
    answer = row.get("answer") or row.get("gold_answer") or row.get("answers") or ""
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return {"qid": str(qid), "question": question, "answer": str(answer)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_DEFAULTS.keys()))
    p.add_argument("--input", default=None, help="Override input file path")
    p.add_argument("--output", required=True, help="Output JSON file (list of dicts)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    in_path = Path(args.input or DATASET_DEFAULTS[args.dataset])
    rows = []
    for i, row in enumerate(read_json_or_jsonl(in_path)):
        rows.append(normalize_item(i, row))
        if args.limit and len(rows) >= args.limit:
            break

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} items -> {out}")

if __name__ == "__main__":
    main()