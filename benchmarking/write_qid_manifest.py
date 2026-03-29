#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    end_idx = args.end_idx if args.end_idx is not None else len(questions)
    selected = questions[args.start_idx:end_idx]
    manifest = {
        "dataset": args.dataset,
        "questions_path": args.questions,
        "start_idx": args.start_idx,
        "end_idx": end_idx,
        "count": len(selected),
        "qids": [str(item.get("qid") or item.get("id")) for item in selected],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
