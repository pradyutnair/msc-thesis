#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _merge_jsonl(paths: list[Path], output_path: Path) -> None:
    seen = set()
    with output_path.open("w", encoding="utf-8") as out_f:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = record.get("qid") or record.get("id")
                    if qid in seen:
                        continue
                    seen.add(qid)
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["predictions.jsonl", "decompositions.jsonl", "autonomy_trace.jsonl", "worker_messages.jsonl", "claim_graph.jsonl"]:
        shard_paths = sorted(input_dir.glob(f"*/{name}"))
        if shard_paths:
            _merge_jsonl(shard_paths, output_dir / name)


if __name__ == "__main__":
    main()
