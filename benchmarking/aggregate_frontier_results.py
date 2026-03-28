#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarking.qa_benchmark import write_dataset_artifacts, write_jsonl, write_pairwise_artifacts


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    raise ValueError(f"Unsupported benchmark input: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records: list[dict] = []
    for raw_path in args.inputs:
        records.extend(load_records(Path(raw_path)))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "merged_per_example.jsonl", records)
    write_dataset_artifacts(records, output_dir)
    write_pairwise_artifacts(
        records,
        output_dir,
        comparisons=[
            ("eamd_micro", "aram"),
            ("eamd_micro", "ircot"),
            ("aram", "spread"),
            ("ircot", "e2_react"),
        ],
    )


if __name__ == "__main__":
    main()
