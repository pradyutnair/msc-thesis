#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    shard_size = (len(data) + args.num_shards - 1) // args.num_shards
    for idx in range(args.num_shards):
        shard = data[idx * shard_size : (idx + 1) * shard_size]
        if not shard:
            continue
        out_path = output_dir / f"shard_{idx:02d}.json"
        out_path.write_text(json.dumps(shard, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out_path)


if __name__ == "__main__":
    main()
