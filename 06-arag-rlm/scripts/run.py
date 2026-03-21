#!/usr/bin/env python3
"""CLI entrypoint for the RLM ARAG runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from arag.core.config import Config
from arag.runner import BatchRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RLM-style ARAG batch runner")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file")
    parser.add_argument("--questions", "-q", default=None, help="Override questions file")
    parser.add_argument("--output", "-o", default=None, help="Override output directory")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Optional question limit")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    runner = BatchRunner(
        config=config,
        questions_file=args.questions,
        output_dir=args.output,
        limit=args.limit,
    )
    predictions_path = runner.run()
    print(predictions_path)


if __name__ == "__main__":
    main()
