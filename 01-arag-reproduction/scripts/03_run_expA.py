#!/usr/bin/env python3
"""
Run Experiment A for one or all datasets by wrapping external/arag batch_runner.

Docs:
- A-RAG batch runner: external/arag/scripts/batch_runner.py
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path("/projects/prjs1800")
ARAG_ROOT = ROOT / "external" / "arag"
EXP_ROOT = ROOT / "msc-thesis" / "01-arag-reproduction"

DATASETS = ["hotpotqa", "musique", "2wikimultihopqa"]

def run_one(dataset: str, workers: int, limit: int | None):
    qfile = EXP_ROOT / "data" / "questions" / f"{dataset}.json"
    outdir = ROOT / "results" / "arag-expA" / dataset
    cfg = EXP_ROOT / "configs" / "arag_qwen25_template.yaml"

    cmd = [
        "python", str(ARAG_ROOT / "scripts" / "batch_runner.py"),
        "--config", str(cfg),
        "--questions", str(qfile),
        "--output", str(outdir),
        "--workers", str(workers),
    ]
    if limit:
        cmd += ["--limit", str(limit)]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="all", choices=["all"] + DATASETS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    targets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in targets:
        run_one(ds, args.workers, args.limit)

if __name__ == "__main__":
    main()