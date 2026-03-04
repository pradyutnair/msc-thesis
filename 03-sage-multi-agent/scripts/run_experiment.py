#!/usr/bin/env python3
"""CLI entry point for running MA²RAG experiments."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASETS = {
    "hotpotqa": {
        "questions": str(PROJECT_ROOT / "data" / "hotpotqa" / "questions.json"),
        "chunks": str(PROJECT_ROOT / "data" / "hotpotqa" / "chunks.json"),
        "index_dir": str(PROJECT_ROOT / "data" / "hotpotqa" / "index_e5_base_v2"),
        "name": "HotpotQA",
    },
    "musique": {
        "questions": str(PROJECT_ROOT / "data" / "musique" / "questions.json"),
        "chunks": str(PROJECT_ROOT / "data" / "musique" / "chunks.json"),
        "index_dir": str(PROJECT_ROOT / "data" / "musique" / "index_e5_base_v2"),
        "name": "MuSiQue",
    },
    "2wiki": {
        "questions": str(PROJECT_ROOT / "data" / "2wikimultihop" / "questions.json"),
        "chunks": str(PROJECT_ROOT / "data" / "2wikimultihop" / "chunks.json"),
        "index_dir": str(PROJECT_ROOT / "data" / "2wikimultihop" / "index_e5_base_v2"),
        "name": "2WikiMultihopQA",
    },
}

EXPERIMENTS = {
    "m1": {"config": "m1_multi_agent.yaml", "desc": "Multi-agent, no cache"},
    "m2": {"config": "m2_doc_cache.yaml", "desc": "Multi-agent + doc cache"},
    "m3": {"config": "m3_kv_cache.yaml", "desc": "Multi-agent + doc + KV cache"},
    "m4": {"config": "m4_single_2x.yaml", "desc": "Single-agent 2x iterations"},
    "a1": {"config": "ablations/a1_no_decomposer.yaml", "desc": "No decomposer"},
    "a2": {"config": "ablations/a2_no_aggregator.yaml", "desc": "No aggregator"},
    "a3": {"config": "ablations/a3_sequential.yaml", "desc": "Sequential dispatch"},
    "a4": {"config": "ablations/a4_no_verify.yaml", "desc": "No self-verification"},
    "sage": {"config": "sage.yaml", "desc": "SAGE multi-agent"},
}

PILOT_LIMIT = 100
FULL_LIMIT = 1000


def run_experiment(
    experiment: str,
    dataset: str,
    pilot: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Run a single (experiment, dataset) combination."""
    exp = EXPERIMENTS[experiment]
    ds = DATASETS[dataset]

    config_path = CONFIGS_DIR / exp["config"]
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    limit = PILOT_LIMIT if pilot else FULL_LIMIT
    tag = "pilot" if pilot else "full"
    output_dir = RESULTS_DIR / experiment / dataset / tag

    logger.info(
        "Experiment: %s (%s) | Dataset: %s | Limit: %d | Output: %s",
        experiment,
        exp["desc"],
        ds["name"],
        limit,
        output_dir,
    )

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "multi_agent_runner.py"),
        "--config",
        str(config_path),
        "--questions",
        ds["questions"],
        "--chunks-file",
        ds["chunks"],
        "--index-dir",
        ds["index_dir"],
        "--output",
        str(output_dir),
        "--limit",
        str(limit),
    ]
    if verbose:
        cmd.append("--verbose")

    if dry_run:
        logger.info("DRY RUN: %s", " ".join(cmd))
        return

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error(
            "Experiment %s/%s failed with return code %d",
            experiment,
            dataset,
            result.returncode,
        )
    else:
        logger.info("Experiment %s/%s completed successfully", experiment, dataset)


def main():
    parser = argparse.ArgumentParser(description="Run MA²RAG experiments")
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment ID",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset name or 'all'",
    )
    parser.add_argument("--pilot", action="store_true", help="Run pilot (100 Qs)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_experiment(
            experiment=args.experiment,
            dataset=ds,
            pilot=args.pilot,
            verbose=args.verbose,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
