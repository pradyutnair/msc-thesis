#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_mapping(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        key, value = item.split("=", 1)
        mapping[key] = value
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--n-questions", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--corpus", default="wiki18_100w")
    parser.add_argument("--retriever", default="E5-base-v2 + FAISS")
    parser.add_argument("--ar-model", default="Qwen3-8B")
    parser.add_argument("--dllm-model", default="LLaDA-8B")
    parser.add_argument("--question-spec", action="append", default=[])
    parser.add_argument("--arag-config", action="append", default=[])
    parser.add_argument("--ircot-config", action="append", default=[])
    parser.add_argument("--ar-methods", default="b0,e2_react,ircot")
    parser.add_argument("--dllm-methods", default="baseline,spread,aram,pool,eamd_micro")
    parser.add_argument("--eamd-param", action="append", default=[])
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    qid_root = output_root / "qid_manifests"
    qid_root.mkdir(parents=True, exist_ok=True)

    question_specs = parse_mapping(args.question_spec)
    arag_configs = parse_mapping(args.arag_config)
    ircot_configs = parse_mapping(args.ircot_config)
    eamd_params = parse_mapping(args.eamd_param)

    dataset_manifests: dict[str, dict] = {}
    for dataset, question_path in question_specs.items():
        rows = json.loads(Path(question_path).read_text(encoding="utf-8"))
        selected = rows[: args.n_questions]
        manifest = {
            "dataset": dataset,
            "questions_path": question_path,
            "count": len(selected),
            "qids": [str(item.get("qid") or item.get("id")) for item in selected],
        }
        dataset_manifests[dataset] = manifest
        (qid_root / f"{dataset}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    snapshot = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": args.corpus,
        "retriever": args.retriever,
        "n_questions_per_dataset": args.n_questions,
        "shard_size": args.shard_size,
        "models": {
            "ar": args.ar_model,
            "dllm": args.dllm_model,
        },
        "methods": {
            "ar": [method for method in args.ar_methods.split(",") if method],
            "dllm": [method for method in args.dllm_methods.split(",") if method],
        },
        "question_files": question_specs,
        "arag_configs": arag_configs,
        "ircot_configs": ircot_configs,
        "eamd_micro_config": eamd_params,
    }
    (output_root / "config_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
