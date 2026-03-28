#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * pct
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sacct_elapsed_raw(job_id: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobIDRaw,State,ElapsedRaw", "-Pn"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        raw_job_id, state, elapsed_raw = parts
        if raw_job_id == job_id:
            return int(elapsed_raw), state
    raise RuntimeError(f"Could not find sacct row for job {job_id}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows: list[dict] = []
    for entry in manifest:
        metrics_dir = Path(entry["metrics_dir"])
        method_filter = entry.get("method_filter")
        per_example_path = metrics_dir / "per_example.jsonl"
        predictions_path = metrics_dir / "predictions.jsonl"
        source_path = per_example_path if per_example_path.exists() else predictions_path
        records = load_jsonl(source_path)
        if method_filter:
            records = [record for record in records if record.get("method") == method_filter]
        if not records:
            continue
        wall_sec, state = sacct_elapsed_raw(str(entry["job_id"]))
        latencies = [float(record["elapsed_sec_total"]) for record in records]
        rows.append(
            {
                "dataset": entry["dataset"],
                "method": entry["method"],
                "job_id": str(entry["job_id"]),
                "state": state,
                "n": len(records),
                "wall_sec": wall_sec,
                "queries_per_sec": round(len(records) / max(wall_sec, 1e-9), 6),
                "median_latency_sec_under_load": round(percentile(latencies, 0.5), 6),
                "p90_latency_sec_under_load": round(percentile(latencies, 0.9), 6),
                "mean_f1": round(sum(float(record["f1"]) for record in records) / len(records), 6),
                "mean_em": round(sum(float(record["em"]) for record in records) / len(records), 6),
            }
        )

    write_csv(Path(args.output), rows)


if __name__ == "__main__":
    main()
