import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    shard_paths = sorted(glob.glob(args.input_glob))
    if not shard_paths:
        raise ValueError(f"No shard files matched: {args.input_glob}")

    payloads = [json.load(open(path)) for path in shard_paths]
    payloads.sort(key=lambda item: item["metadata"]["start_idx"])

    methods = payloads[0]["metadata"]["methods"]
    results = []
    for payload in payloads:
        results.extend(payload["results"])

    totals = {name: {"f1": 0.0, "em": 0.0, "contain": 0.0} for name in methods}
    for row in results:
        for key in methods:
            totals[key]["f1"] += row[key]["f1"]
            totals[key]["em"] += row[key]["em"]
            totals[key]["contain"] += float(row[key]["contain"])

    summary = {
        key: {
            "f1": totals[key]["f1"] / len(results),
            "em": totals[key]["em"] / len(results),
            "contain": totals[key]["contain"] / len(results),
        }
        for key in methods
    }

    total_elapsed = sum(row["elapsed_sec"] for row in results)
    merged = {
        "metadata": {
            "dataset": payloads[0]["metadata"]["dataset"],
            "methods": methods,
            "n_shards": len(payloads),
            "n_results": len(results),
            "input_glob": args.input_glob,
            "total_elapsed_sec": round(total_elapsed, 2),
            "avg_elapsed_sec": round(total_elapsed / len(results), 4),
            "shards": [
                {
                    "path": path,
                    "start_idx": payload["metadata"]["start_idx"],
                    "end_idx": payload["metadata"]["end_idx"],
                    "n_questions": payload["metadata"]["n_questions"],
                }
                for path, payload in zip(shard_paths, payloads)
            ],
        },
        "summary": summary,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f)

    print("Merged shards", flush=True)
    for key, values in summary.items():
        print(
            f"  {key:12s} F1={values['f1']:.3f} EM={values['em']:.3f} contain={values['contain']:.3f}",
            flush=True,
        )
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
