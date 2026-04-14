import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    overview = {}
    for path in args.inputs:
        payload = json.load(open(path))
        dataset = payload["metadata"]["dataset"]
        overview[dataset] = {
            "metadata": payload["metadata"],
            "summary": payload["summary"],
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(overview, f)

    for dataset, payload in overview.items():
        print(dataset, flush=True)
        for method, metrics in payload["summary"].items():
            print(
                f"  {method:12s} F1={metrics['f1']:.3f} EM={metrics['em']:.3f} contain={metrics['contain']:.3f}",
                flush=True,
            )
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
