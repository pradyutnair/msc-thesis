"""
LLM Judge Evaluation for DNMR Paper
====================================
Uses OpenAI API (GPT-4o-mini) to judge correctness of predicted answers.
Applied UNIFORMLY to ALL methods on ALL datasets.

Reads existing prediction files, does NOT modify them.
Outputs judge results to results/llm_judge/
"""

import json
import os
import sys
import time
import argparse
import string
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/projects/prjs1800/.env")

client = OpenAI()

JUDGE_PROMPT = """You are an expert judge for question answering evaluation.

Given a question, a gold (correct) answer, and a predicted answer, determine if the prediction is correct.

Rules:
- The prediction is CORRECT if it contains the same factual answer as the gold, even if it includes extra explanation or context.
- Accept paraphrases, synonyms, and minor variations (e.g., "NYC" = "New York City", "William Shakespeare" = "Shakespeare").
- Accept if the gold answer appears as a substring within a longer predicted answer.
- For yes/no questions, the prediction must have the same polarity as the gold.
- The prediction is INCORRECT if it gives a different entity, fact, or polarity than the gold.

Respond with ONLY "correct" or "incorrect". Nothing else.

Question: {question}
Gold answer: {gold}
Predicted answer: {prediction}"""

EXTRACT_PROMPT = """Extract ONLY the short factual answer from this response. Copy the exact words from the response — do not rephrase. If the response contains multiple entities, extract only the one that directly answers the question. Output ONLY the extracted answer, nothing else.

Question: {question}
Response: {prediction}"""


def judge_answer(question: str, gold: str, prediction: str, model: str = "gpt-4.1-mini") -> str:
    """Returns 'correct' or 'incorrect'."""
    if not prediction or not prediction.strip():
        return "incorrect"

    prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0,
            )
            result = resp.choices[0].message.content.strip().lower()
            if "correct" in result and "incorrect" not in result:
                return "correct"
            return "incorrect"
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                print(f"  Judge API error: {e}")
                return "error"


def extract_answer(question: str, prediction: str, model: str = "gpt-4.1-mini") -> str:
    """Extract short answer span from verbose prediction."""
    if not prediction or not prediction.strip():
        return ""
    # If already short (< 40 chars), don't extract
    if len(prediction.strip()) < 40:
        return prediction.strip()

    prompt = EXTRACT_PROMPT.format(question=question, prediction=prediction)

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                print(f"  Extract API error: {e}")
                return prediction.strip()


def normalize_answer(text: str) -> str:
    """Normalize short-answer strings before token-overlap scoring."""
    if not text:
        return ""

    text = text.lower().strip()
    text = text.replace("-", " ").replace("/", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_f1(prediction: str, gold: str) -> dict:
    """Compute normalized token-level F1, precision, recall, EM, contain."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    if not pred_tokens or not gold_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "em": 0.0, "contain": 0.0}

    common = set(pred_tokens) & set(gold_tokens)
    n_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)

    precision = n_common / len(pred_tokens) if pred_tokens else 0
    recall = n_common / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    em = 1.0 if pred_norm == gold_norm else 0.0
    contain = 1.0 if gold_norm in pred_norm else 0.0

    return {"f1": f1, "precision": precision, "recall": recall, "em": em, "contain": contain}


def load_all_predictions(dataset: str, base_dir: str) -> dict:
    """Load all LLaDA predictions for a dataset, merging baselines + mixed + completion.
    Returns dict: question_id -> {question, gold, methods: {method_name: answer}}
    """
    questions = {}

    # 1. Load baselines (spread, aram, ispread, iaram)
    for s in range(5):
        path = f"{base_dir}/baselines/llada_{dataset}_baselines_s{s}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        results = data["results"] if isinstance(data, dict) and "results" in data else data
        for q in results:
            qid = q["id"]
            if qid not in questions:
                questions[qid] = {"question": q["question"], "gold": q["gold"], "methods": {}}
            for method in ["spread", "aram", "ispread", "iaram"]:
                if method in q and isinstance(q[method], dict):
                    ans = q[method].get("answer", q[method].get("pred", ""))
                    questions[qid]["methods"][method] = ans

    # 2. Load mixed (baseline, pool, ipool, idnmr, idnmr_2round)
    for s in range(5):
        path = f"{base_dir}/mixed/llada_{dataset}_mix_s{s}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        results = data["results"] if isinstance(data, dict) and "results" in data else data
        for q in results:
            qid = q["id"]
            if qid not in questions:
                questions[qid] = {"question": q["question"], "gold": q["gold"], "methods": {}}
            for method in ["baseline", "pool", "ipool", "idnmr", "idnmr_2round"]:
                if method in q and isinstance(q[method], dict):
                    ans = q[method].get("answer", q[method].get("pred", ""))
                    questions[qid]["methods"][method] = ans

    # 3. Load completion files (fill gaps for mixed methods)
    comp_dir = f"{base_dir}/completion"
    if os.path.exists(comp_dir):
        for fname in sorted(os.listdir(comp_dir)):
            if not fname.startswith(f"llada_{dataset}") or not fname.endswith(".json"):
                continue
            path = os.path.join(comp_dir, fname)
            with open(path) as f:
                data = json.load(f)
            results = data["results"] if isinstance(data, dict) and "results" in data else data
            for q in results:
                qid = q["id"]
                if qid not in questions:
                    questions[qid] = {"question": q["question"], "gold": q["gold"], "methods": {}}
                for method in ["baseline", "pool", "ipool", "idnmr", "idnmr_2round"]:
                    if method in q and isinstance(q[method], dict):
                        ans = q[method].get("answer", q[method].get("pred", ""))
                        # Only fill if not already present
                        if method not in questions[qid]["methods"]:
                            questions[qid]["methods"][method] = ans

    return questions


def _is_method_result_block(v: object) -> bool:
    return isinstance(v, dict) and ("answer" in v or "pred" in v)


def load_predictions_from_results_json(path: str) -> dict:
    """Load predictions from a single results JSON (top-level ``results`` list).
    Method columns are any keys whose value is a dict with ``answer`` or ``pred``.
    """
    path = os.path.abspath(path)
    with open(path) as f:
        data = json.load(f)
    rows = data["results"] if isinstance(data, dict) and "results" in data else data
    questions = {}
    for q in rows:
        qid = q["id"]
        methods = {}
        for k, v in q.items():
            if _is_method_result_block(v):
                methods[k] = v.get("answer", v.get("pred", ""))
        questions[qid] = {
            "question": q["question"],
            "gold": q["gold"],
            "methods": methods,
        }
    return questions


def run_judge(dataset: str, base_dir: str, out_dir: str, model: str = "gpt-4.1-mini",
              do_extract: bool = True, limit: int = None, results_json: str = None,
              out_name: str = None):
    """Run LLM judge + optional extraction on all methods for a dataset."""

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} | Model: {model} | Extract: {do_extract}")
    if results_json:
        print(f"Results file: {results_json}")
    print(f"{'='*60}")

    if results_json:
        questions = load_predictions_from_results_json(results_json)
    else:
        questions = load_all_predictions(dataset, base_dir)
    print(f"Loaded {len(questions)} questions")

    # Sort by ID for determinism
    qids = sorted(questions.keys())
    if limit:
        qids = qids[:limit]
        print(f"Limited to {limit} questions")

    # Track all methods seen
    all_methods = set()
    for qid in qids:
        all_methods.update(questions[qid]["methods"].keys())
    all_methods = sorted(all_methods)
    print(f"Methods: {all_methods}")

    # Count total API calls needed
    total_calls = sum(len(questions[qid]["methods"]) for qid in qids)
    if do_extract:
        total_calls *= 2  # judge + extract
    print(f"Total API calls: ~{total_calls}")

    # Output file (save incrementally)
    os.makedirs(out_dir, exist_ok=True)
    if out_name:
        out_filename = out_name if out_name.endswith(".json") else f"{out_name}.json"
    elif results_json:
        stem = Path(results_json).stem
        out_filename = f"llm_judge_{stem}.json"
    else:
        out_filename = f"llm_judge_{dataset}.json"
    out_path = os.path.join(out_dir, out_filename)

    # Load existing results if resuming
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = {r["id"]: r for r in json.load(f)}
        print(f"Resuming: {len(existing)} already done")

    results = list(existing.values())
    done_ids = set(existing.keys())

    n_done = 0
    n_total = len(qids)

    for qid in tqdm(qids, desc="Processing questions"):
        if qid in done_ids:
            n_done += 1
            continue

        q = questions[qid]
        entry = {
            "id": qid,
            "question": q["question"],
            "gold": q["gold"],
            "methods": {}
        }

        for method in all_methods:
            answer = q["methods"].get(method, "")

            # Judge
            verdict = judge_answer(q["question"], q["gold"], answer, model=model)

            method_result = {
                "answer": answer,
                "judge": verdict,
            }

            # Extract + recompute F1
            if do_extract and answer:
                extracted = extract_answer(q["question"], answer, model=model)
                extracted_metrics = compute_f1(extracted, q["gold"])
                method_result["extracted_answer"] = extracted
                method_result["extracted_f1"] = extracted_metrics["f1"]
                method_result["extracted_em"] = extracted_metrics["em"]
                method_result["extracted_precision"] = extracted_metrics["precision"]
                method_result["extracted_recall"] = extracted_metrics["recall"]
                method_result["extracted_contain"] = extracted_metrics["contain"]

            entry["methods"][method] = method_result

        results.append(entry)
        n_done += 1

        # Save every 10 questions
        if n_done % 10 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

            # Print progress
            method_accs = defaultdict(list)
            for r in results:
                for m, v in r["methods"].items():
                    if v["judge"] in ("correct", "incorrect"):
                        method_accs[m].append(1 if v["judge"] == "correct" else 0)

            status = " | ".join(f"{m}={sum(v)/len(v):.3f}" for m, v in sorted(method_accs.items()) if v)
            print(f"[{n_done}/{n_total}] {status}")

    # Final save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n--- {dataset} SUMMARY ---")
    method_stats = defaultdict(lambda: {"judge_correct": 0, "judge_total": 0,
                                         "ext_f1_sum": 0, "ext_em_sum": 0, "ext_n": 0})
    for r in results:
        for m, v in r["methods"].items():
            s = method_stats[m]
            if v["judge"] in ("correct", "incorrect"):
                s["judge_total"] += 1
                if v["judge"] == "correct":
                    s["judge_correct"] += 1
            if "extracted_f1" in v:
                s["ext_f1_sum"] += v["extracted_f1"]
                s["ext_em_sum"] += v["extracted_em"]
                s["ext_n"] += 1

    print(f"{'Method':<15} {'Judge Acc':>10} {'Ext F1':>10} {'Ext EM':>10} {'N':>6}")
    print("-" * 55)
    for m in sorted(method_stats.keys()):
        s = method_stats[m]
        acc = s["judge_correct"] / s["judge_total"] if s["judge_total"] else 0
        ext_f1 = s["ext_f1_sum"] / s["ext_n"] if s["ext_n"] else 0
        ext_em = s["ext_em_sum"] / s["ext_n"] if s["ext_n"] else 0
        n = s["judge_total"]
        print(f"{m:<15} {acc:>10.3f} {ext_f1:>10.3f} {ext_em:>10.3f} {n:>6}")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["musique", "hotpotqa", "2wikimultihopqa"],
        help="Dataset name (required unless --results-json is set; used in logs and default output name).",
    )
    parser.add_argument("--base-dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--out-dir", default="/projects/prjs1800/msc-thesis/07-daes/results/llm_judge")
    parser.add_argument(
        "--out-name",
        default=None,
        help="Output JSON filename (under --out-dir). Default: llm_judge_<dataset>.json or llm_judge_<results-json stem>.json",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Single results file with a top-level results[] list; method columns are dicts with answer/pred.",
    )
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--no-extract", action="store_true", help="Skip answer extraction")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    args = parser.parse_args()

    if args.results_json:
        dataset = args.dataset or "musique"
    else:
        if not args.dataset:
            parser.error("--dataset is required when --results-json is not set")
        dataset = args.dataset

    run_judge(
        dataset=dataset,
        base_dir=args.base_dir,
        out_dir=args.out_dir,
        model=args.model,
        do_extract=not args.no_extract,
        limit=args.limit,
        results_json=args.results_json,
        out_name=args.out_name,
    )
