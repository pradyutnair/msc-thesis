"""
LLM Judge Evaluation for Dream-7B predictions.
Same judge logic as llm_judge_eval.py but loads Dream result files.
"""

import json
import os
import sys
import time
import glob
import argparse
from collections import defaultdict

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


def _backoff_wait(attempt):
    """Exponential backoff wait."""
    import time as _t
    _t.sleep(2 ** attempt)


def judge_answer(question, gold, prediction, model="gpt-4.1-mini"):
    if not prediction or not prediction.strip():
        return "incorrect"
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=5, temperature=0,
            )
            result = resp.choices[0].message.content.strip().lower()
            if "correct" in result and "incorrect" not in result:
                return "correct"
            return "incorrect"
        except Exception as e:
            if attempt < 2:
                _backoff_wait(attempt)
            else:
                print(f"  Judge API error: {e}")
                return "error"


def extract_answer(question, prediction, model="gpt-4.1-mini"):
    if not prediction or not prediction.strip():
        return ""
    if len(prediction.strip()) < 40:
        return prediction.strip()
    prompt = EXTRACT_PROMPT.format(question=question, prediction=prediction)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                max_tokens=50, temperature=0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                _backoff_wait(attempt)
            else:
                return prediction.strip()


def compute_f1(prediction, gold):
    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "em": 0.0, "contain": 0.0}
    common = set(pred_tokens) & set(gold_tokens)
    n_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = n_common / len(pred_tokens) if pred_tokens else 0
    recall = n_common / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    em = 1.0 if prediction.lower().strip() == gold.lower().strip() else 0.0
    contain = 1.0 if gold.lower() in prediction.lower() else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "em": em, "contain": contain}


def load_dream_predictions(dataset, base_dir):
    questions = {}
    for f in sorted(glob.glob(f"{base_dir}/idnmr/dream_{dataset}_idnmr_1k_s*.json")):
        data = json.load(open(f))
        for r in data.get("results", []):
            qid = r["id"]
            if qid not in questions:
                questions[qid] = {"question": r["question"], "gold": r["gold"], "methods": {}}
            for method in ["baseline", "pool", "ipool", "idnmr", "idnmr_2round"]:
                if method in r and isinstance(r[method], dict):
                    questions[qid]["methods"][method] = r[method].get("answer", "")
    for f in sorted(glob.glob(f"{base_dir}/baselines/dream_{dataset}_baselines_s*.json")):
        data = json.load(open(f))
        for r in data.get("results", []):
            qid = r["id"]
            if qid not in questions:
                questions[qid] = {"question": r["question"], "gold": r["gold"], "methods": {}}
            for method in ["spread", "aram", "ispread", "iaram"]:
                if method in r and isinstance(r[method], dict):
                    questions[qid]["methods"][method] = r[method].get("answer", "")
    return questions


def run_judge(dataset, base_dir, out_dir, model="gpt-4.1-mini", do_extract=True, limit=None):
    print(f"\n{'='*60}")
    print(f"Dream Judge | Dataset: {dataset} | Model: {model}")
    print(f"{'='*60}")

    questions = load_dream_predictions(dataset, base_dir)
    print(f"Loaded {len(questions)} questions")

    qids = sorted(questions.keys())
    if limit:
        qids = qids[:limit]

    all_methods = sorted(set(m for qid in qids for m in questions[qid]["methods"]))
    print(f"Methods: {all_methods}")

    total_calls = sum(len(questions[qid]["methods"]) for qid in qids)
    if do_extract:
        total_calls *= 2
    print(f"Total API calls: ~{total_calls}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"llm_judge_dream_{dataset}.json")

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = {r["id"]: r for r in json.load(f)}
        print(f"Resuming: {len(existing)} already done")

    results = list(existing.values())
    done_ids = set(existing.keys())
    n_done = len(done_ids)
    n_total = len(qids)

    for qid in qids:
        if qid in done_ids:
            continue

        q = questions[qid]
        entry = {"id": qid, "question": q["question"], "gold": q["gold"], "methods": {}}

        for method in all_methods:
            answer = q["methods"].get(method, "")
            verdict = judge_answer(q["question"], q["gold"], answer, model=model)
            method_result = {"answer": answer, "judge": verdict}

            if do_extract and answer:
                extracted = extract_answer(q["question"], answer, model=model)
                ext_metrics = compute_f1(extracted, q["gold"])
                method_result["extracted_answer"] = extracted
                method_result["extracted_f1"] = ext_metrics["f1"]
                method_result["extracted_em"] = ext_metrics["em"]
                method_result["extracted_precision"] = ext_metrics["precision"]
                method_result["extracted_recall"] = ext_metrics["recall"]
                method_result["extracted_contain"] = ext_metrics["contain"]

            entry["methods"][method] = method_result

        results.append(entry)
        n_done += 1

        if n_done % 10 == 0:
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            method_accs = defaultdict(list)
            for r in results:
                for m, v in r["methods"].items():
                    if v["judge"] in ("correct", "incorrect"):
                        method_accs[m].append(1 if v["judge"] == "correct" else 0)
            status = " | ".join(f"{m}={sum(v)/len(v):.3f}" for m, v in sorted(method_accs.items()) if v)
            print(f"[{n_done}/{n_total}] {status}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- Dream {dataset} SUMMARY ---")
    method_stats = defaultdict(lambda: {"jc": 0, "jn": 0, "ef1": 0, "en": 0})
    for r in results:
        for m, v in r["methods"].items():
            s = method_stats[m]
            if v["judge"] in ("correct", "incorrect"):
                s["jn"] += 1
                if v["judge"] == "correct":
                    s["jc"] += 1
            if "extracted_f1" in v:
                s["ef1"] += v["extracted_f1"]
                s["en"] += 1
    for m in sorted(method_stats):
        s = method_stats[m]
        acc = s["jc"]/s["jn"] if s["jn"] else 0
        ef1 = s["ef1"]/s["en"] if s["en"] else 0
        print(f"  {m:<15} Judge={acc:.3f} ExtF1={ef1:.3f} N={s['jn']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["musique", "hotpotqa", "2wikimultihopqa"])
    parser.add_argument("--base-dir", default="/projects/prjs1800/msc-thesis/07-daes/results")
    parser.add_argument("--out-dir", default="/projects/prjs1800/msc-thesis/07-daes/results/llm_judge")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_judge(args.dataset, args.base_dir, args.out_dir, args.model, not args.no_extract, args.limit)
