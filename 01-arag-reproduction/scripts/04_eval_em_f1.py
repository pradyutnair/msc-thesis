#!/usr/bin/env python3
"""
Comprehensive evaluator for A-RAG predictions.jsonl.

Metrics:
  SQuAD-style:  EM, F1, Precision, Recall
  A-RAG paper:  Contains-match accuracy, LLM-judge accuracy
  Agent stats:  Finish rate, avg loops, avg tool calls

LLM-judge requires a running vLLM instance (or any OpenAI-compatible endpoint).
Pass --llm-judge-url to enable. If omitted, LLM judge metrics are skipped.
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path


# ── SQuAD-style helpers ──────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _token_scores(pred: str, gold: str):
    """Return (precision, recall, f1) at token level."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def contains_match(pred: str, gold: str) -> float:
    return float(normalize_answer(gold) in normalize_answer(pred))


def score_single(pred: str, gold_answers):
    """Compute all token-level metrics, taking best over gold answers."""
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    gold_answers = [g for g in gold_answers if g.strip()]
    if not gold_answers:
        return {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "contains": 0.0}

    best_em = max(exact_match(pred, g) for g in gold_answers)
    best_contains = max(contains_match(pred, g) for g in gold_answers)

    # For P/R/F1, pick the gold answer giving highest F1
    best_f1, best_p, best_r = 0.0, 0.0, 0.0
    for g in gold_answers:
        p, r, f = _token_scores(pred, g)
        if f > best_f1:
            best_f1, best_p, best_r = f, p, r

    return {"em": best_em, "f1": best_f1, "precision": best_p, "recall": best_r, "contains": best_contains}


# ── LLM Judge ────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an evaluation judge. Given a question, the gold (correct) answer, and a predicted answer, determine if the predicted answer is semantically correct.

The predicted answer is correct if:
- It conveys the same meaning as the gold answer
- It may use different wording but refers to the same entity, fact, or value
- For yes/no questions, it must match the gold answer's polarity

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Respond with ONLY "correct" or "incorrect"."""


def llm_judge_batch(rows, base_url, model, batch_size=20):
    """Score predictions using LLM-as-judge. Returns list of 0/1 scores."""
    import requests

    scores = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        for r in batch:
            question = r.get("question", "")
            pred = r.get("pred_answer", "")
            gold = r.get("gold_answer", "")

            if pred.startswith("Error:"):
                scores.append(0)
                continue

            prompt = JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": "Bearer dummy"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 10,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
                scores.append(1 if "correct" in answer else 0)
            except Exception as e:
                print(f"  LLM judge error: {e}", file=sys.stderr)
                scores.append(0)

        print(f"  LLM judge: {i + len(batch)}/{len(rows)}", file=sys.stderr)

    return scores


# ── Main ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--llm-judge-url", default=None,
                    help="OpenAI-compatible base URL for LLM judge (e.g. http://127.0.0.1:8000/v1)")
    p.add_argument("--llm-judge-model", default="Qwen2.5-7B-Instruct")
    args = p.parse_args()

    # Load predictions (deduplicate by qid)
    rows = []
    seen_qids = set()
    with open(args.predictions, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            qid = d.get("qid", "")
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            rows.append(d)

    # Token-level metrics
    metric_keys = ["em", "f1", "precision", "recall", "contains"]
    metrics = {k: [] for k in metric_keys}
    errors = 0
    total_loops = 0
    total_tools = 0
    finish_called_count = 0

    for r in rows:
        pred = r.get("pred_answer", "")
        gold = r.get("gold_answer", "") or r.get("golden_answers", "") or r.get("answer", "")
        total_loops += r.get("loops", 0)
        total_tools += len(r.get("trajectory", []))
        if r.get("finish_called", False):
            finish_called_count += 1

        if pred.startswith("Error:"):
            errors += 1
            for k in metric_keys:
                metrics[k].append(0.0)
            continue

        scores = score_single(pred, gold)
        for k in metric_keys:
            metrics[k].append(scores[k])

    n = len(rows)
    summary = {
        "count": n,
        "unique_questions": len(seen_qids),
        "errors": errors,
        "error_rate": round(errors / n, 4) if n else 0,
        # SQuAD-style
        "em": round(sum(metrics["em"]) / n, 4) if n else 0,
        "f1": round(sum(metrics["f1"]) / n, 4) if n else 0,
        "precision": round(sum(metrics["precision"]) / n, 4) if n else 0,
        "recall": round(sum(metrics["recall"]) / n, 4) if n else 0,
        # A-RAG style
        "contains_match": round(sum(metrics["contains"]) / n, 4) if n else 0,
        # Agent behavior
        "finish_rate": round(finish_called_count / n, 4) if n else 0,
        "avg_loops": round(total_loops / n, 2) if n else 0,
        "avg_tool_calls": round(total_tools / n, 2) if n else 0,
    }

    # LLM judge (optional)
    if args.llm_judge_url:
        print("Running LLM-as-judge evaluation...", file=sys.stderr)
        judge_scores = llm_judge_batch(rows, args.llm_judge_url, args.llm_judge_model)
        summary["llm_judge_accuracy"] = round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else 0
    else:
        summary["llm_judge_accuracy"] = None

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
