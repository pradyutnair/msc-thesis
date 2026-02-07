#!/usr/bin/env python3
"""
Evaluate MA-RAG outputs on QA benchmarks.

Computes Exact Match (EM) and F1 scores.
"""

import json
import os
import re
import string
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Check if prediction exactly matches ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_metrics(prediction: str, ground_truths: List[str]) -> Dict[str, float]:
    """Compute best EM and F1 against multiple ground truths."""
    em = max(exact_match_score(prediction, gt) for gt in ground_truths)
    f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    return {"em": em, "f1": f1}


def load_marag_results(results_dir: str) -> Dict[str, Any]:
    """Load MA-RAG output files from a directory."""
    results = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            question_id = json_file.stem
            results[question_id] = data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def load_ground_truth(dataset_path: str) -> Dict[str, List[str]]:
    """Load ground truth answers from FlashRAG dataset."""
    ground_truth = {}
    
    with open(dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            qid = item["id"]
            answers = item.get("golden_answers", [])
            ground_truth[qid] = answers
    
    return ground_truth


def extract_answer_from_marag_output(output: Dict) -> str:
    """Extract the final answer from MA-RAG output structure."""
    # MA-RAG stores answer in past_exp[-1]["plan_summary"]["answer"]
    try:
        past_exp = output.get("past_exp", [])
        if past_exp:
            last_exp = past_exp[-1]
            plan_summary = last_exp.get("plan_summary", {})
            answer = plan_summary.get("answer", "")
            return answer
    except (KeyError, IndexError, TypeError):
        pass
    return ""


def evaluate(results_dir: str, dataset_path: str) -> Dict[str, float]:
    """
    Evaluate MA-RAG results against ground truth.
    
    Args:
        results_dir: Path to MA-RAG output directory (e.g., plan_rag_extract_qwen3-8b_hotpotqa/)
        dataset_path: Path to FlashRAG dataset JSONL file
    
    Returns:
        Dictionary with em, f1, and count metrics
    """
    # Load data
    results = load_marag_results(results_dir)
    ground_truth = load_ground_truth(dataset_path)
    
    print(f"Loaded {len(results)} MA-RAG results")
    print(f"Loaded {len(ground_truth)} ground truth entries")
    
    # Compute metrics
    total_em = 0.0
    total_f1 = 0.0
    evaluated = 0
    missing = 0
    
    for qid, output in results.items():
        if qid not in ground_truth:
            missing += 1
            continue
        
        prediction = extract_answer_from_marag_output(output)
        gt_answers = ground_truth[qid]
        
        if not gt_answers:
            continue
        
        metrics = compute_metrics(prediction, gt_answers)
        total_em += metrics["em"]
        total_f1 += metrics["f1"]
        evaluated += 1
    
    if evaluated == 0:
        print("Warning: No samples could be evaluated!")
        return {"em": 0.0, "f1": 0.0, "count": 0}
    
    avg_em = total_em / evaluated * 100
    avg_f1 = total_f1 / evaluated * 100
    
    return {
        "em": avg_em,
        "f1": avg_f1,
        "count": evaluated,
        "missing": missing
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MA-RAG results")
    parser.add_argument("--results_dir", required=True, help="Path to MA-RAG results directory")
    parser.add_argument("--dataset", required=True, choices=["hotpotqa", "nq", "triviaqa", "2wiki"],
                       help="Dataset name")
    parser.add_argument("--dataset_path", default=None,
                       help="Path to dataset JSONL (auto-detected if not provided)")
    args = parser.parse_args()
    
    # Auto-detect dataset path
    if args.dataset_path is None:
        base_path = "/projects/prjs1800/datasets/flashrag"
        dataset_map = {
            "hotpotqa": "hotpotqa/test.jsonl",
            "nq": "nq/test.jsonl",
            "triviaqa": "triviaqa/test.jsonl",
            "2wiki": "2wikimultihopqa/test.jsonl"
        }
        args.dataset_path = os.path.join(base_path, dataset_map[args.dataset])
    
    print(f"\n=== Evaluating MA-RAG on {args.dataset} ===")
    print(f"Results: {args.results_dir}")
    print(f"Dataset: {args.dataset_path}")
    print()
    
    metrics = evaluate(args.results_dir, args.dataset_path)
    
    print(f"\n=== Results ===")
    print(f"Exact Match (EM): {metrics['em']:.2f}%")
    print(f"F1 Score:         {metrics['f1']:.2f}%")
    print(f"Evaluated:        {metrics['count']} samples")
    if metrics.get('missing', 0) > 0:
        print(f"Missing GT:       {metrics['missing']} samples")


if __name__ == "__main__":
    main()
