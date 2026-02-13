"""Re-evaluate Day 5 Reranker+CoT results with fixed answer extraction regex.

Reads saved intermediate_data.json (contains raw_cot_output), applies fixed
regex that doesn't truncate on internal periods (Dr., U.S.A., 3.14, etc.),
and re-computes EM/F1.

Usage:
    python reevaluate_cot.py --results_dir /projects/prjs1800/results/day5/hotpotqa_2026_02_07_12_43_reranker_cot_qwen25_hotpotqa
    python reevaluate_cot.py --results_dir /projects/prjs1800/results/day5/musique_2026_02_07_13_23_reranker_cot_qwen25_musique
"""

import argparse
import json
import os
import re
import string
from collections import Counter


# ── Fixed answer extraction ─────────────────────────────────────────────────

def _clean_extracted(answer):
    """Clean extracted answer: remove verbose framing, keep only the core answer."""
    answer = answer.strip().rstrip(".")
    answer = re.sub(r"^that\s+(the\s+)?", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s*\(.*?\)\s*$", "", answer)
    for stop in [" because ", " since ", " as we ", " which ", " due to "]:
        if stop in answer.lower():
            idx = answer.lower().index(stop)
            answer = answer[:idx].strip().rstrip(",")
    return answer.strip().rstrip(".")


def extract_answer_fixed(cot_output):
    """Fixed extraction: period only terminates at end-of-string or before newline."""
    text = cot_output.strip()

    # Pattern 1: "So the answer is: ..." — use LAST occurrence
    # FIXED: \. only matches when followed by optional whitespace + end-of-string/newline
    matches = list(re.finditer(r"[Ss]o the answer is:?\s*(.+?)(?:\.\s*(?:\n|$)|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))

    # Pattern 2: "the answer is ..."
    matches = list(re.finditer(r"[Tt]he answer is:?\s*(.+?)(?:\.\s*(?:\n|$)|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))

    # Pattern 3: "Final answer: ..."
    match = re.search(r"[Ff]inal answer:?\s*(.+?)(?:\.\s*(?:\n|$)|\n|$)", text)
    if match:
        return _clean_extracted(match.group(1))

    # Pattern 4: Last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return _clean_extracted(lines[-1])

    return text


def extract_answer_old(cot_output):
    """Original buggy extraction for comparison."""
    text = cot_output.strip()
    matches = list(re.finditer(r"[Ss]o the answer is:?\s*(.+?)(?:\.|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))
    matches = list(re.finditer(r"[Tt]he answer is:?\s*(.+?)(?:\.|\n|$)", text))
    if matches:
        return _clean_extracted(matches[-1].group(1))
    match = re.search(r"[Ff]inal answer:?\s*(.+?)(?:\.|\n|$)", text)
    if match:
        return _clean_extracted(match.group(1))
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        return _clean_extracted(lines[-1])
    return text


# ── Evaluation (same as FlashRAG) ───────────────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(pred, gold_list):
    norm_pred = normalize_answer(pred)
    return max(float(norm_pred == normalize_answer(g)) for g in gold_list)


def compute_f1(pred, gold_list):
    best = 0.0
    for gold in gold_list:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_tokens)
        rec = num_same / len(gold_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


def compute_precision(pred, gold_list):
    best = 0.0
    for gold in gold_list:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_tokens)
        best = max(best, prec)
    return best


def compute_recall(pred, gold_list):
    best = 0.0
    for gold in gold_list:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        rec = num_same / len(gold_tokens)
        best = max(best, rec)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    data_path = os.path.join(args.results_dir, "intermediate_data.json")
    print(f"Loading {data_path}")
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")

    # Re-extract answers with both old and new regex
    old_ems, new_ems = [], []
    old_f1s, new_f1s = [], []
    new_precs, new_recs = [], []
    n_changed = 0
    n_improved_em = 0
    n_improved_f1 = 0
    examples = []

    for item in data:
        raw_cot = item["output"].get("raw_cot_output", "")
        gold = item.get("golden_answers", [])
        if not gold:
            continue

        old_pred = extract_answer_old(raw_cot)
        new_pred = extract_answer_fixed(raw_cot)

        old_em = compute_em(old_pred, gold)
        new_em = compute_em(new_pred, gold)
        old_f1 = compute_f1(old_pred, gold)
        new_f1 = compute_f1(new_pred, gold)
        new_prec = compute_precision(new_pred, gold)
        new_rec = compute_recall(new_pred, gold)

        old_ems.append(old_em)
        new_ems.append(new_em)
        old_f1s.append(old_f1)
        new_f1s.append(new_f1)
        new_precs.append(new_prec)
        new_recs.append(new_rec)

        if old_pred != new_pred:
            n_changed += 1
            if new_em > old_em:
                n_improved_em += 1
            if new_f1 > old_f1:
                n_improved_f1 += 1
            if len(examples) < 10:
                examples.append({
                    "gold": gold,
                    "old_pred": old_pred,
                    "new_pred": new_pred,
                    "old_em": old_em,
                    "new_em": new_em,
                    "old_f1": round(old_f1, 3),
                    "new_f1": round(new_f1, 3),
                })

    n = len(old_ems)
    print(f"\n{'='*60}")
    print(f"RE-EVALUATION RESULTS (n={n})")
    print(f"{'='*60}")
    print(f"\n  OLD regex (dot-as-terminator):")
    print(f"    EM:        {sum(old_ems)/n:.4f} ({100*sum(old_ems)/n:.2f}%)")
    print(f"    F1:        {sum(old_f1s)/n:.4f} ({100*sum(old_f1s)/n:.2f}%)")
    print(f"\n  NEW regex (dot-only-at-end):")
    print(f"    EM:        {sum(new_ems)/n:.4f} ({100*sum(new_ems)/n:.2f}%)")
    print(f"    F1:        {sum(new_f1s)/n:.4f} ({100*sum(new_f1s)/n:.2f}%)")
    print(f"    Precision: {sum(new_precs)/n:.4f} ({100*sum(new_precs)/n:.2f}%)")
    print(f"    Recall:    {sum(new_recs)/n:.4f} ({100*sum(new_recs)/n:.2f}%)")
    print(f"\n  DELTA (new - old):")
    print(f"    EM:  {100*(sum(new_ems)-sum(old_ems))/n:+.2f}pp")
    print(f"    F1:  {100*(sum(new_f1s)-sum(old_f1s))/n:+.2f}pp")
    print(f"\n  Changed predictions: {n_changed}/{n} ({100*n_changed/n:.1f}%)")
    print(f"  EM improved: {n_improved_em} | F1 improved: {n_improved_f1}")

    if examples:
        print(f"\n--- Examples where extraction changed ---")
        for ex in examples:
            print(f"  Gold: {ex['gold']}")
            print(f"  Old:  '{ex['old_pred']}' (EM={ex['old_em']}, F1={ex['old_f1']})")
            print(f"  New:  '{ex['new_pred']}' (EM={ex['new_em']}, F1={ex['new_f1']})")
            print()

    # Save new metric scores
    out_path = os.path.join(args.results_dir, "metric_score_fixed_regex.txt")
    with open(out_path, "w") as f:
        f.write(f"em: {sum(new_ems)/n}\n")
        f.write(f"f1: {sum(new_f1s)/n}\n")
        f.write(f"precision: {sum(new_precs)/n}\n")
        f.write(f"recall: {sum(new_recs)/n}\n")
    print(f"\nSaved fixed metrics to {out_path}")


if __name__ == "__main__":
    main()
