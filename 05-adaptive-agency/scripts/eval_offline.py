#!/usr/bin/env python3
"""
Offline evaluation for ARAG predictions using standard HotPotQA metrics.
No GPU or LLM judge required — runs instantly on CPU.

Usage:
    python scripts/eval_offline.py results/sage_v6_p100/hotpotqa/predictions.jsonl
    python scripts/eval_offline.py pred1.jsonl pred2.jsonl pred3.jsonl  # compare
"""

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

REASONING_TAG_BLOCK_RE = re.compile(
    r"<(?:think|thnk)(?:\s[^>]*)?>.*?</(?:think|thnk)>",
    flags=re.IGNORECASE | re.DOTALL,
)
REASONING_TAG_OPEN_RE = re.compile(r"<(?:think|thnk)(?:\s[^>]*)?>", flags=re.IGNORECASE)
REASONING_TAG_CLOSE_RE = re.compile(r"</(?:think|thnk)>", flags=re.IGNORECASE)

# Number word <-> digit mapping
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}
_DIGIT_TO_WORD = {v: k for k, v in _NUM_WORDS.items()}


def strip_reasoning(text):
    if not text or not isinstance(text, str):
        return ""
    text = REASONING_TAG_BLOCK_RE.sub("", text)
    text = REASONING_TAG_OPEN_RE.sub("", text)
    text = REASONING_TAG_CLOSE_RE.sub("", text)
    return text.strip()


def normalize(s):
    """Standard HotPotQA normalization: lower, strip articles/punctuation, collapse whitespace."""
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


_DEMONYM_TO_COUNTRY = {
    "danish": "denmark", "french": "france", "german": "germany",
    "italian": "italy", "spanish": "spain", "british": "united kingdom",
    "american": "united states", "japanese": "japan", "chinese": "china",
    "russian": "russia", "polish": "poland", "dutch": "netherlands",
    "swedish": "sweden", "norwegian": "norway", "finnish": "finland",
    "brazilian": "brazil", "mexican": "mexico", "canadian": "canada",
    "australian": "australia", "indian": "india", "korean": "south korea",
    "irish": "ireland", "scottish": "scotland", "welsh": "wales",
    "english": "england", "swiss": "switzerland", "austrian": "austria",
    "belgian": "belgium", "portuguese": "portugal", "greek": "greece",
    "turkish": "turkey", "egyptian": "egypt", "argentinian": "argentina",
    "colombian": "colombia", "chilean": "chile", "peruvian": "peru",
    "czech": "czech republic", "hungarian": "hungary", "romanian": "romania",
    "neapolitan": "naples",
}
_COUNTRY_TO_DEMONYM = {v: k for k, v in _DEMONYM_TO_COUNTRY.items()}


def _number_normalize(s):
    """Convert number words to digits and vice versa for matching."""
    words = s.split()
    # Try converting number words to digits
    digit_form = []
    for w in words:
        digit_form.append(_NUM_WORDS.get(w, w))
    digit_str = " ".join(digit_form)

    # Try converting digits to number words
    word_form = []
    for w in words:
        word_form.append(_DIGIT_TO_WORD.get(w, w))
    word_str = " ".join(word_form)

    return digit_str, word_str


def _demonym_normalize(s):
    """Convert between demonyms and country names."""
    words = s.split()
    country_form = " ".join(_DEMONYM_TO_COUNTRY.get(w, w) for w in words)
    demonym_form = " ".join(_COUNTRY_TO_DEMONYM.get(w, w) for w in words)
    return country_form, demonym_form


def _word_overlap_match(pred, gold, threshold=0.8):
    """Check if word-level overlap is above threshold (handles reordering, extra words)."""
    pred_words = set(pred.split())
    gold_words = set(gold.split())
    if not gold_words:
        return False
    overlap = pred_words & gold_words
    # Gold words found in pred (recall)
    recall = len(overlap) / len(gold_words)
    return recall >= threshold


def contain_bi_check(pred_norm, gold_norm):
    """Enhanced bidirectional containment check."""
    # Standard substring check
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True

    # Number normalization: try digit and word forms
    pred_digit, pred_word = _number_normalize(pred_norm)
    gold_digit, gold_word = _number_normalize(gold_norm)

    for p in (pred_norm, pred_digit, pred_word):
        for g in (gold_norm, gold_digit, gold_word):
            if g in p or p in g:
                return True

    # Demonym normalization: "danish" = "denmark", etc.
    pred_country, pred_demonym = _demonym_normalize(pred_norm)
    gold_country, gold_demonym = _demonym_normalize(gold_norm)
    for p in (pred_norm, pred_country, pred_demonym):
        for g in (gold_norm, gold_country, gold_demonym):
            if g in p or p in g:
                return True

    # Word-level containment: all words of shorter string appear in longer string
    pred_words = set(pred_norm.split())
    gold_words = set(gold_norm.split())
    if gold_words and gold_words.issubset(pred_words):
        return True
    if pred_words and pred_words.issubset(gold_words):
        return True

    return False


def em_check(pred_norm, gold_norm):
    """Enhanced exact match check with number and demonym normalization."""
    if pred_norm == gold_norm:
        return True

    # Number normalization
    pred_digit, pred_word = _number_normalize(pred_norm)
    gold_digit, gold_word = _number_normalize(gold_norm)

    for p in (pred_norm, pred_digit, pred_word):
        for g in (gold_norm, gold_digit, gold_word):
            if p == g:
                return True

    # Demonym normalization
    pred_country, pred_demonym = _demonym_normalize(pred_norm)
    gold_country, gold_demonym = _demonym_normalize(gold_norm)
    for p in (pred_norm, pred_country, pred_demonym):
        for g in (gold_norm, gold_country, gold_demonym):
            if p == g:
                return True

    return False


def token_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_file(path):
    preds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))

    n = len(preds)
    em_count = 0
    f1_sum = 0.0
    contain_bi_count = 0
    llm_count = 0
    has_llm = False

    for p in preds:
        pred = strip_reasoning(p.get("pred_answer", ""))
        gold = p.get("gold_answer") or p.get("answer", "")

        np, ng = normalize(pred), normalize(gold)

        if em_check(np, ng):
            em_count += 1
        if contain_bi_check(np, ng):
            contain_bi_count += 1
        f1_sum += token_f1(pred, gold)

        if "llm_accuracy" in p:
            has_llm = True
            llm_count += int(p["llm_accuracy"])

    results = {
        "file": str(path),
        "total": n,
        "norm_em": round(em_count / n, 4),
        "token_f1": round(f1_sum / n, 4),
        "contain_bi": round(contain_bi_count / n, 4),
        "correct_em": em_count,
        "correct_contain_bi": contain_bi_count,
    }
    if has_llm:
        results["llm_accuracy"] = round(llm_count / n, 4)
        results["correct_llm"] = llm_count

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_offline.py <predictions.jsonl> [...]")
        sys.exit(1)

    all_results = []
    for fpath in sys.argv[1:]:
        r = evaluate_file(fpath)
        all_results.append(r)

    # Print comparison table
    header = f"{'File':<55} {'N':>4} {'NormEM':>7} {'F1':>7} {'Cont':>7}"
    if any("llm_accuracy" in r for r in all_results):
        header += f" {'LLM':>7}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        name = Path(r["file"]).parts[-3] + "/" + Path(r["file"]).parts[-2]
        line = f"{name:<55} {r['total']:>4} {r['norm_em']:>7.1%} {r['token_f1']:>7.1%} {r['contain_bi']:>7.1%}"
        if "llm_accuracy" in r:
            line += f" {r['llm_accuracy']:>7.1%}"
        print(line)

    # Save per-file summaries
    for r in all_results:
        out = Path(r["file"]).parent / "offline_eval_summary.json"
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
