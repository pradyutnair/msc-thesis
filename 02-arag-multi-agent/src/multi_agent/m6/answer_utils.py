"""Shared answer normalization utilities for M6 agents.

Used by both PlannerAgent and SynthesizerAgent to ensure consistent
answer post-processing across the pipeline.
"""

from __future__ import annotations

import re


def infer_answer_type(question: str) -> str:
    """Infer the expected answer type from the question form."""
    q = (question or "").strip().lower()
    if re.match(r"^(is|are|was|were|do|does|did|has|have|had|can|could|would|should)\b", q):
        return "yes_no"
    if q.startswith("when ") or "what year" in q or "what date" in q:
        return "date"
    if q.startswith("where "):
        return "location"
    if q.startswith("who "):
        return "person"
    if "how many" in q or "how much" in q or q.startswith("what age "):
        return "number"
    return "entity"


_DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}


def normalize_answer(answer: str, question: str) -> str:
    """Full normalization pipeline matching gold label granularity."""
    text = (answer or "").strip()

    # Strip LLM artifacts
    text = re.sub(r"^\s*(final answer\s*:|answer\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the answer is|answer is)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")

    # Yes/no detection
    expected_type = infer_answer_type(question)
    if expected_type == "yes_no":
        lowered = text.lower()
        if lowered.startswith("yes"):
            return "yes"
        if lowered.startswith("no"):
            return "no"

    # Convert digit-only answers to words for "how many" questions
    if expected_type == "number" and text in _DIGIT_TO_WORD:
        text = _DIGIT_TO_WORD[text]

    # Strip sentence wrappers (conservative — only known patterns)
    for pattern in [
        r"^(.*?)\s+was\s+born\s+first\.?$",
        r"^(.*?)\s+was\s+(?:produced|released|published|created|formed|founded)\s+first\.?$",
        r"^(.*?)\s+(?:died|passed away)\s+(?:first|earlier|before)\.?$",
        r"^(.*?)\s+is\s+(?:the\s+)?(?:older|younger|taller|shorter|bigger|smaller)\.?$",
        r"^(.*?)\s+is\s+the\s+answer\.?$",
        r"^The\s+answer\s+is\s+(.+?)\.?$",
    ]:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
            break

    # Strip parenthetical annotations
    text = re.sub(r"\s*\([^)]*\)\s*", " ", text).strip()
    text = re.sub(r"\s*[\.,;:!?]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Strip trailing noise words from comparison answers
    text = re.sub(
        r"\s+(?:theme|genre|style|type|form|category)$",
        "", text, flags=re.IGNORECASE,
    )

    # Truncate verbose answers
    if len(text) > 60:
        m = re.match(r"^(.{3,50}?)[.,;]", text)
        if m:
            text = m.group(1).strip()

    return text


def is_refusal(text: str) -> bool:
    """Detect refusal / non-answer responses."""
    lowered = (text or "").strip().lower()
    if lowered in ("none", "n/a", "unknown", "null", ""):
        return True
    patterns = [
        "cannot be determined", "insufficient information", "not mentioned",
        "no evidence", "unable to determine", "not enough information",
        "unknown", "cannot determine", "no information",
        "not found in the provided", "the question is invalid",
        "the question is asking", "not applicable",
        "no answer", "i don't know", "i cannot",
    ]
    return any(p in lowered for p in patterns)


def extract_answer(raw: str) -> str:
    """Extract FINAL ANSWER from LLM response, stripping think tags."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
    raw = raw.replace("**", "")
    match = re.search(r"FINAL\s*ANSWER\s*:\s*(.+)", raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    return lines[-1] if lines else raw.strip()
