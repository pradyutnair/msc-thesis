"""Answer extraction and minimal normalization for M6 agents.

Only performs genuine LLM output cleaning — no benchmark-specific
normalization, no hardcoded pattern lists, no answer-type heuristics.
"""

from __future__ import annotations

import re


def normalize_answer(answer: str, question: str) -> str:
    """Minimal normalization: strip LLM artifacts and whitespace."""
    text = (answer or "").strip()
    text = re.sub(r"^\s*(final answer\s*:|answer\s*:)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(the answer is|answer is)\s+", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")
    text = re.sub(r"\s*[\.,;:!?]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_refusal(text: str) -> bool:
    """Detect clear non-answers (empty or placeholder strings only)."""
    lowered = (text or "").strip().lower()
    return lowered in ("", "none", "n/a", "unknown", "null", "error")


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
