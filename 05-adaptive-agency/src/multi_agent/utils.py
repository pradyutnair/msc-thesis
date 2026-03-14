"""Shared utilities for adaptive agency agents."""

from __future__ import annotations

import json
import re
from typing import Any


def strip_llm_wrappers(text: str) -> str:
    """Remove markdown code fences and <think> tags from LLM output."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_json_robust(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM output with tolerance for common formatting issues."""
    text = strip_llm_wrappers(raw or "")
    if not text:
        raise ValueError("empty JSON payload")

    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        cleaned = re.sub(r"[\x00-\x1f\x7f]", "", candidate).strip()
        if not cleaned:
            continue
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = cleaned.replace("\\'", "'")
            repaired = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", repaired)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                continue
    raise ValueError("unable to parse JSON")


def clean_answer(answer: str) -> str:
    """Strip thinking tags, formatting, LLM prefixes, and trailing punctuation."""
    answer = re.sub(r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
    answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)
    answer = answer.split("\n")[0].strip()
    answer = re.sub(r"^(?:Answer|ANSWER|Final Answer|FINAL ANSWER)\s*[:：]\s*", "", answer)
    answer = answer.strip().strip("\"'`*")
    answer = re.sub(r"\s*[\.,;:!?]+$", "", answer)
    return answer


def resolve_placeholders(text: str, entity_registry: dict[str, str]) -> str:
    """Replace [answer_N] placeholders with resolved values from the entity registry."""
    def replacer(match: re.Match) -> str:
        return entity_registry.get(match.group(1), match.group(0))
    return re.sub(r"\[(answer_\d+)\]", replacer, text)


def dedupe_keep_order(values: list[str]) -> list[str]:
    """Deduplicate a list of strings preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out
