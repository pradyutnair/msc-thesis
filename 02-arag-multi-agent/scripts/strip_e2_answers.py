#!/usr/bin/env python3
"""Strip E2 verbose answers down to core answer entity for fair LLM judge comparison.

Extracts the final answer from E2's verbose responses using the same patterns
the LLM uses: looks for bold text, "the answer is X", or last short sentence.

Usage:
    python strip_e2_answers.py INPUT.jsonl OUTPUT.jsonl
"""

import json
import re
import sys

BLOCK_RE = re.compile(r"<(?:think|thnk)(?:\s[^>]*)?>.*?</(?:think|thnk)>", flags=re.IGNORECASE | re.DOTALL)
OPEN_RE = re.compile(r"<(?:think|thnk)(?:\s[^>]*)?>", flags=re.IGNORECASE)
CLOSE_RE = re.compile(r"</(?:think|thnk)>", flags=re.IGNORECASE)


def strip_reasoning(text):
    text = BLOCK_RE.sub("", text)
    text = OPEN_RE.sub("", text)
    text = CLOSE_RE.sub("", text)
    return text.strip()


def extract_core_answer(verbose_answer):
    """Extract the core answer entity from a verbose E2 response."""
    text = strip_reasoning(verbose_answer)
    if not text:
        return ""

    # Strategy 1: Look for bold markers **answer** — LLMs often bold the key entity
    bold_matches = re.findall(r"\*\*([^*]+?)\*\*", text)
    if bold_matches:
        # Filter out non-answer bolds (common headers/labels)
        skip_patterns = [
            "key evidence", "evidence", "chunk", "note", "important",
            "answer", "summary", "conclusion", "thus", "therefore",
            "source", "reference",
        ]
        candidates = [
            b.strip() for b in bold_matches
            if b.strip().lower() not in skip_patterns
            and len(b.strip()) < 200
            and len(b.strip()) > 0
        ]
        if candidates:
            # Prefer the last substantive bold (usually the final answer)
            return candidates[-1]

    # Strategy 2: "the answer is X" / "thus, X" / "therefore, X"
    patterns = [
        r"(?:the answer is|answer is|the answer to .+ is)\s+[\"']?(.+?)[\"']?[.\n]",
        r"(?:thus|therefore|hence|so),?\s+(?:the answer is\s+)?[\"']?(.+?)[\"']?[.\n]",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            answer = m.group(1).strip().rstrip(".")
            if len(answer) < 200:
                return answer

    # Strategy 3: Last sentence (often the concluding answer)
    sentences = re.split(r"[.!?]\s+", text)
    sentences = [s.strip().rstrip(".!?") for s in sentences if s.strip()]
    if sentences:
        last = sentences[-1]
        # Strip common prefixes
        last = re.sub(r"^(?:Thus|Therefore|Hence|So|In conclusion|In summary),?\s*", "", last, flags=re.IGNORECASE)
        if len(last) < 200:
            return last

    # Fallback: return first 200 chars
    return text[:200]


def main():
    if len(sys.argv) != 3:
        print("Usage: python strip_e2_answers.py INPUT.jsonl OUTPUT.jsonl")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    count = 0
    total_orig_len = 0
    total_stripped_len = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            d = json.loads(line)
            orig = d.get("pred_answer", "")
            stripped = extract_core_answer(orig)
            total_orig_len += len(strip_reasoning(orig))
            total_stripped_len += len(stripped)

            d["pred_answer"] = stripped
            d["pred_answer_original"] = orig
            # Clear old eval scores so we know to re-evaluate
            d.pop("llm_accuracy", None)
            d.pop("contain_accuracy", None)
            d.pop("status", None)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            count += 1

    avg_orig = total_orig_len / count if count else 0
    avg_stripped = total_stripped_len / count if count else 0
    print(f"Processed {count} predictions")
    print(f"Avg length: {avg_orig:.0f} -> {avg_stripped:.0f} chars ({avg_stripped/avg_orig*100:.1f}%)")


if __name__ == "__main__":
    main()
