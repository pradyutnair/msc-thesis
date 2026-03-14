#!/usr/bin/env python3
"""Inject distractor passages into a chunk corpus.

Strategy: inter-question cross-contamination.
For each question Q_i, take gold evidence passages from K other questions
that share at least one entity token with Q_i. These are real, coherent
passages that mention similar entities but answer different questions,
simulating a realistic noisy retrieval scenario.

Usage:
    python scripts/inject_noise.py \
        --chunks data/hotpotqa/chunks.json \
        --questions data/hotpotqa/questions.json \
        --output data/hotpotqa/chunks_noisy.json \
        --noise-ratio 1.0 \
        --limit 100
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def extract_entities(text: str) -> set[str]:
    """Extract likely entity tokens (capitalized words, multi-word phrases)."""
    words = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text)
    tokens = set()
    for w in words:
        tokens.add(w.lower())
        for part in w.split():
            if len(part) > 2:
                tokens.add(part.lower())
    return tokens


def evidence_to_text(evidence: list) -> list[str]:
    """Convert ARAG evidence format [[title, [sentences]], ...] to text passages."""
    passages = []
    for entry in evidence:
        if isinstance(entry, list) and len(entry) >= 2:
            title = entry[0]
            sentences = entry[1] if isinstance(entry[1], list) else [entry[1]]
            text = f"{title}: {' '.join(str(s) for s in sentences)}"
            passages.append(text)
    return passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noise-ratio", type=float, default=1.0,
                        help="Ratio of distractor chunks to add (1.0 = double corpus)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.chunks) as f:
        original_chunks = json.load(f)

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]

    target_distractors = int(len(original_chunks) * args.noise_ratio)
    per_question = max(1, target_distractors // len(questions))

    q_entities = []
    q_evidence_passages = []
    for q in questions:
        entities = extract_entities(q["question"])
        q_entities.append(entities)
        passages = evidence_to_text(q.get("evidence", []))
        q_evidence_passages.append(passages)

    distractors = []
    distractor_id = len(original_chunks)

    for i, q in enumerate(questions):
        my_entities = q_entities[i]
        if not my_entities:
            continue

        candidates = []
        for j, other_q in enumerate(questions):
            if i == j:
                continue
            overlap = my_entities & q_entities[j]
            if overlap:
                for passage in q_evidence_passages[j]:
                    candidates.append((len(overlap), passage))

        if not candidates:
            for j, other_q in enumerate(questions):
                if i == j:
                    continue
                for passage in q_evidence_passages[j]:
                    candidates.append((0, passage))

        candidates.sort(key=lambda x: -x[0])
        selected = candidates[:per_question * 2]
        random.shuffle(selected)
        selected = selected[:per_question]

        for _, passage in selected:
            chunk_str = f"{distractor_id}:{passage}"
            distractors.append(chunk_str)
            distractor_id += 1

    noisy_chunks = original_chunks + distractors
    random.shuffle(noisy_chunks)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(noisy_chunks, f, ensure_ascii=False)

    print(f"Original chunks: {len(original_chunks)}")
    print(f"Distractors added: {len(distractors)}")
    print(f"Total noisy corpus: {len(noisy_chunks)}")
    print(f"Noise ratio: {len(distractors)/len(original_chunks):.2f}")
    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
