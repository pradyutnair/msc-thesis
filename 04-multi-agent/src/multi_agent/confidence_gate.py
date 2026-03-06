"""Confidence gate for OSPREY Phase 1 → fast exit vs. Phase 2 decision.

v2: Length-first scoring — only SHORT factual answers earn high confidence.
Verbose/explanatory answers are penalized even without bad patterns.
"""

from __future__ import annotations

import re

_BAD_PATTERNS = re.compile(
    r"(i don'?t|cannot find|unable to|not found|couldn'?t find|"
    r"i need (more|additional)|insufficient|no information|not enough|"
    r"cannot (determine|confirm|answer)|no (relevant|specific|direct)|"
    r"not (available|provided|mentioned)|no document|no chunk|"
    r"i (would need|need to|must)|further (search|research)|"
    r"outside (my|the) (knowledge|context)|"
    r"based on (my|the) (knowledge|training)|as of my)",
    re.IGNORECASE,
)

# Explanatory openers: answer starts explaining instead of giving the answer
_EXPLANATORY_START = re.compile(
    r"^(the |a |an |i |it |this |that |these |there |based on |"
    r"according to |from (my|the) |in the |yes,|no,|note that|"
    r"the (film|movie|book|show|person|city|country|place|question))",
    re.IGNORECASE,
)

_VAGUE_PATTERNS = re.compile(
    r"(the answer (is|depends|may|could|might)|it (could|may|depends)|"
    r"this (could|may|depends)|various|it depends|"
    r"the (specific|exact|precise) answer)",
    re.IGNORECASE,
)


class ConfidenceGate:
    """Score an answer for confidence and decide whether to fast-exit OSPREY.

    v2 design: LENGTH-FIRST scoring.

    Short factual answers (≤30 chars) earn high base confidence.
    Longer explanatory answers earn low base confidence even if they
    don't contain explicit uncertainty phrases.

    Score breakdown:
    - ≤30 chars: base = 0.80
    - 31-60 chars: base = 0.60
    - 61-100 chars: base = 0.40
    - 101+ chars: base = 0.10
    - Explanatory opener ("The film was...", "I think..."): -0.20
    - Bad "cannot find" pattern: override → 0.10
    - Vague uncertainty: -0.15
    - Hedging language: -0.08 per occurrence

    Default threshold: 0.65 → only short, crisp factual answers fast-exit.
    """

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def score(self, answer: str) -> float:
        """Compute confidence score for *answer* in [0.0, 1.0]."""
        if not answer or not answer.strip():
            return 0.0

        text = answer.strip()

        if len(text) < 2:
            return 0.0

        # Definitive failure patterns → very low confidence regardless of length
        if _BAD_PATTERNS.search(text):
            return 0.10

        # Length-first base scoring
        n = len(text)
        if n <= 30:
            score = 0.80    # "Peter Chelsom", "1935", "yes", "British Royal Family"
        elif n <= 60:
            score = 0.60    # somewhat short
        elif n <= 100:
            score = 0.40    # getting verbose
        else:
            score = 0.10    # long explanatory answer — very low confidence

        # Penalize explanatory openers ("The film was directed by..." etc.)
        if _EXPLANATORY_START.search(text):
            score -= 0.20

        # Penalize vague uncertainty
        if _VAGUE_PATTERNS.search(text):
            score -= 0.15

        # Penalize hedging language
        hedge_count = len(re.findall(
            r'\b(might|may|could|possibly|perhaps|probably|likely|seemingly)\b',
            text, re.IGNORECASE,
        ))
        score -= hedge_count * 0.08

        return max(0.0, min(1.0, score))

    def is_confident(self, answer: str) -> bool:
        return self.score(answer) >= self.threshold
