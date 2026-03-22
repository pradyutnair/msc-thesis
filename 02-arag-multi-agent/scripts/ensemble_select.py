#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MONTH_PATTERN = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
REFUSALS = {'', 'unknown', 'none', 'n/a', 'error'}


def normalize_space(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '')).strip()


def clean_candidate(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'^(?:FINAL ANSWER|ANSWER)\s*:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:The answer is|Answer is)\s+', '', text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`*")
    return normalize_space(text)


def is_empty(text: str) -> bool:
    text = clean_candidate(text).lower()
    return any(text == x or text.startswith(f'{x}:') for x in REFUSALS)


def extract_date_candidates(text: str) -> list[str]:
    text = clean_candidate(text)
    out = []
    patterns = [
        rf'\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{4}}\b',
        rf'\b{MONTH_PATTERN}\s+\d{{1,2}},\s+\d{{4}}\b',
        rf'\bmid-{MONTH_PATTERN.lower()}\b',
        rf'\b{MONTH_PATTERN}\s+\d{{4}}\b',
        r'\b\d{4}\b',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            cand = clean_candidate(m.group(0))
            if cand and cand not in out:
                out.append(cand)
    return out


def extract_outlet_candidates(text: str) -> list[str]:
    text = clean_candidate(text)
    out = []
    patterns = [
        r'expecting the ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,3} (?:River Delta|Delta|Bay|Estuary|Mouth|Channel|Strait))',
        r'\b([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,3} (?:River Delta|Delta|Bay|Estuary|Mouth|Channel|Strait))\b',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            cand = clean_candidate(m.group(1))
            if cand.lower() in {'gulf of mexico', 'mississippi river'}:
                continue
            if cand and cand not in out:
                out.append(cand)
    return out


def extract_candidates(record: dict) -> list[str]:
    cands = []

    def add(text: str):
        text = clean_candidate(text)
        if text and text not in cands and not is_empty(text):
            cands.append(text)

    add(record.get('pred_answer', ''))
    for text in record.get('entity_registry', {}).values():
        add(text)
        for cand in extract_date_candidates(text):
            add(cand)
        for cand in extract_outlet_candidates(text):
            add(cand)
    for sq in record.get('sub_questions', []):
        add(sq.get('answer', ''))
        for cand in extract_date_candidates(sq.get('answer', '')):
            add(cand)
    return cands


def date_specificity(text: str) -> int:
    text = clean_candidate(text)
    if re.search(rf'\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{4}}\b', text, flags=re.IGNORECASE):
        return 4
    if re.search(rf'\b{MONTH_PATTERN}\s+\d{{1,2}},\s+\d{{4}}\b', text, flags=re.IGNORECASE):
        return 4
    if re.search(rf'\bmid-{MONTH_PATTERN.lower()}\b', text.lower()):
        return 3
    if re.search(rf'\b{MONTH_PATTERN}\s+\d{{4}}\b', text, flags=re.IGNORECASE):
        return 3
    if re.fullmatch(r'\d{4}', text):
        return 1
    if re.fullmatch(rf'{MONTH_PATTERN}', text, flags=re.IGNORECASE):
        return 1
    return 0


def score_candidate(question: str, candidate: str, support: int) -> tuple:
    q = (question or '').lower()
    c = candidate.lower()
    score = 0
    if 'empty into' in q or 'flows into' in q:
        if 'delta' in c:
            score += 20
        elif any(x in c for x in ['estuary', 'mouth', 'channel', 'strait']):
            score += 16
        elif 'bay' in c:
            score += 12
        elif any(x in c for x in ['gulf', 'sea', 'river', 'lake']):
            score += 4
    if q.startswith('when ') or 'what year' in q or 'birth date' in q or 'what month' in q:
        score += 8 * date_specificity(candidate)
    if 'county' in q and 'county' in c:
        score += 8
    if len(candidate.split()) <= 5:
        score += 2
    if len(candidate) > 120:
        score -= 12
    if ',' in candidate and len(candidate.split()) > 8:
        score -= 6
    if candidate.startswith('The ') or candidate.startswith('Based on'):
        score -= 8
    score += 3 * support
    return (score, support, -len(candidate))


def main():
    if len(sys.argv) < 4:
        print('Usage: ensemble_select.py <output_predictions> <input1> <input2> [...]')
        sys.exit(1)
    out_path = Path(sys.argv[1])
    input_paths = [Path(p) for p in sys.argv[2:]]
    by_qid = defaultdict(list)
    for path in input_paths:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                by_qid[row['qid']].append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for qid, records in by_qid.items():
            question = records[0]['question']
            gold = records[0].get('gold_answer', '')
            all_candidates = []
            for record in records:
                all_candidates.extend(extract_candidates(record))
            freq = Counter(all_candidates)
            best = ''
            best_score = (-10**9, -10**9, -10**9)
            for cand, support in freq.items():
                sc = score_candidate(question, cand, support)
                if sc > best_score:
                    best_score = sc
                    best = cand
            merged = dict(records[0])
            merged['gold_answer'] = gold
            merged['pred_answer'] = best
            merged['selector_support'] = freq.get(best, 0)
            merged['selector_candidates'] = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:12]
            f.write(json.dumps(merged, ensure_ascii=False) + '\n')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
