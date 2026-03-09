"""Check what _extract_entities_from_question returns for 2Wiki film questions."""
import json, re

def extract_entities(question):
    q = question.strip().rstrip("?").strip()
    m = re.search(r",\s*(.+?)\s+or\s+(.+?)$", q, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    m = re.match(r"(?:between|in between)\s+(.+?)\s+and\s+(.+?),", q, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    m = re.match(r"(?:are|were|is|do|does)\s+(.+?)\s+and\s+(.+?)\s+both\b", q, re.IGNORECASE)
    if m:
        return [m.group(1).strip(), m.group(2).strip()]
    return []

with open("results/m6v13k_pilot100/2wiki/predictions.jsonl") as f:
    preds = [json.loads(l) for l in f if l.strip()]

temporal_patterns = [
    "born first", "born earlier", "born later", "died first", "died earlier", "died later",
    "lived longer", "is older", "is younger",
    "which film has the director born", "which film has the director died",
    "which film has the director who was born", "which film has the director who died",
    "which film whose director",
    "established first",
]

for p in preds:
    q = p["question"].lower()
    if not any(pat in q for pat in temporal_patterns):
        continue
    entities = extract_entities(p["question"])
    print(f"Q: {p['question'][:95]}")
    print(f"  Extracted: {entities}")
    print(f"  Gold: {p['gold_answer']}")
    print()
