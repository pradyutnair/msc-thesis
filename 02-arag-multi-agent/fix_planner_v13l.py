"""Apply targeted fixes to planner_agent.py for v13l.

Fix 1: "how many" / "how much" detection anywhere in question (not just startswith)
Fix 2: _compare_by_date extracts entity names from original question text
Fix 3: Add "what age" to number type detection
"""

import re

path = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m6/planner_agent.py"
with open(path, "r") as f:
    content = f.read()

# ── Fix 1+3: "how many"/"how much" anywhere + "what age" ──
old = '''    if q.startswith("how many ") or q.startswith("how much "):
        return "number"'''
new = '''    if "how many" in q or "how much" in q or q.startswith("what age "):
        return "number"'''
assert old in content, "Fix 1: old string not found"
content = content.replace(old, new)

# ── Fix 2: _compare_by_date extracts entities from question ──
old_compare = '''    def _compare_by_date(
        self,
        question: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
        pick: str,
    ) -> str:
        """Pick the entity whose sub-answer has the smallest/largest numeric value."""
        pairs: list[tuple[str, int]] = []
        for sq in sub_questions:
            sq_id = sq["id"]
            val = entity_registry.get(f"answer_{sq_id}", "")
            year = self._extract_year(val)
            if year is None:
                continue
            entity = sq.get("text", "")
            for known in sq.get("known_entities", []):
                entity = known
                break
            pairs.append((entity, year))

        if len(pairs) < 2:
            return ""

        if pick == "smallest":
            pairs.sort(key=lambda x: x[1])
        else:
            pairs.sort(key=lambda x: -x[1])

        winner_entity = pairs[0][0]
        q_lower = question.lower()
        for ent, _ in pairs:
            if ent.lower() in q_lower:
                if ent == winner_entity:
                    return ent
        return winner_entity'''

new_compare = '''    def _compare_by_date(
        self,
        question: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
        pick: str,
    ) -> str:
        """Pick the entity whose sub-answer has the smallest/largest numeric value."""
        # Extract entity names from the original question
        q_entities = self._extract_entities_from_question(question)

        pairs: list[tuple[str, int]] = []
        for i, sq in enumerate(sub_questions):
            sq_id = sq["id"]
            val = entity_registry.get(f"answer_{sq_id}", "")
            year = self._extract_year(val)
            if year is None:
                continue
            # Map SQ to entity: prefer entity from question, fall back to known_entities
            entity = q_entities[i] if i < len(q_entities) else ""
            if not entity:
                for known in sq.get("known_entities", []):
                    entity = known
                    break
            if not entity:
                entity = sq.get("text", "")
            pairs.append((entity, year))

        if len(pairs) < 2:
            return ""

        if pick == "smallest":
            pairs.sort(key=lambda x: x[1])
        else:
            pairs.sort(key=lambda x: -x[1])

        return pairs[0][0]

    @staticmethod
    def _extract_entities_from_question(question: str) -> list[str]:
        """Extract entity names from comparison questions.

        Handles: "Who was born first, X or Y?", "Between X and Y, which..."
        """
        q = question.strip().rstrip("?").strip()
        # Pattern: "..., X or Y"
        m = re.search(r",\\s*(.+?)\\s+or\\s+(.+?)$", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "Between X and Y, ..."
        m = re.match(r"(?:between|in between)\\s+(.+?)\\s+and\\s+(.+?),", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "X and Y" for "Are X and Y both Z?"
        m = re.match(r"(?:are|were|is|do|does)\\s+(.+?)\\s+and\\s+(.+?)\\s+both\\b", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        return []'''

assert old_compare in content, "Fix 2: old _compare_by_date not found"
content = content.replace(old_compare, new_compare)

with open(path, "w") as f:
    f.write(content)

# Verify
with open(path, "r") as f:
    t = f.read()
assert "_extract_entities_from_question" in t, "Missing entity extraction method"
assert '"how many" in q' in t, "Missing how many fix"
assert 'q.startswith("what age ")' in t, "Missing what age fix"
print("Applied 3 planner fixes (how many, what age, compare_by_date)")
