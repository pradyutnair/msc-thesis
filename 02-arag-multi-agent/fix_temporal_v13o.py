"""Fix temporal comparison handling for 2Wiki.

Two changes:
1. Expand comparison patterns to cover 2Wiki's "born later", "is older", "died later", etc.
2. Use HYBRID entity matching: try content-match first, fall back to index-based.
   This handles both HotpotQA (index works) and 2Wiki (SQ order may differ).
"""

import re

path = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m6/planner_agent.py"
with open(path, "r") as f:
    content = f.read()

# ── Fix 1: Expand comparison patterns ──
old_first = '''        is_first = any(p in q for p in (
            "born first", "formed first", "formed earlier", "founded first",
            "released first", "published first", "created first", "came first",
            "which was founded", "in between",
        ))'''

new_first = '''        is_first = any(p in q for p in (
            "born first", "born earlier", "formed first", "formed earlier",
            "founded first", "founded earlier", "released first", "released earlier",
            "published first", "published earlier", "created first", "created earlier",
            "came first", "which was founded", "in between",
            "died first", "died earlier",
            "is older", "who is older",
            "established first", "established earlier",
        ))'''

assert old_first in content, "Old is_first not found"
content = content.replace(old_first, new_first)

old_more = '''        is_more = any(p in q for p in (
            "has more", "have more", "which is longer", "which is larger",
            "which is bigger", "more acts", "more episodes", "more seasons",
        ))'''

new_more = '''        is_more = any(p in q for p in (
            "has more", "have more", "which is longer", "which is larger",
            "which is bigger", "more acts", "more episodes", "more seasons",
            "born later", "died later", "lived longer", "is younger",
        ))'''

assert old_more in content, "Old is_more not found"
content = content.replace(old_more, new_more)

# ── Fix 2: Hybrid entity matching in _compare_by_date ──
old_compare = '''    def _compare_by_date(
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

        return pairs[0][0]'''

new_compare = '''    def _compare_by_date(
        self,
        question: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
        pick: str,
    ) -> str:
        """Pick the entity whose sub-answer has the smallest/largest numeric value.

        Uses hybrid entity matching:
        1. Try content-match (entity name appears in SQ text or known_entities)
        2. Fall back to index-based mapping (SQ order = question entity order)
        3. Last resort: known_entities or SQ text
        """
        q_entities = self._extract_entities_from_question(question)

        pairs: list[tuple[str, int]] = []
        for i, sq in enumerate(sub_questions):
            sq_id = sq["id"]
            val = entity_registry.get(f"answer_{sq_id}", "")
            year = self._extract_year(val)
            if year is None:
                continue

            entity = ""

            # Strategy 1: content-match question entity to SQ
            if q_entities:
                sq_text_lower = sq.get("text", "").lower()
                known_lower = [k.lower() for k in sq.get("known_entities", [])]
                for qe in q_entities:
                    qe_lower = qe.lower()
                    if (qe_lower in sq_text_lower
                        or any(qe_lower in k for k in known_lower)
                        or any(k in qe_lower for k in known_lower)):
                        entity = qe
                        break

            # Strategy 2: index-based fallback
            if not entity and i < len(q_entities):
                entity = q_entities[i]

            # Strategy 3: known_entities
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

        return pairs[0][0]'''

assert old_compare in content, "Old _compare_by_date not found"
content = content.replace(old_compare, new_compare)

with open(path, "w") as f:
    f.write(content)

# Verify
with open(path, "r") as f:
    t = f.read()
assert "Strategy 1: content-match" in t
assert "Strategy 2: index-based" in t
assert '"born later"' in t
assert '"is older"' in t
print("Applied hybrid entity matching + expanded temporal patterns")
