"""Fix _compare_by_date: match entities to SQs by content, not index.

The v13l bug: q_entities[i] assumes SQ order matches question order.
But decomposer may create SQ 0 for entity B and SQ 1 for entity A.

Fix: Check which question-entity appears in each SQ's text or known_entities.
"""

path = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m6/planner_agent.py"
with open(path, "r") as f:
    content = f.read()

old = '''    def _compare_by_date(
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

new = '''    def _compare_by_date(
        self,
        question: str,
        sub_questions: list[dict[str, Any]],
        entity_registry: dict[str, str],
        pick: str,
    ) -> str:
        """Pick the entity whose sub-answer has the smallest/largest numeric value."""
        q_entities = self._extract_entities_from_question(question)

        pairs: list[tuple[str, int]] = []
        for sq in sub_questions:
            sq_id = sq["id"]
            val = entity_registry.get(f"answer_{sq_id}", "")
            year = self._extract_year(val)
            if year is None:
                continue

            # Match question entity to SQ by checking SQ text/known_entities
            entity = ""
            sq_text_lower = sq.get("text", "").lower()
            known_lower = [k.lower() for k in sq.get("known_entities", [])]

            for qe in q_entities:
                qe_lower = qe.lower()
                if (qe_lower in sq_text_lower
                    or any(qe_lower in k for k in known_lower)
                    or any(k in qe_lower for k in known_lower)):
                    entity = qe
                    break

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

assert old in content, "Old _compare_by_date not found"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

print("Fixed _compare_by_date: match entities by SQ content, not index")
