"""Add missing entity extraction patterns for 2Wiki questions.

Missing patterns:
1. "Was X or Y born first?" — no comma
2. "born first out of X and Y" — uses "out of"
"""

path = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m6/planner_agent.py"
with open(path, "r") as f:
    content = f.read()

old = '''    @staticmethod
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

new = '''    @staticmethod
    def _extract_entities_from_question(question: str) -> list[str]:
        """Extract entity names from comparison questions.

        Handles: "Who was born first, X or Y?", "Was X or Y born first?",
                 "Between X and Y, ...", "born first out of X and Y"
        """
        q = question.strip().rstrip("?").strip()
        # Pattern: "..., X or Y"
        m = re.search(r",\\s*(.+?)\\s+or\\s+(.+?)$", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "Was/Is X or Y ..."  (no comma)
        m = re.match(r"(?:was|is|were|are|did|has)\\s+(.+?)\\s+or\\s+(.+?)\\s+(?:born|died|formed|founded|established|released|created|published)", q, re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # Pattern: "... out of X and Y"
        m = re.search(r"out of\\s+(.+?)\\s+and\\s+(.+?)$", q, re.IGNORECASE)
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

assert old in content, "Old _extract_entities_from_question not found"
content = content.replace(old, new)

with open(path, "w") as f:
    f.write(content)

print("Added 'Was X or Y ...' and 'out of X and Y' entity extraction patterns")
