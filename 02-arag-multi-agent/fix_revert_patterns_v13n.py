"""Revert comparison patterns to v13k originals while keeping content-based entity matching.

The expanded patterns ("is older", "died later", "lived longer", "born later")
override correct LLM synthesis answers with broken programmatic comparisons.
Keep only the original v13k patterns.
"""

path = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m6/planner_agent.py"
with open(path, "r") as f:
    content = f.read()

# Revert is_first to v13k patterns
old_first = '''        is_first = any(p in q for p in (
            "born first", "born earlier", "formed first", "formed earlier",
            "founded first", "founded earlier", "released first", "released earlier",
            "published first", "published earlier", "created first", "created earlier",
            "came first", "which was founded", "in between",
            "died first", "died earlier",
            "is older", "who is older", "director born first",
        ))'''

new_first = '''        is_first = any(p in q for p in (
            "born first", "formed first", "formed earlier", "founded first",
            "released first", "published first", "created first", "came first",
            "which was founded", "in between",
        ))'''

assert old_first in content, "Old is_first not found"
content = content.replace(old_first, new_first)

# Revert is_more to v13k patterns
old_more = '''        is_more = any(p in q for p in (
            "has more", "have more", "which is longer", "which is larger",
            "which is bigger", "more acts", "more episodes", "more seasons",
            "died later", "born later", "lived longer", "is younger",
            "director born later", "director died later",
            "director who was born later", "director who died later",
        ))'''

new_more = '''        is_more = any(p in q for p in (
            "has more", "have more", "which is longer", "which is larger",
            "which is bigger", "more acts", "more episodes", "more seasons",
        ))'''

assert old_more in content, "Old is_more not found"
content = content.replace(old_more, new_more)

with open(path, "w") as f:
    f.write(content)

print("Reverted comparison patterns to v13k originals")
