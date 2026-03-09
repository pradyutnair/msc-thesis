"""Add missing comparison patterns for 2Wiki questions.

2Wiki has patterns like "died earlier", "died later", "lived longer",
"born later", "is older", "is younger", "director born first/later".
"""

path = "/projects/prjs1800/msc-thesis/02-arag-multi-agent/src/multi_agent/m6/planner_agent.py"
with open(path, "r") as f:
    content = f.read()

# Expand "is_first" patterns to catch more comparison types
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
            "is older", "who is older", "director born first",
        ))'''

assert old_first in content, "Old is_first not found"
content = content.replace(old_first, new_first)

# Add "died later", "born later", "lived longer", "is younger" to is_more (pick largest)
old_more = '''        is_more = any(p in q for p in (
            "has more", "have more", "which is longer", "which is larger",
            "which is bigger", "more acts", "more episodes", "more seasons",
        ))'''

new_more = '''        is_more = any(p in q for p in (
            "has more", "have more", "which is longer", "which is larger",
            "which is bigger", "more acts", "more episodes", "more seasons",
            "died later", "born later", "lived longer", "is younger",
            "director born later", "director died later",
            "director who was born later", "director who died later",
        ))'''

assert old_more in content, "Old is_more not found"
content = content.replace(old_more, new_more)

with open(path, "w") as f:
    f.write(content)

print("Added comparison patterns: died earlier/later, born later, lived longer, older/younger")
