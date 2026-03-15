"""Fix read_chunk.py output format to be LLM-friendly.

Removes separator lines (====, ----) that the LLM copies as answers.
"""
from pathlib import Path

path = Path("src/arag/tools/read_chunk.py")
lines = path.read_text().splitlines()

new_lines = []
for line in lines:
    stripped = line.strip()
    # Remove lines that produce separator output
    if "{'=' * 80}" in stripped or "{'-' * 80}" in stripped:
        continue
    # Remove the commented separator lines too
    if stripped == "# separator removed":
        continue
    new_lines.append(line)

path.write_text("\n".join(new_lines) + "\n")

import ast
ast.parse(path.read_text())
print("read_chunk.py: separators removed, syntax OK")

# Show what the output looks like now
lines = path.read_text().splitlines()
for i, line in enumerate(lines):
    if "result_parts" in line:
        print(f"  L{i+1}: {line.strip()}")
