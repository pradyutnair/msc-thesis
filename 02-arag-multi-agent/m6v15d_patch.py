#!/usr/bin/env python3
"""M6 v15d: Remove minimum effort enforcement, keep improved synthesizer."""

from pathlib import Path

BASE = Path("/projects/prjs1800/msc-thesis/02-arag-multi-agent")
M6 = BASE / "src" / "multi_agent" / "m6"

def patch(path, old, new):
    content = path.read_text()
    if old not in content:
        print(f"  WARNING: not found in {path.name}")
        return False
    content = content.replace(old, new, 1)
    path.write_text(content)
    print(f"  OK: {path.name}")
    return True

print("=== Remove minimum effort enforcement, simple break ===")
patch(
    M6 / "worker_agent.py",
    '''            # When LLM stops calling tools, check if it's done enough work
            if not tool_calls:
                read_count = sum(1 for a in memory.actions if a.tool_name == "read_chunk")
                search_count = sum(
                    1 for a in memory.actions
                    if a.tool_name in ("keyword_search", "semantic_search")
                )

                if read_count < 2 and step < self.max_steps - 1:
                    messages.append({
                        "role": "user",
                        "content": (
                            "You have only read {} chunks. Read at least 2 relevant chunks "
                            "to verify your answer before concluding."
                        ).format(read_count),
                    })
                    continue
                if search_count < 2 and step < self.max_steps - 1:
                    messages.append({
                        "role": "user",
                        "content": "You need to search more. Execute your pre-planned queries.",
                    })
                    continue
                break''',
    '''            # When LLM stops calling tools, accept its answer
            if not tool_calls:
                break''',
)

print("\n=== Done ===")
