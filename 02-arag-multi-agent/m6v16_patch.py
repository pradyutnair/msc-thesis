#!/usr/bin/env python3
"""M6 v16: expected_answer type flow + simplified prompts."""

from pathlib import Path

BASE = Path("/projects/prjs1800/msc-thesis/02-arag-multi-agent")
M6 = BASE / "src" / "multi_agent" / "m6"

def patch(path, old, new):
    content = path.read_text()
    if old not in content:
        print(f"  WARNING: not found in {path.name}")
        print(f"  Looking for: {repr(old[:100])}")
        return False
    content = content.replace(old, new, 1)
    path.write_text(content)
    print(f"  OK: {path.name}")
    return True

# ═══════════════════════════════════════════════════════════════
# 1. Parse expected_answer from decomposition
# ═══════════════════════════════════════════════════════════════

print("=== 1: Parse expected_answer ===")

patch(
    M6 / "planner_agent.py",
    '        self._question_type = data.get("question_type", "unknown")',
    '        self._question_type = data.get("question_type", "unknown")\n        self._expected_answer = data.get("expected_answer", "")',
)

# ═══════════════════════════════════════════════════════════════
# 2. Store expected_answer on blackboard
# ═══════════════════════════════════════════════════════════════

print("\n=== 2: Blackboard field ===")

patch(
    M6 / "blackboard.py",
    "        # Warm-start context from full-question retrieval\n        self.warm_start_context: str = \"\"",
    "        # Warm-start context from full-question retrieval\n        self.warm_start_context: str = \"\"\n\n        # Expected answer type from decomposer\n        self.expected_answer: str = \"\"",
)

print("\n=== 2b: Store after decomposition ===")

patch(
    M6 / "planner_agent.py",
    '                await blackboard.set_search_plan(sub_questions)',
    '                await blackboard.set_search_plan(sub_questions)\n                blackboard.expected_answer = getattr(self, "_expected_answer", "")',
)

# ═══════════════════════════════════════════════════════════════
# 3. Pass expected_answer to synthesizer template
# ═══════════════════════════════════════════════════════════════

print("\n=== 3: Synthesizer gets expected_answer ===")

patch(
    M6 / "planner_agent.py",
    '''        prompt = self._synthesize_template.format(
            question=question,
            evidence_blocks=evidence_blocks,
            entity_registry=entity_str,
        )''',
    '''        expected_answer = getattr(blackboard, "expected_answer", "") or "an entity"
        prompt = self._synthesize_template.format(
            question=question,
            evidence_blocks=evidence_blocks,
            entity_registry=entity_str,
            expected_answer=expected_answer,
        )''',
)

# ═══════════════════════════════════════════════════════════════
# 4. Copy new prompts
# ═══════════════════════════════════════════════════════════════

print("\n=== 4: Prompts already copied via scp ===")

print("\n=== All v16 patches applied ===")
