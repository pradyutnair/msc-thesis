"""Fix blackboard.py to reject garbage entities and improve cross-pollination."""

import re as re_module
from pathlib import Path


def main():
    path = Path("src/multi_agent/m6/blackboard.py")
    text = path.read_text()

    # Add 'import re' if not present
    if "import re\n" not in text:
        text = text.replace("import asyncio\n", "import asyncio\nimport re\n", 1)

    # Fix: sanitize entity values before posting to registry
    old = '                entity_value = sq.answer or "unknown"'
    new = '''                entity_value = sq.answer or "unknown"
                # Sanitize: reject garbage answers from entity registry
                if entity_value and (
                    entity_value.startswith(("keyword_search(", "semantic_search(", "read_chunk(", "search_and_read("))
                    or re.match(r"^[=\\-_]{5,}$", entity_value)
                    or entity_value.startswith("[Chunk ")
                    or entity_value.startswith("Chunk ID:")
                    or entity_value.startswith("Title:")
                    or len(entity_value) > 200
                ):
                    entity_value = "unknown"'''

    if old in text:
        text = text.replace(old, new, 1)
        print("blackboard.py: added entity sanitization")
    else:
        print("WARNING: could not find entity_value assignment")

    # Also fix _build_cross_agent_context to skip garbage answers
    old_ctx = '            answer_str = sq.answer or "unknown"'
    new_ctx = '''            answer_str = sq.answer or "unknown"
            # Skip garbage answers in cross-agent context
            if answer_str.startswith(("keyword_search(", "semantic_search(")) or re.match(r"^[=\\-_]{5,}$", answer_str):
                answer_str = "(answer unclear)"'''

    if old_ctx in text:
        text = text.replace(old_ctx, new_ctx, 1)
        print("blackboard.py: added garbage filter in cross-agent context")
    else:
        print("WARNING: could not find answer_str in cross-agent context")

    path.write_text(text)

    import ast
    ast.parse(text)
    print("blackboard.py: syntax OK")


if __name__ == "__main__":
    main()
