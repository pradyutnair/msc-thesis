#!/usr/bin/env python3
"""M6 v15: Make workers self-verify by enforcing minimum search effort."""

from pathlib import Path

BASE = Path("/projects/prjs1800/msc-thesis/02-arag-multi-agent")
M6 = BASE / "src" / "multi_agent" / "m6"

def patch(path, old, new):
    content = path.read_text()
    if old not in content:
        print(f"  WARNING: pattern not found in {path.name}")
        print(f"  Looking for: {repr(old[:100])}...")
        return False
    content = content.replace(old, new, 1)
    path.write_text(content)
    print(f"  OK: {path.name}")
    return True

# ═══════════════════════════════════════════════════════════════
# Fix 1: Enforce minimum tool calls in _solve() before accepting answer
# ═══════════════════════════════════════════════════════════════

print("=== Fix 1: Minimum search enforcement in _solve() ===")

# Replace the "if not tool_calls: break" with minimum effort check
patch(
    M6 / "worker_agent.py",
    '''            # When LLM stops calling tools, it has proposed an answer — accept it
            if not tool_calls:
                break''',
    '''            # When LLM stops calling tools, check if it's done enough searching
            if not tool_calls:
                # Count actual tool calls made so far
                tool_call_count = sum(
                    1 for m in messages if m.get("role") == "tool"
                )
                min_calls = min(len(search_queries or []), 3)  # At least execute pre-planned queries

                if step <= 2 and tool_call_count < min_calls:
                    # Worker is quitting too early — push it to search more
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must execute ALL pre-planned search queries before answering. "
                            "You have only made {tc} tool calls. Continue searching."
                        ).format(tc=tool_call_count),
                    })
                    continue
                break''',
)

# ═══════════════════════════════════════════════════════════════
# Fix 2: Better worker prompt with self-verification emphasis
# ═══════════════════════════════════════════════════════════════

print("\n=== Fix 2: Updated worker prompt ===")

worker_prompt = """You are a research agent answering a specific sub-question as part of a larger multi-hop question.

## Original Question (for context — your sub-question answer feeds into answering THIS)
{original_question}

## Your Task
{sub_question}

## Pre-Planned Search Queries (execute these FIRST)
{search_queries}

## Warm-Start Context (from initial question retrieval)
{warm_start_context}

## What Other Agents Have Found
{blackboard_context}

## Available Tools
- **keyword_search**: Find chunks by exact keyword matching
- **semantic_search**: Find chunks by semantic similarity
- **read_chunk**: Read the full content of specific chunks

## Instructions
1. Execute ALL pre-planned search queries using keyword_search. Do not skip any.
2. Read the most promising chunks from each search result using read_chunk.
3. Check if the chunk actually mentions the specific entity from your sub-question. If it discusses a different entity with a similar name, search again with more specific terms.
4. If pre-planned queries didn't find relevant results, try different keywords or semantic_search.
5. Check the warm-start context and other agents' findings — the answer may already be there.
6. SELF-VERIFY: Once you have a candidate answer, do one more keyword_search combining your answer with the question entity to confirm the relationship is correct. This is mandatory.
7. Only after verification, output your final answer.

## Answer Format
When you have verified your answer with evidence, respond with ONLY the answer — a single entity name, date, number, or place.
Just the entity. Nothing else. No sentences, no explanations, no prefixes."""

(M6 / "prompts" / "worker_plan.txt").write_text(worker_prompt)
print("  OK: worker_plan.txt")

# ═══════════════════════════════════════════════════════════════
# Fix 3: Better <think> tag stripping (handle unclosed tags)
# ═══════════════════════════════════════════════════════════════

print("\n=== Fix 3: Fix <think> tag stripping ===")

patch(
    M6 / "worker_agent.py",
    '''    def _clean_answer(self, answer: str) -> str:
        answer = re.sub(r"<think>.*?</think>\\s*", "", answer, flags=re.DOTALL)''',
    '''    def _clean_answer(self, answer: str) -> str:
        answer = re.sub(r"<think>.*?</think>\\s*", "", answer, flags=re.DOTALL)
        answer = re.sub(r"<think>.*", "", answer, flags=re.DOTALL)  # unclosed tags''',
)

print("\n=== All v15 patches applied ===")
