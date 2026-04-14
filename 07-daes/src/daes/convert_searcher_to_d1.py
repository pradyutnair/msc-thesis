"""
Convert DLLM-Searcher SFT data to d1 format for multi-hop QA training.

DLLM-Searcher format: {"prompt": "<|im_start|>...", "response": "<think>...<tool_call>...<tool_response>...<|box_start|>answer<|box_end|>"}
d1 format: {"question": "...", "thinking_trajectories": ["..."], "attempt": "..."}

We extract:
- question: the user's original question
- evidence: all search results from tool_response tags (becomes context)
- reasoning: the final think block before the answer
- answer: from box_start/box_end tags
"""

import json
import re
import sys


def extract_question(prompt):
    """Extract the user question from DLLM-Searcher prompt."""
    # Question is at the end after "User: "
    match = re.search(r'User:\s*(.+?)(?:<\|im_end\|>|\Z)', prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: last line
    lines = prompt.strip().split('\n')
    return lines[-1].strip()


def extract_tool_responses(response):
    """Extract all search results from tool_response tags."""
    responses = re.findall(r'<tool_response>(.*?)</tool_response>', response, re.DOTALL)
    # Take unique, deduplicate
    seen = set()
    unique = []
    for r in responses:
        r_clean = r.strip()[:2000]  # truncate long responses
        if r_clean not in seen:
            seen.add(r_clean)
            unique.append(r_clean)
    return unique


def extract_reasoning(response):
    """Extract all think blocks."""
    thinks = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
    if thinks:
        # Return the final (most refined) reasoning
        return thinks[-1].strip()
    return ""


def extract_answer(response):
    """Extract answer from box_start/box_end tags."""
    match = re.search(r'<\|box_start\|>(.*?)<\|box_end\|>', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: last line
    return response.strip().split('\n')[-1].strip()


def convert(input_path, output_path, max_examples=None):
    data = json.load(open(input_path))
    if max_examples:
        data = data[:max_examples]

    converted = []
    skipped = 0
    for i, ex in enumerate(data):
        try:
            question = extract_question(ex["prompt"])
            tool_responses = extract_tool_responses(ex["response"])
            reasoning = extract_reasoning(ex["response"])
            answer = extract_answer(ex["response"])

            if not question or not answer:
                skipped += 1
                continue

            # Build context from search results (top 3 most relevant passages)
            context_parts = []
            for j, tr in enumerate(tool_responses[:3]):
                # Extract just the useful text, skip metadata
                # Take first 1000 chars of each response
                context_parts.append(f"Passage {j+1}: {tr[:1000]}")
            context = "\n\n".join(context_parts)

            # Format question with context (for RAG setting)
            full_question = f"Context:\n{context}\n\nQuestion: {question}" if context else f"Question: {question}"

            converted.append({
                "question": full_question,
                "thinking_trajectories": [reasoning if reasoning else f"Let me think about this question: {question}"],
                "attempt": answer,
            })
        except Exception as e:
            skipped += 1
            if i < 5:
                print(f"Error on example {i}: {e}")

    print(f"Converted: {len(converted)}, Skipped: {skipped}")
    print(f"Sample question: {converted[0]['question'][:200]}")
    print(f"Sample reasoning: {converted[0]['thinking_trajectories'][0][:200]}")
    print(f"Sample answer: {converted[0]['attempt'][:100]}")

    # Save as HF-compatible jsonl
    with open(output_path, "w") as f:
        for ex in converted:
            f.write(json.dumps(ex) + "\n")

    # Also save as json for d1 compatibility
    json_path = output_path.replace(".jsonl", ".json")
    json.dump(converted, open(json_path, "w"))
    print(f"Saved to {output_path} and {json_path}")


if __name__ == "__main__":
    convert(
        "/projects/prjs1800/msc-thesis/07-daes/data/dllm_searcher_sft.json",
        "/projects/prjs1800/msc-thesis/07-daes/data/d1_qa_sft.jsonl"
    )
