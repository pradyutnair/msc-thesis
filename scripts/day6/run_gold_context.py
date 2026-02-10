"""Run Gold Context (Perfect Retrieval) — Upper Bound Baseline.

Day 6: Generate answers using ground-truth supporting paragraphs as context.
This establishes the upper bound: how well the LLM can answer with perfect
retrieval, quantifying the room for improvement.

Gold extraction per dataset:
- HotpotQA / 2Wiki: metadata.context has {title, sentences}, metadata.supporting_facts
  has {title, sent_id}. Filter context paragraphs whose titles appear in supporting_facts.
- MuSiQue: metadata.question_decomposition has steps with support_paragraph {title, paragraph_text}.

Usage:
    python -u scripts/day6/run_gold_context.py --config configs/day6/gold_context_qwen25_hotpotqa.yaml
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter
from pathlib import Path


# ── Evaluation functions (same as FlashRAG) ─────────────────────────────────

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(pred, gold_list):
    norm_pred = normalize_answer(pred)
    return max(float(norm_pred == normalize_answer(g)) for g in gold_list)


def compute_f1(pred, gold_list):
    best = 0.0
    for gold in gold_list:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        prec = num_same / len(pred_tokens)
        rec = num_same / len(gold_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


# ── Same prompt as Day 1/2 (standard RAG) ──────────────────────────────────

STANDARD_SYSTEM_PROMPT = (
    "Answer the question based on the given document. "
    "Only give me the answer and do not output any other words. "
    "The following are given documents.\n\n{reference}"
)

STANDARD_USER_PROMPT = "Question: {question}"


def format_reference(docs):
    """Format gold docs the same way FlashRAG PromptTemplate does."""
    # FlashRAG default: "[{idx}] {content}\n"
    parts = []
    for idx, doc in enumerate(docs, 1):
        parts.append(f"[{idx}] {doc['contents']}")
    return "\n".join(parts)


def build_prompt(question, gold_docs, tokenizer):
    """Build chat-formatted prompt with gold context."""
    reference = format_reference(gold_docs)
    system = STANDARD_SYSTEM_PROMPT.format(reference=reference)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": STANDARD_USER_PROMPT.format(question=question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Gold context extraction ─────────────────────────────────────────────────

def extract_gold_docs_hotpotqa(metadata):
    """Extract gold supporting paragraphs from HotpotQA metadata.

    metadata.context: {title: [t1, t2, ...], sentences: [[s1, s2, ...], [s1, s2, ...], ...]}
    metadata.supporting_facts: {title: [t1, t2, ...], sent_id: [id1, id2, ...]}
    """
    context = metadata.get("context", {})
    supporting_facts = metadata.get("supporting_facts", {})

    if not context or not supporting_facts:
        return []

    ctx_titles = context.get("title", [])
    ctx_sentences = context.get("sentences", [])
    sf_titles = set(supporting_facts.get("title", []))

    gold_docs = []
    seen_titles = set()
    for i, title in enumerate(ctx_titles):
        if title in sf_titles and title not in seen_titles:
            seen_titles.add(title)
            if i < len(ctx_sentences):
                text = " ".join(ctx_sentences[i])
                # Match FlashRAG format: "Title"\n content
                contents = f'"{title}"\n{text}'
                gold_docs.append({"contents": contents})

    return gold_docs


def extract_gold_docs_2wiki(metadata):
    """Extract gold supporting paragraphs from 2WikiMultihopQA metadata.

    metadata.context: {title: [t1, t2, ...], content: [[s1, s2, ...], [s1, s2, ...], ...]}
    metadata.supporting_facts: {title: [t1, t2, ...], sent_id: [id1, id2, ...]}

    Note: 2Wiki uses 'content' instead of 'sentences' for context paragraphs.
    """
    context = metadata.get("context", {})
    supporting_facts = metadata.get("supporting_facts", {})

    if not context or not supporting_facts:
        return []

    ctx_titles = context.get("title", [])
    # 2Wiki uses "content" instead of "sentences"
    ctx_content = context.get("content", context.get("sentences", []))
    sf_titles = set(supporting_facts.get("title", []))

    gold_docs = []
    seen_titles = set()
    for i, title in enumerate(ctx_titles):
        if title in sf_titles and title not in seen_titles:
            seen_titles.add(title)
            if i < len(ctx_content):
                text = " ".join(ctx_content[i])
                contents = f'"{title}"\n{text}'
                gold_docs.append({"contents": contents})

    return gold_docs


def extract_gold_docs_musique(metadata):
    """Extract gold supporting paragraphs from MuSiQue metadata.

    metadata.question_decomposition: list of steps, each with
    support_paragraph: {title, paragraph_text}
    """
    decomposition = metadata.get("question_decomposition", [])
    if not decomposition:
        return []

    gold_docs = []
    seen_titles = set()
    for step in decomposition:
        sp = step.get("support_paragraph", {})
        if not sp:
            continue
        title = sp.get("title", "")
        text = sp.get("paragraph_text", "")
        if title and text and title not in seen_titles:
            seen_titles.add(title)
            contents = f'"{title}"\n{text}'
            gold_docs.append({"contents": contents})

    return gold_docs


DATASET_EXTRACTORS = {
    "hotpotqa": extract_gold_docs_hotpotqa,
    "2wikimultihopqa": extract_gold_docs_2wiki,
    "musique": extract_gold_docs_musique,
}


def main():
    parser = argparse.ArgumentParser(description="Gold context (perfect retrieval)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    from flashrag.config import Config
    config = Config(config_file_path=args.config)

    dataset_name = config["dataset_name"]
    save_dir = config["save_dir"]
    generator_model = config["generator_model"]
    max_tokens = config["generation_params"]["max_tokens"]

    print(f"=== Gold Context (Perfect Retrieval) ===")
    print(f"Dataset: {dataset_name}")
    print(f"Generator: {generator_model}")
    print(f"Max tokens: {max_tokens}")
    print(f"Save dir: {save_dir}")

    # Select extractor
    extractor = DATASET_EXTRACTORS.get(dataset_name)
    if extractor is None:
        raise ValueError(f"No gold extractor for dataset: {dataset_name}. "
                         f"Available: {list(DATASET_EXTRACTORS.keys())}")

    # Load dataset
    from flashrag.utils import get_dataset
    all_split = get_dataset(config)
    test_data = all_split["test"]
    print(f"Loaded {len(test_data)} test examples")

    # Extract gold docs
    print(f"\nExtracting gold documents...")
    all_gold_docs = []
    n_empty = 0
    total_docs = 0
    for item in test_data:
        meta = item.metadata if hasattr(item, 'metadata') else {}
        gold_docs = extractor(meta)
        if not gold_docs:
            n_empty += 1
        total_docs += len(gold_docs)
        all_gold_docs.append(gold_docs)

    avg_docs = total_docs / len(test_data) if test_data else 0
    print(f"  Average gold docs per question: {avg_docs:.2f}")
    print(f"  Questions with no gold docs: {n_empty}/{len(test_data)}")

    # Show sample gold doc
    for i in range(min(3, len(all_gold_docs))):
        if all_gold_docs[i]:
            print(f"\n  Sample gold docs for Q{i}: {test_data[i].question[:80]}...")
            for j, doc in enumerate(all_gold_docs[i][:2]):
                print(f"    Doc {j+1}: {doc['contents'][:120]}...")
            break

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(generator_model)

    # Build prompts with gold context
    print(f"\nBuilding prompts with gold context...")
    prompts = []
    for i, item in enumerate(test_data):
        gold_docs = all_gold_docs[i]
        if not gold_docs:
            # Fallback: use naive prompt for items with no gold docs
            fallback_sys = ("Answer the following question. Give a concise answer: "
                            "a single entity, name, number, yes/no, or short phrase.")
            messages = [
                {"role": "system", "content": fallback_sys},
                {"role": "user", "content": f"Question: {item.question}"},
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(build_prompt(item.question, gold_docs, tokenizer))

    # Show sample prompt
    print(f"\n--- Sample prompt ---")
    print(prompts[0][:800])
    print(f"---\n")

    # Generate with vLLM
    print(f"Initializing vLLM generator...")
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=generator_model,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        max_model_len=16384,
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
    )

    print(f"Generating answers for {len(prompts)} questions...")
    t_start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t_gen = time.time() - t_start
    print(f"Generation: {t_gen:.1f}s ({t_gen/len(prompts):.3f}s/example)")

    predictions = [o.outputs[0].text.strip() for o in outputs]

    # Evaluate
    ems, f1s = [], []
    for i, item in enumerate(test_data):
        gold = item.golden_answers
        pred = predictions[i]
        ems.append(compute_em(pred, gold))
        f1s.append(compute_f1(pred, gold))

    avg_em = sum(ems) / len(ems)
    avg_f1 = sum(f1s) / len(f1s)

    print(f"\n{'='*60}")
    print(f"GOLD CONTEXT RESULTS (n={len(ems)})")
    print(f"{'='*60}")
    print(f"  EM:  {avg_em:.4f} ({100*avg_em:.2f}%)")
    print(f"  F1:  {avg_f1:.4f} ({100*avg_f1:.2f}%)")

    # Show examples
    print(f"\n--- Sample predictions (first 5) ---")
    for i in range(min(5, len(predictions))):
        print(f"  Q: {test_data[i].question}")
        print(f"  Pred: {predictions[i]}")
        print(f"  Gold: {test_data[i].golden_answers}")
        print(f"  #GoldDocs: {len(all_gold_docs[i])}")
        print(f"  EM={ems[i]:.0f} F1={f1s[i]:.3f}")
        print()

    # Save results
    os.makedirs(save_dir, exist_ok=True)

    # Save metric scores
    metric_path = os.path.join(save_dir, "metric_score.txt")
    with open(metric_path, "w") as f:
        f.write(f"em: {avg_em}\n")
        f.write(f"f1: {avg_f1}\n")
    print(f"Saved metrics to {metric_path}")

    # Save intermediate data (matching FlashRAG format for bootstrap analysis)
    intermediate = []
    for i, item in enumerate(test_data):
        intermediate.append({
            "id": item.id if hasattr(item, 'id') else str(i),
            "question": item.question,
            "golden_answers": item.golden_answers,
            "metadata": item.metadata if hasattr(item, 'metadata') else {},
            "output": {
                "pred": predictions[i],
                "prompt": prompts[i],
                "gold_docs": all_gold_docs[i],
                "retrieval_result": all_gold_docs[i],  # For compatibility with analysis
            },
        })

    data_path = os.path.join(save_dir, "intermediate_data.json")
    with open(data_path, "w") as f:
        json.dump(intermediate, f, indent=2, ensure_ascii=False)
    print(f"Saved intermediate data to {data_path}")

    # Save per-item scores for bootstrap
    scores_path = os.path.join(save_dir, "per_item_scores.json")
    per_item = [{"id": intermediate[i]["id"], "em": ems[i], "f1": f1s[i]}
                for i in range(len(ems))]
    with open(scores_path, "w") as f:
        json.dump(per_item, f, indent=2)
    print(f"Saved per-item scores to {scores_path}")

    print(f"\nTotal time: {t_gen:.1f}s")
    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
