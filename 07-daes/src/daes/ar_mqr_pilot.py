"""Matched AR multi-query retrieval smoke baseline.

Single-round AR-MQR using the same retrieval stack as DNMR:
  baseline - initial retrieval C0, AR decode
  ar_mqr   - initial retrieval, AR seed answer, AR multi-query expansion, AR decode
"""

import argparse
import json
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eamd_v2_wiki18 import QUESTION_FILES, Wiki18Retriever, compute_f1, compute_em


AR_SHORT_INSTRUCTIONS = """You are a helpful assistant.
Answer the question using the context when possible.
Give a direct concise answer in 1 to 6 words.
Do not explain.
Do not write a sentence if a short phrase is enough.
"""


def extract_visible_answer(text: str) -> str:
    # If </think> present, take everything after it
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "</thinking>" in text:
        text = text.split("</thinking>")[-1]
    elif "<think>" in text and "</think>" not in text:
        # Thinking started but never finished (token limit hit)
        # Try to find any answer-like content in the thinking
        # Fall back to empty — better than returning thinking content
        return ""
    # Strip any remaining think tags
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Look for explicit answer markers
    for line in reversed(lines):
        lower = line.lower()
        if lower.startswith("final answer:") or lower.startswith("answer:"):
            return line.split(":", 1)[1].strip()
    # Fallback: last non-empty line
    return lines[-1] if lines else text.strip()


def normalize_candidate(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "</thinking>" in text:
        text = text.split("</thinking>")[-1]
    text = text.strip()
    text = text.split("\n")[0].split(". ")[0].strip()
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[:8])
    return text.strip()


def build_prompt(
    context: str,
    question: str,
    answer_prefix: str = "Answer:",
    thinking: bool = False,
) -> str:
    format_hint = (
        "You may think first, but end with exactly one line formatted as 'Final answer: <short phrase>'."
        if thinking
        else ""
    )
    return (
        f"{AR_SHORT_INSTRUCTIONS}\n"
        f"{format_hint}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"{answer_prefix}"
    )


def apply_qwen_template(tokenizer, prompt: str, thinking: bool = False) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.inference_mode()
def ar_generate_answer(
    model,
    tokenizer,
    context: str,
    question: str,
    max_new_tokens: int = 32,
    thinking: bool = False,
) -> str:
    effective_tokens = max_new_tokens if not thinking else max(max_new_tokens, 8192)
    text = apply_qwen_template(tokenizer, build_prompt(context, question, thinking=thinking), thinking=thinking)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=effective_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    answer = extract_visible_answer(raw)
    return normalize_candidate(answer)


@torch.inference_mode()
def ar_generate_candidates(
    model,
    tokenizer,
    context: str,
    question: str,
    n_candidates: int = 3,
    max_new_tokens: int = 16,
    thinking: bool = False,
) -> list[dict]:
    effective_tokens = max_new_tokens if not thinking else max(max_new_tokens, 4096)
    text = apply_qwen_template(
        tokenizer,
        build_prompt(context, question, answer_prefix="The answer is:", thinking=thinking),
        thinking=thinking,
    )
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=effective_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=n_candidates * 2,
        pad_token_id=tokenizer.eos_token_id,
    )

    candidates = []
    seen = set()
    for seq in outputs:
        text_out = tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
        cand = normalize_candidate(extract_visible_answer(text_out))
        if not cand:
            continue
        key = cand.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"text": cand})
        if len(candidates) >= n_candidates:
            break
    return candidates


def expand_evidence(retriever, question: str, seed_answer: str, bridge_cands: list[dict], current_passages: list[str], expand_top_k: int = 3):
    queries = [f"query: {question} {seed_answer}"]
    for cand in bridge_cands:
        text = cand.get("text", "") if isinstance(cand, dict) else str(cand)
        if text and len(text) > 1:
            queries.append(f"query: {question} {text}")
    results = retriever.retrieve_batch(queries, expand_top_k)
    existing = set(current_passages)
    new_passages = []
    for result_list in results:
        for passage in result_list:
            if passage not in existing:
                new_passages.append(passage)
                existing.add(passage)
    return list(current_passages) + new_passages, new_passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--seed_tokens", type=int, default=16)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()

    t_start = time.time()
    print("=== AR-MQR Pilot ===", flush=True)
    print(f"Dataset: {args.dataset}, n_questions: {args.n_questions}, thinking: {args.thinking}", flush=True)

    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    print(f"  AR model loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    print("Batch initial retrieval...", flush=True)
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, args.initial_top_k)
    print(f"  Done in {time.time() - t_start:.1f}s", flush=True)

    methods = ["baseline_ar", "ar_mqr"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        baseline_ans = ar_generate_answer(
            model,
            tokenizer,
            old_ctx,
            qtext,
            max_new_tokens=args.answer_tokens,
            thinking=args.thinking,
        )

        seed_ans = ar_generate_answer(
            model,
            tokenizer,
            old_ctx,
            qtext,
            max_new_tokens=args.seed_tokens,
            thinking=args.thinking,
        )
        bridge_cands = ar_generate_candidates(
            model,
            tokenizer,
            old_ctx,
            qtext,
            n_candidates=args.n_candidates,
            max_new_tokens=args.seed_tokens,
            thinking=args.thinking,
        )
        ar_passages, new_p = expand_evidence(retriever, qtext, seed_ans, bridge_cands, initial, args.expand_top_k)
        ar_ctx = "\n\n".join(ar_passages)
        ar_mqr_ans = ar_generate_answer(
            model,
            tokenizer,
            ar_ctx,
            qtext,
            max_new_tokens=args.answer_tokens,
            thinking=args.thinking,
        )

        elapsed = time.time() - tq
        row = {
            "id": q.get("qid") or q.get("id", f"dev_{qi}"),
            "question": qtext,
            "gold": gold,
            "elapsed": round(elapsed, 2),
            "seed_answer": seed_ans,
            "bridge_candidates": [c["text"] for c in bridge_cands],
            "new_passages": len(new_p),
        }

        for method, ans in [("baseline_ar", baseline_ans), ("ar_mqr", ar_mqr_ans)]:
            f1_result = compute_f1(ans, gold)
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[2]
            em = compute_em(ans, gold)
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans, "f1": round(f1, 4), "em": em, "contain": contain}

        results.append(row)

        log_this = (qi + 1) % args.log_every == 0 or qi == 0 or qi == len(questions) - 1
        if log_this:
            print(f"[{qi+1}/{len(questions)}] {row['id']} ({elapsed:.1f}s)", flush=True)
            for m in methods:
                print(f"  {m:16s} {row[m]['answer'][:40]:40s} F1={row[m]['f1']:.3f} EM={row[m]['em']:.0f}", flush=True)

        if (qi + 1) % args.save_every == 0 or qi == len(questions) - 1:
            n_done = len(results)
            summary = {m: {k: round(v / max(1, n_done), 4) for k, v in totals[m].items()} for m in methods}
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "summary": summary,
                        "results": results,
                        "config": vars(args),
                        "timing": {"elapsed_sec": round(time.time() - t_start, 1)},
                    },
                    f,
                    indent=2,
                )

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(f"\n{'Method':<16s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 38)
    for m in methods:
        s = summary[m]
        print(f"{m:<16s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")


if __name__ == "__main__":
    main()
