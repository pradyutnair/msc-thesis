import argparse
import json
import math
import os
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from dgmqr import Retriever, extract_candidates
from spread_v2 import spread_generate
from aram_reproduce import aram_generate

MODEL_REF = None
TOKENIZER_REF = None

SHORT_INSTRUCTIONS = """You are a helpful assistant.
Answer the question using the context when possible.
Give a direct concise answer in 1 to 6 words.
Do not explain.
Do not write a sentence if a short phrase is enough.
"""


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def compute_f1(pred: str, gold: str) -> tuple[float, float, float]:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return precision, recall, 2 * precision * recall / (precision + recall)


def build_short_prompt(tokenizer, context: str, question: str) -> tuple[list[int], int]:
    prompt = (
        f"{SHORT_INSTRUCTIONS}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    return prefix_ids, len(prefix_ids)


def short_generate(model, tokenizer, context: str, question: str, steps: int = 16,
                   n_tokens: int = 16, temperature: float = 0.1):
    device = model.device
    mask_id = tokenizer.mask_token_id
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k_per_step = max(1, math.ceil(n_tokens / steps))
    remaining = n_tokens
    token_confidences = []

    for step in range(steps):
        if remaining <= 0:
            break
        mask_idx = x == mask_id
        if not mask_idx.any():
            break
        with torch.no_grad():
            out = model(x, attention_mask=attn)
        logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
        mask_pos = mask_idx[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mask_pos], temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(conf, min(n_commit, len(conf)))
        selected = mask_pos[topk]
        x[0, selected] = x0[topk]
        token_confidences.extend(conf[topk].tolist())
        remaining -= len(topk)

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()
    answer_text = tokenizer.decode(answer_tokens.tolist(), skip_special_tokens=True).strip()
    avg_conf = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
    return answer_text, answer_tokens, avg_conf


def unique_passages(passages: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for passage in passages:
        key = passage[:120]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(passage)
    return deduped


def expand_evidence(retriever: Retriever, question: str, initial_context: str, initial_passages: list[str],
                    init_answer: str, n_candidates: int = 3) -> tuple[list[str], list[dict]]:
    candidates = extract_candidates(MODEL_REF, TOKENIZER_REF, initial_context, question, n_candidates)
    all_passages = list(initial_passages)

    if init_answer and len(init_answer.strip()) > 2:
        all_passages.extend(retriever.retrieve(f"{question} {init_answer[:100]}", top_k=3))

    for cand in candidates:
        all_passages.extend(retriever.retrieve(f"{question} {cand['text']}", top_k=3))

    return unique_passages(all_passages), candidates


def eamd_guided_short(model, tokenizer, question: str, old_context: str, new_context: str,
                      steps: int = 16, n_tokens: int = 16, temperature: float = 0.1,
                      lambda_max: float = 1.0, beta: float = 0.5, eps: float = 1e-6):
    device = model.device
    mask_id = tokenizer.mask_token_id

    old_prefix_ids, old_n_prefix = build_short_prompt(tokenizer, old_context, question)
    new_prefix_ids, new_n_prefix = build_short_prompt(tokenizer, new_context, question)

    x_base = torch.tensor([old_prefix_ids + [mask_id] * n_tokens], dtype=torch.long, device=device)
    x_full = torch.tensor([new_prefix_ids + [mask_id] * n_tokens], dtype=torch.long, device=device)
    attn_base = torch.ones((1, x_base.shape[1]), dtype=torch.long, device=device)
    attn_full = torch.ones((1, x_full.shape[1]), dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    token_confidences = []
    signal_means = []
    lambda_means = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, new_n_prefix:] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + new_n_prefix
        base_pos = masked_local + old_n_prefix

        with torch.no_grad():
            out_full = model(x_full, attention_mask=attn_full)
            out_base = model(x_base, attention_mask=attn_base)

        logits_full = torch.cat([out_full.logits[:, :1], out_full.logits[:, :-1]], dim=1)[0, full_pos]
        logits_base = torch.cat([out_base.logits[:, :1], out_base.logits[:, :-1]], dim=1)[0, base_pos]

        log_p_full = F.log_softmax(logits_full, dim=-1)
        log_p_base = F.log_softmax(logits_base, dim=-1)
        p_full = log_p_full.exp()
        p_base = log_p_base.exp()

        kl_fwd = (p_full * (log_p_full - log_p_base)).sum(dim=-1)
        kl_rev = (p_base * (log_p_base - log_p_full)).sum(dim=-1)
        signal = kl_fwd + kl_rev
        noise = -(p_full * log_p_full).sum(dim=-1)
        schedule = float(step + 1) / float(steps)
        extra_scale = lambda_max * torch.tanh(beta * signal / (noise + eps)) * schedule
        guidance_scale = 1.0 + extra_scale
        guided_logits = logits_base + guidance_scale.unsqueeze(-1) * (logits_full - logits_base)

        confidence, x0 = sample_tokens(guided_logits, temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, len(confidence)))

        chosen_local = masked_local[topk]
        chosen_full = chosen_local + new_n_prefix
        chosen_base = chosen_local + old_n_prefix
        x_full[0, chosen_full] = x0[topk]
        x_base[0, chosen_base] = x0[topk]
        token_confidences.extend(confidence[topk].tolist())
        signal_means.append(signal.mean().item())
        lambda_means.append(guidance_scale.mean().item())
        remaining -= len(topk)

    answer_tokens = x_full[0, new_n_prefix:new_n_prefix + n_tokens]
    answer_text = tokenizer.decode(answer_tokens.tolist(), skip_special_tokens=True).strip()
    avg_conf = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
    stats = {
        "mean_signal": sum(signal_means) / len(signal_means) if signal_means else 0.0,
        "mean_guidance_scale": sum(lambda_means) / len(lambda_means) if lambda_means else 0.0,
        "avg_conf": avg_conf,
    }
    return answer_text, stats


def evaluate(pred: str, gold: str) -> dict:
    precision, recall, f1 = compute_f1(pred, gold)
    return {
        "pred": pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "contain": normalize_answer(gold) in normalize_answer(pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=5)
    parser.add_argument("--output", default="/projects/prjs1800/msc-thesis/07-daes/results/eamd_smoke_v2.json")
    args = parser.parse_args()

    retriever = Retriever(args.dataset)

    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"

    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())

    global MODEL_REF, TOKENIZER_REF
    MODEL_REF = model
    TOKENIZER_REF = tokenizer

    questions = json.load(open(f"/projects/prjs1800/external/arag/data/{args.dataset}/questions.json"))[args.start_idx:args.start_idx + args.n_questions]

    totals = {name: 0.0 for name in ["baseline", "spread", "aram", "pool", "eamd_guided", "eamd_select"]}
    results = []

    for idx, q in enumerate(questions, start=1):
        qid = q["id"]
        question = q["question"]
        gold = q["answer"]
        print(f"[{idx}/{len(questions)}] {qid}", flush=True)

        initial_passages = retriever.retrieve(question, top_k=5)
        old_context = "\n\n".join(initial_passages)
        init_answer, _, _ = short_generate(model, tokenizer, old_context, question)
        expanded_passages, candidates = expand_evidence(retriever, question, old_context, initial_passages, init_answer, 3)
        new_context = "\n\n".join(expanded_passages)
        candidate_texts = [cand["text"] for cand in candidates]

        t0 = time.time()

        baseline_answer, _, baseline_conf = short_generate(model, tokenizer, old_context, question)
        baseline = evaluate(baseline_answer, gold)
        baseline["avg_conf"] = baseline_conf
        print(f"  baseline:    {baseline_answer} | F1={baseline['f1']:.3f}", flush=True)

        spread_answer, spread_stats = spread_generate(
            model, tokenizer, old_context, question,
            n_tokens=16, steps=16, temperature=0.1,
            mode="spread_weighted", alpha=0.5,
        )
        spread = evaluate(spread_answer, gold)
        spread["stats"] = spread_stats
        print(f"  spread:      {spread_answer} | F1={spread['f1']:.3f}", flush=True)

        aram_answer, aram_stats = aram_generate(
            model, tokenizer, old_context, question,
            n_tokens=16, steps=16, temperature=0.1,
            lambda_max=1.0, beta=0.5, mode="aram",
        )
        aram = evaluate(aram_answer, gold)
        aram["stats"] = aram_stats
        print(f"  aram:        {aram_answer} | F1={aram['f1']:.3f}", flush=True)

        pool_answer, _, pool_conf = short_generate(model, tokenizer, new_context, question)
        pool = evaluate(pool_answer, gold)
        pool["avg_conf"] = pool_conf
        print(f"  pool_short:  {pool_answer} | F1={pool['f1']:.3f}", flush=True)

        eamd_answer, eamd_stats = eamd_guided_short(model, tokenizer, question, old_context, new_context)
        eamd = evaluate(eamd_answer, gold)
        eamd["stats"] = eamd_stats
        print(f"  eamd_short:  {eamd_answer} | F1={eamd['f1']:.3f}", flush=True)

        selected_answer = eamd_answer if eamd_stats["avg_conf"] >= pool_conf else pool_answer
        eamd_select = evaluate(selected_answer, gold)
        eamd_select["selected"] = "eamd_guided" if eamd_stats["avg_conf"] >= pool_conf else "pool_short"
        print(f"  selected:    {selected_answer} | F1={eamd_select['f1']:.3f} via {eamd_select['selected']}", flush=True)

        row = {
            "id": qid,
            "question": question,
            "gold": gold,
            "init_answer": init_answer,
            "candidates": candidate_texts,
            "n_passages_old": len(initial_passages),
            "n_passages_new": len(expanded_passages),
            "baseline": baseline,
            "spread": spread,
            "aram": aram,
            "pool": pool,
            "eamd_guided": eamd,
            "eamd_select": eamd_select,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        results.append(row)
        for key in totals:
            totals[key] += row[key]["f1"]

    summary = {
        key: {
            "f1": totals[key] / len(results),
            "contain": sum(1 for row in results if row[key]["contain"]) / len(results),
        }
        for key in totals
    }

    payload = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSummary", flush=True)
    for key, values in summary.items():
        print(f"  {key:12s} F1={values['f1']:.3f} contain={values['contain']:.3f}", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
