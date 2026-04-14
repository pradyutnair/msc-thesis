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


def build_prompt(tokenizer, context: str, question: str) -> tuple[list[int], int]:
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    return prefix_ids, len(prefix_ids)


def dllm_generate_tracked(model, tokenizer, context: str, question: str, steps: int = 128,
                          n_tokens: int = 512, temperature: float = 0.1):
    device = model.device
    mask_id = tokenizer.mask_token_id
    prefix_ids, n_prefix = build_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k_per_step = max(1, n_tokens // steps)
    remaining = n_tokens
    token_confidences = torch.zeros(n_tokens, device=device)

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
        for idx, pos in enumerate(selected):
            local_pos = pos.item() - n_prefix
            if 0 <= local_pos < n_tokens:
                token_confidences[local_pos] = conf[topk[idx]]
        remaining -= len(topk)

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()
    answer_text = tokenizer.decode(answer_tokens.tolist(), skip_special_tokens=True).strip()
    return answer_text, answer_tokens, token_confidences


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
        cand_text = cand["text"]
        all_passages.extend(retriever.retrieve(f"{question} {cand_text}", top_k=3))

    return unique_passages(all_passages), candidates


def logits_for_answer_tokens(model, tokenizer, context: str, question: str, answer_tokens: torch.Tensor) -> torch.Tensor:
    device = model.device
    prefix_ids, n_prefix = build_prompt(tokenizer, context, question)
    seq = torch.tensor([prefix_ids + answer_tokens.tolist()], dtype=torch.long, device=device)
    attn = torch.ones((1, seq.shape[1]), dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(seq, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    return logits[0, n_prefix:n_prefix + len(answer_tokens)]


def select_remask_positions(model, tokenizer, question: str, old_context: str, new_context: str,
                            answer_tokens: torch.Tensor, remask_frac: float = 0.5) -> tuple[list[int], list[float]]:
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    content_positions = [
        idx for idx, tok in enumerate(answer_tokens.tolist())
        if tok not in (mask_id, eos_id)
    ]
    if not content_positions:
        return [], []

    old_logits = logits_for_answer_tokens(model, tokenizer, old_context, question, answer_tokens)
    new_logits = logits_for_answer_tokens(model, tokenizer, new_context, question, answer_tokens)

    divergences = []
    for idx in content_positions:
        log_p_old = F.log_softmax(old_logits[idx], dim=-1)
        log_p_new = F.log_softmax(new_logits[idx], dim=-1)
        p_old = log_p_old.exp()
        p_new = log_p_new.exp()
        kl_fwd = (p_new * (log_p_new - log_p_old)).sum().item()
        kl_rev = (p_old * (log_p_old - log_p_new)).sum().item()
        divergences.append((idx, kl_fwd + kl_rev))

    divergences.sort(key=lambda item: item[1], reverse=True)
    n_to_remask = max(1, math.ceil(len(divergences) * remask_frac))
    selected = divergences[:n_to_remask]
    return [idx for idx, _ in selected], [val for _, val in selected]


def eamd_refine(model, tokenizer, question: str, old_context: str, new_context: str, answer_tokens: torch.Tensor,
                remask_positions: list[int], steps: int = 64, temperature: float = 0.1,
                lambda_max: float = 1.0, beta: float = 0.5, eps: float = 1e-6) -> str:
    if not remask_positions:
        return tokenizer.decode(answer_tokens.tolist(), skip_special_tokens=True).strip()

    device = model.device
    mask_id = tokenizer.mask_token_id

    old_prefix_ids, old_n_prefix = build_prompt(tokenizer, old_context, question)
    new_prefix_ids, new_n_prefix = build_prompt(tokenizer, new_context, question)

    base_answer = answer_tokens.clone()
    base_answer[remask_positions] = mask_id

    x_base = torch.tensor([old_prefix_ids + base_answer.tolist()], dtype=torch.long, device=device)
    x_full = torch.tensor([new_prefix_ids + base_answer.tolist()], dtype=torch.long, device=device)
    attn_base = torch.ones((1, x_base.shape[1]), dtype=torch.long, device=device)
    attn_full = torch.ones((1, x_full.shape[1]), dtype=torch.long, device=device)

    remaining = len(remask_positions)
    k_per_step = max(1, math.ceil(remaining / max(steps, 1)))

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
        lam = lambda_max * torch.tanh(beta * signal / (noise + eps)) * schedule
        guided_logits = logits_base + lam.unsqueeze(-1) * (logits_full - logits_base)

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
        remaining -= len(topk)

    refined_tokens = x_full[0, new_n_prefix:new_n_prefix + len(answer_tokens)]
    return tokenizer.decode(refined_tokens.tolist(), skip_special_tokens=True).strip()


def run_pool(model, tokenizer, retriever: Retriever, question: str, n_candidates: int = 3) -> tuple[str, dict]:
    initial_passages = retriever.retrieve(question, top_k=5)
    initial_context = "\n\n".join(initial_passages)
    init_answer, _, _ = dllm_generate_tracked(model, tokenizer, initial_context, question)
    expanded_passages, candidates = expand_evidence(retriever, question, initial_context, initial_passages, init_answer, n_candidates)
    pooled_context = "\n\n".join(expanded_passages)
    pooled_answer, _, _ = dllm_generate_tracked(model, tokenizer, pooled_context, question)
    candidate_texts = [candidate["text"] for candidate in candidates]
    return pooled_answer, {"n_candidates": len(candidates), "n_passages": len(expanded_passages), "candidates": candidate_texts}


def run_eamd(model, tokenizer, retriever: Retriever, question: str, n_candidates: int = 3,
             remask_frac: float = 0.5) -> tuple[str, dict]:
    initial_passages = retriever.retrieve(question, top_k=5)
    old_context = "\n\n".join(initial_passages)
    init_answer, answer_tokens, _ = dllm_generate_tracked(model, tokenizer, old_context, question)
    expanded_passages, candidates = expand_evidence(retriever, question, old_context, initial_passages, init_answer, n_candidates)
    new_context = "\n\n".join(expanded_passages)
    remask_positions, remask_scores = select_remask_positions(model, tokenizer, question, old_context, new_context, answer_tokens, remask_frac=remask_frac)
    refined_answer = eamd_refine(model, tokenizer, question, old_context, new_context, answer_tokens, remask_positions)
    candidate_texts = [candidate["text"] for candidate in candidates]
    stats = {
        "init_answer": init_answer,
        "n_candidates": len(candidates),
        "candidates": candidate_texts,
        "n_passages_old": len(initial_passages),
        "n_passages_new": len(expanded_passages),
        "remask_positions": remask_positions,
        "remask_scores": [round(value, 4) for value in remask_scores],
    }
    return refined_answer, stats


def evaluate_method(name: str, pred: str, gold: str) -> dict:
    precision, recall, f1 = compute_f1(pred, gold)
    return {
        "method": name,
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
    parser.add_argument("--n_questions", type=int, default=3)
    parser.add_argument("--output", default="/projects/prjs1800/msc-thesis/07-daes/results/eamd_smoke.json")
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

    results = []
    totals = {name: 0.0 for name in ["baseline", "spread", "aram", "pool", "eamd"]}

    for idx, q in enumerate(questions, start=1):
        qid = q["id"]
        question = q["question"]
        gold = q["answer"]
        print(f"[{idx}/{len(questions)}] {qid}", flush=True)

        initial_passages = retriever.retrieve(question, top_k=5)
        context = "\n\n".join(initial_passages)

        t0 = time.time()

        baseline_answer, _ = aram_generate(model, tokenizer, context, question, mode="baseline")
        baseline_metrics = evaluate_method("baseline", baseline_answer, gold)
        print(f"  baseline: {baseline_answer} | F1={baseline_metrics['f1']:.3f}", flush=True)

        spread_answer, spread_stats = spread_generate(model, tokenizer, context, question, mode="spread_weighted", alpha=0.5)
        spread_metrics = evaluate_method("spread", spread_answer, gold)
        print(f"  spread:   {spread_answer} | F1={spread_metrics['f1']:.3f}", flush=True)

        aram_answer, aram_stats = aram_generate(model, tokenizer, context, question, mode="aram", lambda_max=1.0, beta=0.5)
        aram_metrics = evaluate_method("aram", aram_answer, gold)
        print(f"  aram:     {aram_answer} | F1={aram_metrics['f1']:.3f}", flush=True)

        pool_answer, pool_stats = run_pool(model, tokenizer, retriever, question)
        pool_metrics = evaluate_method("pool", pool_answer, gold)
        print(f"  pool:     {pool_answer} | F1={pool_metrics['f1']:.3f}", flush=True)

        eamd_answer, eamd_stats = run_eamd(model, tokenizer, retriever, question)
        eamd_metrics = evaluate_method("eamd", eamd_answer, gold)
        print(f"  eamd:     {eamd_answer} | F1={eamd_metrics['f1']:.3f}", flush=True)

        row = {
            "id": qid,
            "question": question,
            "gold": gold,
            "baseline": baseline_metrics,
            "spread": {**spread_metrics, "stats": spread_stats},
            "aram": {**aram_metrics, "stats": aram_stats},
            "pool": {**pool_metrics, "stats": pool_stats},
            "eamd": {**eamd_metrics, "stats": eamd_stats},
            "elapsed_sec": round(time.time() - t0, 2),
        }
        results.append(row)
        for key in totals:
            totals[key] += row[key]["f1"]

    summary = {}
    for key in totals:
        summary[key] = {
            "f1": totals[key] / len(results),
            "contain": sum(1 for row in results if row[key]["contain"]) / len(results),
        }

    payload = {"summary": summary, "results": results}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nSummary", flush=True)
    for key, values in summary.items():
        mean_f1 = values["f1"]
        contain = values["contain"]
        print(f"  {key:8s} F1={mean_f1:.3f} contain={contain:.3f}", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
