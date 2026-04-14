"""Single-question DNMR diagnostic for LLaDA/Dream on MuSiQue.

Runs a fixed set of intervention arms on a small hand-picked set of questions
and logs bridge extraction, retrieval, evidence expansion, and final answers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import eamd_v2_wiki18
from eamd_v2_wiki18 import (
    QUESTION_FILES,
    Wiki18Retriever,
    _clean_bridge_candidate,
    _neg_entropy,
    build_short_pair,
    build_short_prompt,
    compute_f1,
    decode_answer,
    extract_candidates_agnostic,
    get_mask_id,
    prepare_logits,
    sample_tokens,
)
from dnmr_pool_v2_lean import build_hint_v2, choose_answer_budget, simple_decode

MUSIQUE_DEV_FULL = "/projects/prjs1800/datasets/musique/musique_full_v1.0_dev.jsonl"
DEFAULT_QUESTION_IDS = ["dev_26", "dev_471", "dev_642"]


def load_questions(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def load_musique_full(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def contain_metric(answer: str, gold: str) -> float:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(answer)
    return float(bool(gold_norm) and gold_norm in pred_norm)


def compute_f1_value(answer: str, gold: str) -> float:
    result = compute_f1(answer, gold)
    if isinstance(result, tuple):
        return float(result[2])
    return float(result)


def shorten_answer(answer: str, max_words: int = 6) -> str:
    text = (answer or "").replace("\r", " ").strip()
    text = text.split("\n", 1)[0].strip()
    text = re.sub(r"^(?:short answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:the answer is|it is|it's)\s+", "", text, flags=re.IGNORECASE)
    for sep in [". ", "; ", " - ", " -- ", " because ", " which ", " who ", " that "]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    text = text.strip(" \t\n\r.,;:!?\"'()[]{}")
    words = text.split()
    if max_words > 0 and len(words) > max_words:
        text = " ".join(words[:max_words]).strip(" \t\n\r.,;:!?\"'()[]{}")
    return text


def unique_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def match_full_example(question_row: dict[str, Any], full_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    q = question_row["question"].strip()
    for row in full_rows:
        if row.get("question", "").strip() == q:
            return row
    return None


def get_gold_bridges(example: dict[str, Any] | None) -> list[str]:
    if not example:
        return []
    gold = normalize_answer(example.get("answer", ""))
    decomposition = example.get("question_decomposition") or []
    answers: list[str] = []
    seen = set()
    for step in decomposition:
        ans = (step.get("answer") or "").strip()
        if not ans:
            continue
        key = normalize_answer(ans)
        if key in seen:
            continue
        seen.add(key)
        answers.append(ans)
    bridges = [ans for ans in answers if normalize_answer(ans) != gold]
    if not bridges and len(answers) > 1:
        bridges = answers[:-1]
    return bridges


def preview_passage(passage: str, limit: int = 80) -> str:
    text = " ".join((passage or "").split())
    return text[:limit]


def passages_to_previews(passages: list[str], limit: int = 80) -> list[str]:
    return [preview_passage(p, limit=limit) for p in passages]


@torch.inference_mode()
def extract_candidates_mixed_custom(
    model,
    tokenizer,
    context: str,
    question: str,
    *,
    n_candidates: int = 4,
    n_branch: int = 3,
    n_mask: int = 8,
    extraction_steps: int = 16,
    min_position_mass: float = 0.02,
    position_temp: float = 0.3,
    sample_temp: float = 0.1,
    clean: bool = True,
) -> list[dict[str, Any]]:
    device = model.device
    mask_id = get_mask_id(tokenizer)
    full_ids, base_ids, n_prefix = build_short_pair(tokenizer, context, "", question, n_mask)

    x_base = torch.tensor([base_ids], dtype=torch.long, device=device)
    x_full = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn_base = torch.ones((1, len(base_ids)), dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)
    logits_pair = prepare_logits(model(torch.cat([x_base, x_full], dim=0), attention_mask=torch.cat([attn_base, attn_full], dim=0)).logits)

    answer_logits_base = logits_pair[0, n_prefix:n_prefix + n_mask]
    answer_logits_full = logits_pair[1, n_prefix:n_prefix + n_mask]
    log_p_full = F.log_softmax(answer_logits_full, dim=-1)
    log_p_base = F.log_softmax(answer_logits_base, dim=-1)
    p_full = log_p_full.exp()
    entropy = -(p_full * log_p_full).sum(dim=-1)
    info_gain = (p_full * (log_p_full - log_p_base)).sum(dim=-1).clamp_min(0.0)
    position_signal = entropy * info_gain
    if position_signal.sum().item() <= 0:
        position_signal = entropy.clamp_min(1e-8)
    if position_signal.sum().item() <= 0:
        position_signal = torch.ones_like(position_signal)
    position_mass = position_signal / position_signal.sum()

    selected_positions = [
        pos for pos, mass in enumerate(position_mass.tolist()) if mass >= min_position_mass
    ]
    if not selected_positions:
        selected_positions = [int(torch.argmax(position_mass).item())]

    branch_canvases: list[list[int]] = []
    branch_meta: list[dict[str, Any]] = []
    pos_temp = max(position_temp, 1e-4)
    for pos_local in selected_positions:
        pos_global = n_prefix + pos_local
        pos_probs = torch.softmax(answer_logits_full[pos_local] / pos_temp, dim=-1)
        top_probs, top_ids = torch.topk(pos_probs, min(n_branch, pos_probs.shape[-1]))
        for token_prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            canvas = list(full_ids)
            canvas[pos_global] = token_id
            branch_canvases.append(canvas)
            branch_meta.append(
                {
                    "position": pos_local,
                    "position_mass": float(position_mass[pos_local].item()),
                    "token_prob": float(token_prob),
                    "init_mass": float(position_mass[pos_local].item() * token_prob),
                }
            )

    if not branch_canvases:
        return []

    x_all = torch.tensor(branch_canvases, dtype=torch.long, device=device)
    attn_batch = torch.ones((len(branch_canvases), x_all.shape[1]), dtype=torch.long, device=device)
    neg_ent = _neg_entropy()
    remaining = torch.full((len(branch_canvases),), n_mask - 1, dtype=torch.long, device=device)

    for step in range(extraction_steps):
        active = remaining > 0
        if not active.any():
            break
        active_idx = active.nonzero(as_tuple=True)[0]
        logits_active = prepare_logits(model(x_all[active_idx], attention_mask=attn_batch[: len(active_idx)]).logits)
        for j, bi in enumerate(active_idx.tolist()):
            masked_positions = (x_all[bi] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_positions) == 0:
                remaining[bi] = 0
                continue
            conf, sampled = sample_tokens(
                logits_active[j, masked_positions],
                temperature=sample_temp,
                neg_entropy=neg_ent,
            )
            rem = remaining[bi].item()
            n_commit = min(max(1, rem // extraction_steps), rem)
            if step == extraction_steps - 1:
                n_commit = rem
            _, topk = torch.topk(conf, min(n_commit, len(conf)))
            x_all[bi, masked_positions[topk]] = sampled[topk]
            remaining[bi] -= len(topk)

    candidate_masses: dict[str, dict[str, Any]] = {}
    for bi, meta in enumerate(branch_meta):
        cand_text = tokenizer.decode(
            x_all[bi, n_prefix:n_prefix + n_mask].tolist(),
            skip_special_tokens=True,
        ).strip()
        if clean:
            cand_text = _clean_bridge_candidate(cand_text, max_words=6)
        if not cand_text or len(cand_text) <= 1:
            continue
        key = cand_text.lower()
        if key not in candidate_masses:
            candidate_masses[key] = {
                "text": cand_text,
                "init_conf": 0.0,
                "position_mass": 0.0,
                "positions": set(),
            }
        candidate_masses[key]["init_conf"] += meta["init_mass"]
        candidate_masses[key]["position_mass"] += meta["position_mass"]
        candidate_masses[key]["positions"].add(meta["position"])

    ranked = sorted(candidate_masses.values(), key=lambda item: item["init_conf"], reverse=True)
    for item in ranked:
        item["positions"] = sorted(item["positions"])
    return ranked[:n_candidates]


@torch.inference_mode()
def extract_candidates_context_subset(
    model,
    tokenizer,
    context: str,
    question: str,
    *,
    n_candidates: int = 4,
    n_mask: int = 8,
    n_branch: int = 2,
    extraction_steps: int = 12,
) -> list[dict[str, Any]]:
    passages = [p for p in context.split("\n\n") if p.strip()]
    if len(passages) <= 1:
        return []

    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    full_canvas = prefix_ids + [mask_id] * n_mask
    full_x = torch.tensor([full_canvas], dtype=torch.long, device=device)
    full_attn = torch.ones((1, len(full_canvas)), dtype=torch.long, device=device)
    full_logits = prepare_logits(model(full_x, attention_mask=full_attn).logits)[0, n_prefix:n_prefix + n_mask]
    full_log_probs = F.log_softmax(full_logits, dim=-1)
    full_probs = full_log_probs.exp()
    full_argmax = torch.argmax(full_probs, dim=-1)
    full_entropy = -(full_probs * full_log_probs).sum(dim=-1)

    branch_canvases: list[list[int]] = []
    branch_meta: list[dict[str, Any]] = []

    for drop_idx in range(len(passages)):
        subset_context = "\n\n".join(passages[:drop_idx] + passages[drop_idx + 1 :])
        subset_prefix, subset_n_prefix = build_short_prompt(tokenizer, subset_context, question)
        subset_canvas = subset_prefix + [mask_id] * n_mask
        subset_x = torch.tensor([subset_canvas], dtype=torch.long, device=device)
        subset_attn = torch.ones((1, len(subset_canvas)), dtype=torch.long, device=device)
        subset_logits = prepare_logits(model(subset_x, attention_mask=subset_attn).logits)[0, subset_n_prefix:subset_n_prefix + n_mask]
        subset_log_probs = F.log_softmax(subset_logits, dim=-1)
        subset_probs = subset_log_probs.exp()
        subset_argmax = torch.argmax(subset_probs, dim=-1)
        delta = (full_probs * (full_log_probs - subset_log_probs)).sum(dim=-1).clamp_min(0.0)
        scores = full_entropy * delta

        changed_positions = [
            pos for pos in range(n_mask)
            if int(full_argmax[pos].item()) != int(subset_argmax[pos].item())
        ]
        changed_positions = sorted(changed_positions, key=lambda pos: float(scores[pos].item()), reverse=True)[:2]

        for pos_local in changed_positions:
            pos_global = n_prefix + pos_local
            cand_token_ids = unique_preserve([
                int(full_argmax[pos_local].item()),
                int(subset_argmax[pos_local].item()),
            ])[:n_branch]
            for token_id in cand_token_ids:
                canvas = list(full_canvas)
                canvas[pos_global] = token_id
                branch_canvases.append(canvas)
                branch_meta.append(
                    {
                        "position": pos_local,
                        "dropped_passage_idx": drop_idx,
                        "drop_preview": preview_passage(passages[drop_idx], limit=60),
                        "init_mass": float(max(scores[pos_local].item(), 1e-8)),
                    }
                )

    if not branch_canvases:
        return []

    x_all = torch.tensor(branch_canvases, dtype=torch.long, device=device)
    attn_batch = torch.ones((len(branch_canvases), x_all.shape[1]), dtype=torch.long, device=device)
    neg_ent = _neg_entropy()
    remaining = torch.full((len(branch_canvases),), n_mask - 1, dtype=torch.long, device=device)

    for step in range(extraction_steps):
        active = remaining > 0
        if not active.any():
            break
        active_idx = active.nonzero(as_tuple=True)[0]
        logits_active = prepare_logits(model(x_all[active_idx], attention_mask=attn_batch[: len(active_idx)]).logits)
        for j, bi in enumerate(active_idx.tolist()):
            masked_positions = (x_all[bi] == mask_id).nonzero(as_tuple=True)[0]
            if len(masked_positions) == 0:
                remaining[bi] = 0
                continue
            conf, sampled = sample_tokens(logits_active[j, masked_positions], temperature=0.1, neg_entropy=neg_ent)
            rem = remaining[bi].item()
            n_commit = min(max(1, rem // extraction_steps), rem)
            if step == extraction_steps - 1:
                n_commit = rem
            _, topk = torch.topk(conf, min(n_commit, len(conf)))
            x_all[bi, masked_positions[topk]] = sampled[topk]
            remaining[bi] -= len(topk)

    pooled: dict[str, dict[str, Any]] = {}
    for bi, meta in enumerate(branch_meta):
        cand_text = tokenizer.decode(
            x_all[bi, n_prefix:n_prefix + n_mask].tolist(),
            skip_special_tokens=True,
        ).strip()
        cand_text = _clean_bridge_candidate(cand_text, max_words=6)
        if not cand_text or len(cand_text) <= 1:
            continue
        key = cand_text.lower()
        if key not in pooled:
            pooled[key] = {
                "text": cand_text,
                "init_conf": 0.0,
                "positions": set(),
                "drop_previews": [],
            }
        pooled[key]["init_conf"] += meta["init_mass"]
        pooled[key]["positions"].add(meta["position"])
        pooled[key]["drop_previews"].append(meta["drop_preview"])

    ranked = sorted(pooled.values(), key=lambda item: item["init_conf"], reverse=True)
    for item in ranked:
        item["positions"] = sorted(item["positions"])
    return ranked[:n_candidates]


def retrieve_and_expand(
    retriever: Wiki18Retriever,
    *,
    question: str,
    initial_passages: list[str],
    seed_answer: str,
    bridge_cands: list[dict[str, Any]],
    expand_top_k: int,
    include_seed_query: bool = True,
    bridge_queries: list[str] | None = None,
) -> dict[str, Any]:
    queries: list[str] = []
    if include_seed_query and seed_answer.strip():
        queries.append(f"query: {question} {seed_answer.strip()}")
    if bridge_queries is None:
        for cand in bridge_cands:
            text = (cand.get("text", "") if isinstance(cand, dict) else str(cand)).strip()
            if text and len(text) > 1:
                queries.append(f"query: {question} {text}")
    else:
        queries.extend(bridge_queries)
    queries = unique_preserve(queries)

    existing = set(initial_passages)
    new_passages: list[str] = []
    retrieved_by_query: list[dict[str, Any]] = []
    if queries:
        for query, hits in zip(queries, retriever.retrieve_batch(queries, expand_top_k)):
            additions: list[str] = []
            for passage in hits:
                if passage not in existing:
                    existing.add(passage)
                    new_passages.append(passage)
                    additions.append(passage)
            retrieved_by_query.append(
                {
                    "query": query,
                    "added_passage_previews": passages_to_previews(additions),
                }
            )

    expanded_passages = list(initial_passages) + new_passages
    return {
        "queries": queries,
        "retrieved_by_query": retrieved_by_query,
        "expanded_passages": expanded_passages,
        "new_passages": new_passages,
    }


def run_arm(
    *,
    arm_name: str,
    model,
    tokenizer,
    retriever: Wiki18Retriever,
    question: str,
    gold: str,
    initial_passages: list[str],
    seed_answer: str,
    gold_bridges: list[str],
    steps: int,
    answer_tokens: int,
    expand_top_k: int,
    n_candidates: int,
) -> dict[str, Any]:
    context = "\n\n".join(initial_passages)
    bridge_cands: list[dict[str, Any]] = []
    hint = ""
    retrieval_meta: dict[str, Any]

    if arm_name == "standard_pool":
        bridge_cands = extract_candidates_agnostic(
            model, tokenizer, context, question, n_candidates=n_candidates, n_mask=12, extraction_steps=12
        )
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "pool_8":
        bridge_cands = extract_candidates_agnostic(
            model, tokenizer, context, question, n_candidates=n_candidates, n_mask=8, extraction_steps=12
        )
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "pool_8_clean":
        raw = extract_candidates_agnostic(
            model, tokenizer, context, question, n_candidates=n_candidates * 2, n_mask=8, extraction_steps=12
        )
        cleaned = []
        seen = set()
        for cand in raw:
            text = _clean_bridge_candidate(cand.get("text", ""), max_words=6)
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            cleaned.append({**cand, "text": text})
            if len(cleaned) >= n_candidates:
                break
        bridge_cands = cleaned
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "pool_kl":
        bridge_cands = extract_candidates_mixed_custom(
            model,
            tokenizer,
            context,
            question,
            n_candidates=n_candidates,
            n_mask=8,
            extraction_steps=12,
            clean=True,
        )
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "pool_kl_hint":
        bridge_cands = extract_candidates_mixed_custom(
            model,
            tokenizer,
            context,
            question,
            n_candidates=n_candidates,
            n_mask=8,
            extraction_steps=12,
            clean=True,
        )
        hint = build_hint_v2(bridge_cands)
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "pool_temp_high":
        bridge_cands = extract_candidates_mixed_custom(
            model,
            tokenizer,
            context,
            question,
            n_candidates=n_candidates,
            n_mask=8,
            extraction_steps=12,
            position_temp=1.0,
            sample_temp=1.0,
            clean=True,
        )
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "pool_nmask6":
        bridge_cands = extract_candidates_mixed_custom(
            model,
            tokenizer,
            context,
            question,
            n_candidates=n_candidates,
            n_mask=6,
            extraction_steps=12,
            clean=True,
        )
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "pool_subset":
        bridge_cands = extract_candidates_context_subset(
            model,
            tokenizer,
            context,
            question,
            n_candidates=n_candidates,
            n_mask=8,
            extraction_steps=12,
        )
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=bridge_cands,
            expand_top_k=expand_top_k,
        )
    elif arm_name == "oracle_bridge":
        oracle_queries = [f"query: {question} {bridge}" for bridge in gold_bridges if bridge.strip()]
        retrieval_meta = retrieve_and_expand(
            retriever,
            question=question,
            initial_passages=initial_passages,
            seed_answer=seed_answer,
            bridge_cands=[],
            expand_top_k=expand_top_k,
            include_seed_query=False,
            bridge_queries=oracle_queries,
        )
    else:
        raise ValueError(f"Unknown arm: {arm_name}")

    final_context = "\n\n".join(retrieval_meta["expanded_passages"])
    if hint:
        final_context = hint + "\n\n" + final_context
    final_tokens = choose_answer_budget(question)
    final_answer = simple_decode(
        model,
        tokenizer,
        final_context,
        question,
        steps=steps,
        n_tokens=final_tokens,
    )
    final_answer = shorten_answer(final_answer, max_words=6)
    f1 = compute_f1_value(final_answer, gold)
    em = float(normalize_answer(final_answer) == normalize_answer(gold))
    contain = contain_metric(final_answer, gold)

    return {
        "arm": arm_name,
        "bridge_candidates": bridge_cands,
        "hint": hint,
        "retrieval_queries": retrieval_meta["queries"],
        "retrieved_by_query": retrieval_meta["retrieved_by_query"],
        "initial_passage_previews": passages_to_previews(initial_passages),
        "expanded_passage_previews": passages_to_previews(retrieval_meta["expanded_passages"]),
        "final_answer": final_answer,
        "metrics": {
            "f1": round(f1, 4),
            "em": round(em, 4),
            "contain": round(contain, 4),
        },
        "success": bool(em or contain > 0.0),
    }


def load_model_and_tokenizer(model_name: str):
    if model_name == "dream":
        import dllm

        model_args = SimpleNamespace(model_name_or_path="Dream-org/Dream-v0-Instruct-7B")
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada", choices=["llada", "dream"])
    parser.add_argument("--dataset", default="musique", choices=sorted(QUESTION_FILES))
    parser.add_argument("--question_ids", nargs="+", default=DEFAULT_QUESTION_IDS)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "standard_pool",
            "pool_8",
            "pool_8_clean",
            "pool_kl",
            "pool_kl_hint",
            "pool_temp_high",
            "pool_nmask6",
            "pool_subset",
            "oracle_bridge",
        ],
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=32)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    t0 = time.time()
    print("=== DNMR Diagnostic ===", flush=True)
    print(f"Model={args.model} Dataset={args.dataset} QuestionIDs={','.join(args.question_ids)}", flush=True)

    question_rows = load_questions(QUESTION_FILES[args.dataset])
    selected = []
    for qid in args.question_ids:
        row = next((r for r in question_rows if r.get("id") == qid or r.get("qid") == qid), None)
        if row is None:
            raise KeyError(f"Question id not found: {qid}")
        selected.append(row)

    full_rows = load_musique_full(MUSIQUE_DEV_FULL) if args.dataset == "musique" else []
    full_match = {row.get("qid") or row.get("id"): match_full_example(row, full_rows) for row in selected}

    model, tokenizer = load_model_and_tokenizer(args.model)
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    query_texts = [f"query: {row['question']}" for row in selected]
    initial_batches = retriever.retrieve_batch(query_texts, args.initial_top_k)

    results: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {arm: {"successes": 0, "contain_hits": 0, "em_hits": 0} for arm in args.arms}

    for idx, row in enumerate(selected):
        qid = row.get("qid") or row.get("id", f"q{idx}")
        question = row["question"]
        gold = row.get("answer") or (row.get("golden_answers") or [""])[0]
        initial_passages = initial_batches[idx]
        initial_context = "\n\n".join(initial_passages)
        seed_answer = shorten_answer(
            simple_decode(model, tokenizer, initial_context, question, steps=args.steps, n_tokens=args.answer_tokens),
            max_words=6,
        )
        gold_bridges = get_gold_bridges(full_match.get(qid))
        print(f"[question] {qid} seed={seed_answer} gold={gold}", flush=True)

        arm_rows = []
        for arm in args.arms:
            t_arm = time.time()
            arm_result = run_arm(
                arm_name=arm,
                model=model,
                tokenizer=tokenizer,
                retriever=retriever,
                question=question,
                gold=gold,
                initial_passages=initial_passages,
                seed_answer=seed_answer,
                gold_bridges=gold_bridges,
                steps=args.steps,
                answer_tokens=args.answer_tokens,
                expand_top_k=args.expand_top_k,
                n_candidates=args.n_candidates,
            )
            arm_result["elapsed_sec"] = round(time.time() - t_arm, 2)
            arm_rows.append(arm_result)
            summary[arm]["successes"] += int(arm_result["success"])
            summary[arm]["contain_hits"] += int(arm_result["metrics"]["contain"] > 0.0)
            summary[arm]["em_hits"] += int(arm_result["metrics"]["em"] > 0.0)
            print(
                f"  {arm:14s} ans={arm_result['final_answer'][:40]:40s} "
                f"F1={arm_result['metrics']['f1']:.3f} EM={arm_result['metrics']['em']:.0f} "
                f"Contain={arm_result['metrics']['contain']:.0f} t={arm_result['elapsed_sec']:.1f}s",
                flush=True,
            )

        results.append(
            {
                "id": qid,
                "question": question,
                "gold": gold,
                "seed_answer": seed_answer,
                "gold_bridges": gold_bridges,
                "initial_passage_previews": passages_to_previews(initial_passages),
                "arms": arm_rows,
            }
        )

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "summary": summary,
                    "results": results,
                    "timing": {"elapsed_sec": round(time.time() - t0, 2)},
                },
                f,
                indent=2,
            )

    print("\nSummary", flush=True)
    for arm in args.arms:
        stats = summary[arm]
        print(
            f"{arm:14s} success={stats['successes']} contain={stats['contain_hits']} em={stats['em_hits']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
