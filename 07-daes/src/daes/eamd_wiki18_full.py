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
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from dgmqr import extract_candidates

MODEL_REF = None
TOKENIZER_REF = None

CORPUS_JSONL = "/projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl"
ID_OFFSET_JSON = "/projects/prjs1800/msc-thesis/01-arag-reproduction/data/index/wiki18_id_offset.json"
FAISS_INDEX = "/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index"
QUESTIONS_DIR = "/projects/prjs1800/msc-thesis/01-arag-reproduction/data/questions_wiki18"
QUESTION_FILES = {
    "hotpotqa": f"{QUESTIONS_DIR}/hotpotqa.json",
    "musique": f"{QUESTIONS_DIR}/musique.json",
    "2wikimultihopqa": f"{QUESTIONS_DIR}/2wikimultihopqa.json",
}
DEFAULT_OUTPUT_ROOT = "/projects/prjs1800/msc-thesis/07-daes/results/eamd_wiki18_full"
DEFAULT_METHODS = ["baseline", "spread", "aram", "pool", "eamd_regen", "eamd_remask"]

SHORT_INSTRUCTIONS = """You are a helpful assistant.
Answer the question using the context when possible.
Give a direct concise answer in 1 to 6 words.
Do not explain.
Do not write a sentence if a short phrase is enough.
"""


class WikiCorpusStore:
    def __init__(self, corpus_jsonl: str, id_offset_json: str):
        self.corpus_jsonl = Path(corpus_jsonl)
        self.id_offset_json = Path(id_offset_json)
        self.id2offset = json.loads(self.id_offset_json.read_text(encoding="utf-8"))

    def get_chunk(self, chunk_id: str) -> str:
        cid = str(chunk_id)
        with self.corpus_jsonl.open("rb") as f:
            f.seek(int(self.id2offset[cid]))
            line = f.readline().decode("utf-8")
        row = json.loads(line)
        return row["contents"]


class Wiki18Retriever:
    def __init__(self, embedding_model: str = "intfloat/e5-base-v2", device: str = "cuda:0",
                 encode_batch_size: int = 64, num_threads: int = 16):
        faiss.omp_set_num_threads(num_threads)
        self.index = faiss.read_index(FAISS_INDEX)
        self.store = WikiCorpusStore(CORPUS_JSONL, ID_OFFSET_JSON)
        self.model = SentenceTransformer(embedding_model, device=device)
        self.encode_batch_size = encode_batch_size

    def retrieve_batch(self, queries: list[str], top_k: int) -> list[list[str]]:
        vecs = self.model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=min(self.encode_batch_size, max(1, len(queries))),
        ).astype(np.float32)
        scores, idxs = self.index.search(vecs, top_k)
        all_results = []
        for row in idxs:
            passages = []
            seen = set()
            for idx in row.tolist():
                if idx < 0:
                    continue
                cid = str(int(idx))
                if cid in seen:
                    continue
                seen.add(cid)
                passages.append(self.store.get_chunk(cid))
            all_results.append(passages)
        return all_results

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return self.retrieve_batch([query], top_k=top_k)[0]


def chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


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


def compute_em(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def short_user_prompt(context: str, question: str) -> str:
    return (
        f"{SHORT_INSTRUCTIONS}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_short_prompt(tokenizer, context: str, question: str) -> tuple[list[int], int]:
    prompt = short_user_prompt(context, question)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    return prefix_ids, len(prefix_ids)


def build_short_cond_and_prior(tokenizer, context: str, question: str, n_tokens: int) -> tuple[list[int], list[int], int]:
    mask_id = tokenizer.mask_token_id

    prompt_cond = short_user_prompt(context, question)
    prompt_prior = short_user_prompt("", question)

    msg_cond = [{"role": "user", "content": prompt_cond}]
    msg_prior = [{"role": "user", "content": prompt_prior}]

    text_cond = tokenizer.apply_chat_template(msg_cond, tokenize=False, add_generation_prompt=True)
    text_prior = tokenizer.apply_chat_template(msg_prior, tokenize=False, add_generation_prompt=True)

    prefix_cond = tokenizer.encode(text_cond, add_special_tokens=False)
    prefix_prior = tokenizer.encode(text_prior, add_special_tokens=False)

    min_len = min(len(prefix_cond), len(prefix_prior))
    ctx_start = min_len
    for i in range(min_len):
        if prefix_cond[i] != prefix_prior[i]:
            ctx_start = i
            break

    n_ctx = max(0, len(prefix_cond) - len(prefix_prior))
    ctx_end = min(len(prefix_cond), ctx_start + n_ctx)

    prior_prefix = list(prefix_cond)
    for i in range(ctx_start, ctx_end):
        prior_prefix[i] = mask_id

    cond_ids = prefix_cond + [mask_id] * n_tokens
    prior_ids = prior_prefix + [mask_id] * n_tokens
    return cond_ids, prior_ids, len(prefix_cond)


def shifted_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def decode_answer(tokenizer, answer_tokens: torch.Tensor) -> str:
    return tokenizer.decode(answer_tokens.tolist(), skip_special_tokens=True).strip()


def content_positions(answer_tokens: torch.Tensor, eos_id: int) -> list[int]:
    tokens = answer_tokens.tolist()
    if eos_id in tokens:
        stop = tokens.index(eos_id)
    else:
        stop = len(tokens)
    return [i for i in range(stop) if tokens[i] != eos_id]


@torch.inference_mode()
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
        out = model(x, attention_mask=attn)
        logits = shifted_logits(out.logits)
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
    answer_text = decode_answer(tokenizer, answer_tokens)
    avg_conf = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
    return answer_text, answer_tokens, avg_conf


@torch.inference_mode()
def spread_generate_shared(model, tokenizer, context: str, question: str, steps: int = 16,
                           n_tokens: int = 16, temperature: float = 0.1, alpha: float = 0.5):
    device = model.device
    mask_id = tokenizer.mask_token_id
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    q_ids = tokenizer.encode(question, return_tensors="pt").to(device)
    q_out = model(q_ids, output_hidden_states=True)
    h_q = F.normalize(q_out.hidden_states[-1].mean(dim=1), dim=-1)

    k_per_step = max(1, math.ceil(n_tokens / steps))
    remaining = n_tokens
    score_stds = []

    for step in range(steps):
        if remaining <= 0:
            break
        mask_idx = (x[0] == mask_id)
        mask_idx[:n_prefix] = False
        if not mask_idx.any():
            break

        mask_pos = mask_idx.nonzero(as_tuple=True)[0]
        out = model(x, attention_mask=attn, output_hidden_states=True)
        logits = shifted_logits(out.logits)
        hs = out.hidden_states[-1]

        mask_logits = logits[0, mask_pos]
        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=True)

        h_masked = F.normalize(hs[0, mask_pos], dim=-1)
        rel = torch.sigmoid(h_masked @ h_q.squeeze(0))

        if len(mask_pos) > 1:
            conf_min, conf_max = confidence.min(), confidence.max()
            rel_min, rel_max = rel.min(), rel.max()
            conf_norm = (confidence - conf_min) / (conf_max - conf_min) if conf_max > conf_min else torch.ones_like(confidence)
            rel_norm = (rel - rel_min) / (rel_max - rel_min) if rel_max > rel_min else torch.ones_like(rel)
            score = alpha * rel_norm + (1.0 - alpha) * conf_norm
            score_stds.append(score.std(unbiased=False).item())
        else:
            score = torch.ones_like(confidence)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(score, min(n_commit, len(score)))
        x[0, mask_pos[topk]] = x0[topk]
        remaining -= len(topk)

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), {
        "mean_score_std": sum(score_stds) / len(score_stds) if score_stds else 0.0,
    }


@torch.inference_mode()
def aram_generate_shared(model, tokenizer, context: str, question: str, steps: int = 16,
                         n_tokens: int = 16, temperature: float = 0.1,
                         lambda_max: float = 1.0, beta: float = 0.5, eps: float = 1e-6):
    device = model.device
    mask_id = tokenizer.mask_token_id
    cond_ids, prior_ids, n_prefix = build_short_cond_and_prior(tokenizer, context, question, n_tokens)

    x = torch.tensor([cond_ids], dtype=torch.long, device=device)
    x_prior = torch.tensor([prior_ids], dtype=torch.long, device=device)
    attn = torch.ones((1, len(cond_ids)), dtype=torch.long, device=device)
    attn_prior = torch.ones((1, len(cond_ids)), dtype=torch.long, device=device)

    k_per_step = max(1, math.ceil(n_tokens / steps))
    remaining = n_tokens
    lambda_traj = []

    for step in range(steps):
        if remaining <= 0:
            break
        mask_idx = (x[0] == mask_id)
        mask_idx[:n_prefix] = False
        if not mask_idx.any():
            break

        mask_pos = mask_idx.nonzero(as_tuple=True)[0]
        x_prior[0, n_prefix:] = x[0, n_prefix:]
        x_batch = torch.cat([x, x_prior], dim=0)
        attn_batch = torch.cat([attn, attn_prior], dim=0)

        out = model(x_batch, attention_mask=attn_batch)

        logits_all = shifted_logits(out.logits)
        logits_cond = logits_all[0, mask_pos]
        logits_prior = logits_all[1, mask_pos]

        log_p_cond = F.log_softmax(logits_cond, dim=-1)
        log_p_prior = F.log_softmax(logits_prior, dim=-1)
        p_cond = log_p_cond.exp()
        p_prior = log_p_prior.exp()

        signal = (p_cond * (log_p_cond - log_p_prior)).sum(dim=-1) + (p_prior * (log_p_prior - log_p_cond)).sum(dim=-1)
        noise = -(p_cond * log_p_cond).sum(dim=-1)
        lam = lambda_max * torch.tanh(beta * signal / (noise + eps))
        guided_logits = logits_prior + lam.unsqueeze(-1) * (logits_cond - logits_prior)

        confidence, x0 = sample_tokens(guided_logits, temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, len(confidence)))
        x[0, mask_pos[topk]] = x0[topk]
        remaining -= len(topk)
        lambda_traj.append(lam.mean().item())

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), {
        "mean_lambda": sum(lambda_traj) / len(lambda_traj) if lambda_traj else 0.0,
    }


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


def expand_evidence(retriever: Wiki18Retriever, question: str, initial_context: str, initial_passages: list[str],
                    init_answer: str, n_candidates: int = 3, expand_top_k: int = 3) -> tuple[list[str], list[dict]]:
    candidates = extract_candidates(MODEL_REF, TOKENIZER_REF, initial_context, question, n_candidates)
    all_passages = list(initial_passages)
    queries = []
    if init_answer and len(init_answer.strip()) > 2:
        queries.append(f"{question} {init_answer[:100]}")
    queries.extend(f"{question} {cand['text']}" for cand in candidates)
    if queries:
        for hits in retriever.retrieve_batch(queries, top_k=expand_top_k):
            all_passages.extend(hits)

    return unique_passages(all_passages), candidates


@torch.inference_mode()
def eamd_regen_shared(model, tokenizer, question: str, old_context: str, new_context: str,
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
    scale_means = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, new_n_prefix:] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + new_n_prefix
        base_pos = masked_local + old_n_prefix

        out_full = model(x_full, attention_mask=attn_full)
        out_base = model(x_base, attention_mask=attn_base)

        logits_full = shifted_logits(out_full.logits)[0, full_pos]
        logits_base = shifted_logits(out_base.logits)[0, base_pos]

        log_p_full = F.log_softmax(logits_full, dim=-1)
        log_p_base = F.log_softmax(logits_base, dim=-1)
        p_full = log_p_full.exp()
        p_base = log_p_base.exp()

        signal = (p_full * (log_p_full - log_p_base)).sum(dim=-1) + (p_base * (log_p_base - log_p_full)).sum(dim=-1)
        noise = -(p_full * log_p_full).sum(dim=-1)
        schedule = float(step + 1) / float(steps)
        extra_scale = lambda_max * torch.tanh(beta * signal / (noise + eps)) * schedule
        guidance_scale = 1.0 + extra_scale
        guided_logits = logits_full + extra_scale.unsqueeze(-1) * (logits_full - logits_base)

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
        scale_means.append(guidance_scale.mean().item())
        remaining -= len(topk)

    answer_tokens = x_full[0, new_n_prefix:new_n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), {
        "mean_signal": sum(signal_means) / len(signal_means) if signal_means else 0.0,
        "mean_guidance_scale": sum(scale_means) / len(scale_means) if scale_means else 1.0,
        "avg_conf": sum(token_confidences) / len(token_confidences) if token_confidences else 0.0,
    }


def remask_span_positions(divergence: torch.Tensor, positions: list[int], committed_tokens: list[int],
                         predicted_tokens: list[int], delta: float) -> list[int]:
    if not positions:
        return []
    flagged = [
        positions[i]
        for i in range(len(positions))
        if predicted_tokens[i] != committed_tokens[i] or divergence[i].item() > delta
    ]
    if not flagged:
        return []
    return list(range(min(positions), max(positions) + 1))


@torch.inference_mode()
def eamd_remask_shared(model, tokenizer, question: str, old_context: str, new_context: str,
                       seed_tokens: torch.Tensor, steps: int = 16, temperature: float = 0.1,
                       lambda_max: float = 1.0, beta: float = 0.5, eps: float = 1e-6,
                       delta: float = 0.05):
    device = model.device
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id
    n_tokens = len(seed_tokens)

    old_prefix_ids, old_n_prefix = build_short_prompt(tokenizer, old_context, question)
    new_prefix_ids, new_n_prefix = build_short_prompt(tokenizer, new_context, question)

    x_base = torch.tensor([old_prefix_ids + seed_tokens.tolist()], dtype=torch.long, device=device)
    x_full = torch.tensor([new_prefix_ids + seed_tokens.tolist()], dtype=torch.long, device=device)
    attn_base = torch.ones((1, x_base.shape[1]), dtype=torch.long, device=device)
    attn_full = torch.ones((1, x_full.shape[1]), dtype=torch.long, device=device)

    positions = content_positions(seed_tokens, eos_id)
    if not positions:
        return decode_answer(tokenizer, seed_tokens), {
            "remasked_positions": [],
            "mean_signal": 0.0,
            "mean_guidance_scale": 1.0,
            "avg_conf": 0.0,
        }

    old_pos = torch.tensor([old_n_prefix + pos for pos in positions], dtype=torch.long, device=device)
    new_pos = torch.tensor([new_n_prefix + pos for pos in positions], dtype=torch.long, device=device)

    out_old = model(x_base, attention_mask=attn_base)
    out_new = model(x_full, attention_mask=attn_full)

    logits_old = shifted_logits(out_old.logits)[0, old_pos]
    logits_new = shifted_logits(out_new.logits)[0, new_pos]
    log_p_old = F.log_softmax(logits_old, dim=-1)
    log_p_new = F.log_softmax(logits_new, dim=-1)
    p_old = log_p_old.exp()
    p_new = log_p_new.exp()
    divergence = (p_new * (log_p_new - log_p_old)).sum(dim=-1) + (p_old * (log_p_old - log_p_new)).sum(dim=-1)
    committed = [seed_tokens[pos].item() for pos in positions]
    predicted = torch.argmax(logits_new, dim=-1).tolist()

    remask_positions = remask_span_positions(divergence, positions, committed, predicted, delta)
    if not remask_positions:
        return decode_answer(tokenizer, seed_tokens), {
            "remasked_positions": [],
            "mean_signal": divergence.mean().item(),
            "mean_guidance_scale": 1.0,
            "avg_conf": 0.0,
            "top1_changes": sum(int(p != c) for p, c in zip(predicted, committed)),
        }

    x_base[0, [old_n_prefix + pos for pos in remask_positions]] = mask_id
    x_full[0, [new_n_prefix + pos for pos in remask_positions]] = mask_id

    remaining = len(remask_positions)
    k_per_step = max(1, math.ceil(remaining / steps))
    token_confidences = []
    signal_means = []
    scale_means = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, new_n_prefix:] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + new_n_prefix
        base_pos = masked_local + old_n_prefix

        out_full = model(x_full, attention_mask=attn_full)
        out_base = model(x_base, attention_mask=attn_base)

        logits_full = shifted_logits(out_full.logits)[0, full_pos]
        logits_base = shifted_logits(out_base.logits)[0, base_pos]

        log_p_full = F.log_softmax(logits_full, dim=-1)
        log_p_base = F.log_softmax(logits_base, dim=-1)
        p_full = log_p_full.exp()
        p_base = log_p_base.exp()

        signal = (p_full * (log_p_full - log_p_base)).sum(dim=-1) + (p_base * (log_p_base - log_p_full)).sum(dim=-1)
        noise = -(p_full * log_p_full).sum(dim=-1)
        schedule = float(step + 1) / float(steps)
        extra_scale = lambda_max * torch.tanh(beta * signal / (noise + eps)) * schedule
        guidance_scale = 1.0 + extra_scale
        guided_logits = logits_full + extra_scale.unsqueeze(-1) * (logits_full - logits_base)

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
        scale_means.append(guidance_scale.mean().item())
        remaining -= len(topk)

    answer_tokens = x_full[0, new_n_prefix:new_n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), {
        "remasked_positions": remask_positions,
        "mean_signal": sum(signal_means) / len(signal_means) if signal_means else divergence.mean().item(),
        "mean_guidance_scale": sum(scale_means) / len(scale_means) if scale_means else 1.0,
        "avg_conf": sum(token_confidences) / len(token_confidences) if token_confidences else 0.0,
        "initial_divergence": divergence.mean().item(),
        "top1_changes": sum(int(p != c) for p, c in zip(predicted, committed)),
    }


def evaluate(pred: str, gold: str) -> dict:
    precision, recall, f1 = compute_f1(pred, gold)
    return {
        "pred": pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": compute_em(pred, gold),
        "contain": normalize_answer(gold) in normalize_answer(pred),
    }


def resolve_questions_file(dataset: str, explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    if dataset not in QUESTION_FILES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return QUESTION_FILES[dataset]


def parse_methods(methods_raw: str) -> list[str]:
    methods = [item.strip() for item in methods_raw.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    return methods


def should_log(idx: int, total: int, log_every: int) -> bool:
    if idx <= 3 or idx == total:
        return True
    return log_every > 0 and idx % log_every == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="musique", choices=sorted(QUESTION_FILES))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--questions_file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--methods", default="baseline,spread,aram,pool,eamd_regen,eamd_remask")
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--retriever_encode_batch_size", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    retriever = Wiki18Retriever(
        device="cuda:0",
        encode_batch_size=args.retriever_encode_batch_size,
        num_threads=max(1, int(os.environ.get("OMP_NUM_THREADS", "16"))),
    )

    @dataclass
    class MA:
        model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"

    model = dllm.utils.get_model(model_args=MA()).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=MA())

    global MODEL_REF, TOKENIZER_REF
    MODEL_REF = model
    TOKENIZER_REF = tokenizer

    questions_file = resolve_questions_file(args.dataset, args.questions_file)
    all_questions = json.load(open(questions_file))
    end_idx = args.end_idx if args.end_idx is not None else args.start_idx + args.n_questions
    questions = all_questions[args.start_idx:end_idx]

    methods = parse_methods(args.methods)
    totals = {name: {"f1": 0.0, "em": 0.0, "contain": 0.0} for name in methods}
    results = []

    if not questions:
        raise ValueError("No questions selected for this shard.")

    question_texts = [q["question"] for q in questions]
    initial_batches = []
    for query_batch in chunked(question_texts, args.retrieval_batch_size):
        initial_batches.extend(retriever.retrieve_batch(query_batch, top_k=args.initial_top_k))

    total_elapsed = 0.0
    for idx, q in enumerate(questions, start=1):
        qid = q.get("qid") or q["id"]
        question = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        log_this = should_log(idx, len(questions), args.log_every)
        if log_this:
            print(f"[{idx}/{len(questions)}] {qid}", flush=True)

        initial_passages = initial_batches[idx - 1]
        old_context = "\n\n".join(initial_passages)

        t0 = time.time()

        baseline_answer, baseline_tokens, baseline_conf = short_generate(model, tokenizer, old_context, question)
        baseline = evaluate(baseline_answer, gold)
        baseline["avg_conf"] = baseline_conf
        if log_this and "baseline" in methods:
            print(f"  baseline:     {baseline_answer} | F1={baseline['f1']:.3f}", flush=True)

        need_new_context = any(name in methods for name in ("pool", "eamd_regen", "eamd_remask"))
        expanded_passages, candidates = (initial_passages, [])
        if need_new_context:
            expanded_passages, candidates = expand_evidence(
                retriever, question, old_context, initial_passages, baseline_answer, 3, args.expand_top_k
            )
        new_context = "\n\n".join(expanded_passages)
        candidate_texts = [cand["text"] for cand in candidates]

        row = {
            "id": qid,
            "question": question,
            "gold": gold,
            "candidates": candidate_texts,
            "n_passages_old": len(initial_passages),
            "n_passages_new": len(expanded_passages),
            "baseline": baseline,
            "elapsed_sec": 0.0,
        }

        if "spread" in methods:
            spread_answer, spread_stats = spread_generate_shared(model, tokenizer, old_context, question)
            spread = evaluate(spread_answer, gold)
            spread["stats"] = spread_stats
            row["spread"] = spread
            if log_this:
                print(f"  spread:       {spread_answer} | F1={spread['f1']:.3f}", flush=True)

        if "aram" in methods:
            aram_answer, aram_stats = aram_generate_shared(model, tokenizer, old_context, question)
            aram = evaluate(aram_answer, gold)
            aram["stats"] = aram_stats
            row["aram"] = aram
            if log_this:
                print(f"  aram:         {aram_answer} | F1={aram['f1']:.3f}", flush=True)

        if "pool" in methods:
            pool_answer, _, pool_conf = short_generate(model, tokenizer, new_context, question)
            pool = evaluate(pool_answer, gold)
            pool["avg_conf"] = pool_conf
            row["pool"] = pool
            if log_this:
                print(f"  pool_short:   {pool_answer} | F1={pool['f1']:.3f}", flush=True)

        if "eamd_regen" in methods:
            eamd_regen_answer, eamd_regen_stats = eamd_regen_shared(model, tokenizer, question, old_context, new_context)
            eamd_regen = evaluate(eamd_regen_answer, gold)
            eamd_regen["stats"] = eamd_regen_stats
            row["eamd_regen"] = eamd_regen
            if log_this:
                print(f"  eamd_regen:   {eamd_regen_answer} | F1={eamd_regen['f1']:.3f}", flush=True)

        if "eamd_remask" in methods:
            eamd_answer, eamd_stats = eamd_remask_shared(
                model, tokenizer, question, old_context, new_context, baseline_tokens
            )
            eamd = evaluate(eamd_answer, gold)
            eamd["stats"] = eamd_stats
            row["eamd_remask"] = eamd
            if log_this:
                print(f"  eamd_remask:  {eamd_answer} | F1={eamd['f1']:.3f}", flush=True)
        elapsed = round(time.time() - t0, 2)
        row["elapsed_sec"] = elapsed
        total_elapsed += elapsed
        results.append(row)
        for key in methods:
            totals[key]["f1"] += row[key]["f1"]
            totals[key]["em"] += row[key]["em"]
            totals[key]["contain"] += float(row[key]["contain"])

    summary = {
        key: {
            "f1": totals[key]["f1"] / len(results),
            "em": totals[key]["em"] / len(results),
            "contain": totals[key]["contain"] / len(results),
        }
        for key in methods
    }

    output_path = args.output
    if output_path is None:
        output_path = (
            f"{DEFAULT_OUTPUT_ROOT}/{args.dataset}/shards/"
            f"eamd_wiki18_{args.dataset}_{args.start_idx}_{end_idx}.json"
        )
    payload = {
        "metadata": {
            "dataset": args.dataset,
            "questions_file": questions_file,
            "start_idx": args.start_idx,
            "end_idx": end_idx,
            "n_questions": len(results),
            "methods": methods,
            "initial_top_k": args.initial_top_k,
            "expand_top_k": args.expand_top_k,
            "total_elapsed_sec": round(total_elapsed, 2),
            "avg_elapsed_sec": round(total_elapsed / len(results), 4),
        },
        "summary": summary,
        "results": results,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f)

    print("\nSummary", flush=True)
    for key, values in summary.items():
        print(f"  {key:12s} F1={values['f1']:.3f} EM={values['em']:.3f} contain={values['contain']:.3f}", flush=True)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
