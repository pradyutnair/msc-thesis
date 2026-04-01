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
from types import SimpleNamespace

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import dllm
from dllm.pipelines.dream.sampler import sample_tokens
from dgmqr import extract_candidates as extract_candidates_dream

MODEL_REF = None
TOKENIZER_REF = None
MODEL_TYPE_REF = "dream"

def _neg_entropy():
    """Return neg_entropy flag based on model type. Dream=True (entropy-based), LLaDA=False (low-confidence)."""
    return MODEL_TYPE_REF != "llada"

CORPUS_JSONL = "/projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl"
ID_OFFSET_JSON = "/projects/prjs1800/msc-thesis/01-arag-reproduction/data/index/wiki18_id_offset.json"
FAISS_INDEX = "/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index"
QUESTIONS_DIR = "/projects/prjs1800/msc-thesis/01-arag-reproduction/data/questions_wiki18"
QUESTION_FILES = {
    "hotpotqa": f"{QUESTIONS_DIR}/hotpotqa.json",
    "musique": f"{QUESTIONS_DIR}/musique.json",
    "2wikimultihopqa": f"{QUESTIONS_DIR}/2wikimultihopqa.json",
}
DEFAULT_OUTPUT_ROOT = "/projects/prjs1800/msc-thesis/07-daes/results/eamd_v2_wiki18"
DEFAULT_METHODS = ["baseline", "spread", "aram", "pool", "eamd_v2_regen", "eamd_v2_remask"]

SHORT_INSTRUCTIONS = """You are a helpful assistant.
Answer the question using the context when possible.
Give a direct concise answer in 1 to 6 words.
Do not explain.
Do not write a sentence if a short phrase is enough.
"""

LLADA_SHORT_INSTRUCTIONS = """Use the following passages as context and provide a SHORT and DIRECT answer.
RULES:
- Use ONLY the context passages to answer
- Answer must be 1-6 words maximum
- Do NOT include any explanations or extra text
- Do NOT repeat or summarize the passages

Example:
Question: Where is the Eiffel Tower?
Short Answer: Paris, France

Example:
Question: Who wrote Hamlet?
Short Answer: William Shakespeare

Example:
Question: What year did World War 2 end?
Short Answer: 1945
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
    instructions = LLADA_SHORT_INSTRUCTIONS if MODEL_TYPE_REF == "llada" else SHORT_INSTRUCTIONS
    suffix = "Short Answer:" if MODEL_TYPE_REF == "llada" else "Answer:"
    return (
        f"{instructions}\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"{suffix}"
    )


def build_short_prompt(tokenizer, context: str, question: str) -> tuple[list[int], int]:
    prompt = short_user_prompt(context, question)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    return prefix_ids, len(prefix_ids)


def build_short_pair(tokenizer, full_context: str, base_context: str, question: str, n_tokens: int) -> tuple[list[int], list[int], int]:
    mask_id = get_mask_id(tokenizer)

    prompt_full = short_user_prompt(full_context, question)
    prompt_base = short_user_prompt(base_context, question)

    msg_full = [{"role": "user", "content": prompt_full}]
    msg_base = [{"role": "user", "content": prompt_base}]

    text_full = tokenizer.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=True)
    text_base = tokenizer.apply_chat_template(msg_base, tokenize=False, add_generation_prompt=True)

    prefix_full = tokenizer.encode(text_full, add_special_tokens=False)
    prefix_base = tokenizer.encode(text_base, add_special_tokens=False)

    min_len = min(len(prefix_full), len(prefix_base))
    diff_start = min_len
    for i in range(min_len):
        if prefix_full[i] != prefix_base[i]:
            diff_start = i
            break

    diff_len = max(0, len(prefix_full) - len(prefix_base))
    diff_end = min(len(prefix_full), diff_start + diff_len)

    masked_base_prefix = list(prefix_full)
    for i in range(diff_start, diff_end):
        masked_base_prefix[i] = mask_id

    full_ids = prefix_full + [mask_id] * n_tokens
    base_ids = masked_base_prefix + [mask_id] * n_tokens
    return full_ids, base_ids, len(prefix_full)


def build_short_cond_and_prior(tokenizer, context: str, question: str, n_tokens: int) -> tuple[list[int], list[int], int]:
    return build_short_pair(tokenizer, context, "", question, n_tokens)


def shifted_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def prepare_logits(logits: torch.Tensor) -> torch.Tensor:
    if MODEL_TYPE_REF == "dream":
        return shifted_logits(logits)
    return logits


def get_mask_id(tokenizer) -> int:
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    if MODEL_TYPE_REF == "llada":
        return 126336
    raise ValueError("Tokenizer has no mask_token_id")


def decode_answer(tokenizer, answer_tokens: torch.Tensor) -> str:
    return tokenizer.decode(answer_tokens.tolist(), skip_special_tokens=True).strip()


def content_positions(answer_tokens: torch.Tensor, eos_id: int) -> list[int]:
    tokens = answer_tokens.tolist()
    if eos_id in tokens:
        stop = tokens.index(eos_id)
    else:
        stop = len(tokens)
    return [i for i in range(stop) if tokens[i] != eos_id]


def compute_signal_and_scale(
    logits_full: torch.Tensor,
    logits_base: torch.Tensor,
    lambda_max: float,
    beta: float,
    eps: float,
    schedule: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_p_full = F.log_softmax(logits_full, dim=-1)
    log_p_base = F.log_softmax(logits_base, dim=-1)
    p_full = log_p_full.exp()
    p_base = log_p_base.exp()

    signal = (p_full * (log_p_full - log_p_base)).sum(dim=-1) + (p_base * (log_p_base - log_p_full)).sum(dim=-1)
    noise = -(p_full * log_p_full).sum(dim=-1)
    extra_scale = lambda_max * torch.tanh(beta * signal / (noise + eps)) * schedule
    guidance_scale = 1.0 + extra_scale
    return signal, noise, extra_scale, guidance_scale


def compute_v2_guidance(
    logits_full: torch.Tensor,
    logits_base: torch.Tensor,
    w_t: float,
    eps: float = 1e-6,
    gamma_cap: float | None = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    log_p_full = F.log_softmax(logits_full, dim=-1)
    log_p_base = F.log_softmax(logits_base, dim=-1)
    p_full = log_p_full.exp()
    r = log_p_full - log_p_base
    ig = (p_full * r).sum(dim=-1)
    var = (p_full * (r - ig.unsqueeze(-1)).pow(2)).sum(dim=-1)
    gamma = w_t * ig / var.clamp_min(eps)
    gamma = torch.clamp(gamma, min=0.0)
    if gamma_cap is not None and gamma_cap > 0:
        gamma = torch.clamp(gamma, max=gamma_cap)
    guidance_scale = 1.0 + gamma
    return ig, var, gamma, guidance_scale


def compute_v2_remask_probabilities(
    logits_old: torch.Tensor,
    logits_new: torch.Tensor,
    remask_prior: float,
    remask_cost: float,
    tau: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_p_old = F.log_softmax(logits_old, dim=-1)
    log_p_new = F.log_softmax(logits_new, dim=-1)
    p_old = log_p_old.exp()
    delta_refresh = (p_old * (log_p_old - log_p_new)).sum(dim=-1)

    prior = torch.full_like(delta_refresh, fill_value=float(remask_prior))
    prior = prior.clamp(min=eps, max=1.0 - eps)
    logit_prior = torch.log(prior) - torch.log1p(-prior)
    rho = torch.sigmoid(logit_prior + (delta_refresh - remask_cost) / max(tau, eps))
    return delta_refresh, rho


def compute_w_t(n_masked: int, n_tokens: int) -> tuple[float, float]:
    if n_tokens <= 0:
        return 0.0, 1.0
    mu_t = float(n_masked) / float(max(1, n_tokens))
    w_t = max(0.0, 1.0 - mu_t)
    return mu_t, w_t


def topm_mean(values: torch.Tensor, m: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_tensor(0.0)
    k = min(max(1, m), values.numel())
    topk = torch.topk(values, k).values
    return topk.mean()


@torch.inference_mode()
def short_generate(model, tokenizer, context: str, question: str, steps: int = 16,
                   n_tokens: int = 16, temperature: float = 0.1):
    device = model.device
    mask_id = get_mask_id(tokenizer)
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
        logits = prepare_logits(out.logits)
        mask_pos = mask_idx[0].nonzero(as_tuple=True)[0]
        conf, x0 = sample_tokens(logits[0, mask_pos], temperature=temperature, neg_entropy=_neg_entropy())
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
def extract_candidates_generic(model, tokenizer, context: str, question: str, n_candidates: int = 3, extraction_steps: int = 12):
    # Use model-agnostic multi-position extractor for all models
    return extract_candidates_agnostic(model, tokenizer, context, question, n_candidates, extraction_steps=extraction_steps)

    device = model.device
    mask_id = get_mask_id(tokenizer)
    prompt = f"{context}\n\nQuestion: {question}\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)

    n_mask = 20
    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    out = model(x, attention_mask=attn)
    logits = prepare_logits(out.logits)
    probs = torch.softmax(logits[0, n_prefix] / 0.3, dim=-1)
    top_probs, top_ids = torch.topk(probs, n_candidates * 3)

    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        x_c = torch.tensor([canvas], dtype=torch.long, device=device)
        x_c[0, n_prefix] = tid
        rem = n_mask - 1
        for step in range(16):
            if rem <= 0:
                break
            mi = x_c == mask_id
            if not mi.any():
                break
            o2 = model(x_c, attention_mask=attn)
            l2 = prepare_logits(o2.logits)
            mp = mi[0].nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=_neg_entropy())
            k = min(max(1, rem // 16), rem)
            if step == 15:
                k = rem
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            rem -= len(tk)

        cand_text = tokenizer.decode(x_c[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
        cand_text = cand_text.split("\n")[0].split(". ")[0].strip()
        # For LLaDA: aggressively extract entity from verbose candidates
        if MODEL_TYPE_REF == "llada":
            import re
            # Strip common preambles
            cand_text = re.sub(r"^(?:The answer is:?|The)\s+", "", cand_text).strip()
            # Strip "X is/was/of/that Y" -> keep Y (the actual entity)
            m = re.match(r"^.*?(?:is|was|are|were)\s+(.+)", cand_text)
            if m and len(m.group(1).split()) <= 6:
                cand_text = m.group(1).strip().rstrip(",.")
            # Also try: strip everything before last proper noun sequence
            # Truncate to max 4 words for cleaner retrieval queries
            words = cand_text.split()
            if len(words) > 4:
                cand_text = " ".join(words[:4])
            cand_text = cand_text.strip().rstrip(",.")

        if cand_text and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append({"text": cand_text, "init_conf": prob.item()})
            if len(candidates) >= n_candidates:
                break
    return candidates


@torch.inference_mode()
def extract_candidates_agnostic(model, tokenizer, context: str, question: str,
                                n_candidates: int = 3, n_positions: int = 3,
                                n_branch: int = 2, n_mask: int = 12,
                                extraction_steps: int = 12):
    """Model-agnostic bridge candidate extraction via multi-position posterior sampling.

    Batched implementation: all branch canvases are denoised in parallel.

    Args:
        extraction_steps: Number of denoising steps per branch rollout (default 12).
            Can be reduced (e.g. 4) for speed at slight quality cost.
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)

    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)

    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    # Step 1: Forward pass on fully masked canvas
    out = model(x, attention_mask=attn)
    logits = prepare_logits(out.logits)

    # Step 2: Select positions — ALWAYS include position 0 + top entropy positions
    answer_logits = logits[0, n_prefix:n_prefix + n_mask]
    probs = torch.softmax(answer_logits / 0.3, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    entropy_positions = torch.topk(entropy, min(n_positions, n_mask)).indices.tolist()
    top_positions = [0]
    for p in entropy_positions:
        if p not in top_positions:
            top_positions.append(p)
        if len(top_positions) >= n_positions + 1:
            break

    # Step 3: Build ALL branch canvases at once
    branch_canvases = []
    branch_meta = []  # (pos_local, prob, tid)
    for pos_local in top_positions:
        pos_global = n_prefix + pos_local
        pos_probs = torch.softmax(answer_logits[pos_local] / 0.3, dim=-1)
        top_probs, top_ids = torch.topk(pos_probs, n_branch)
        for i in range(len(top_probs)):
            c = list(canvas)
            c[pos_global] = top_ids[i].item()
            branch_canvases.append(c)
            branch_meta.append((pos_local, top_probs[i].item(), top_ids[i].item()))

    if not branch_canvases:
        return []

    # Step 4: Sequential denoising of each branch
    B = len(branch_canvases)
    x_all = torch.tensor(branch_canvases, dtype=torch.long, device=device)  # [B, seq_len]
    neg_ent = _neg_entropy()

    for bi in range(B):
        x_c = x_all[bi:bi+1]  # [1, seq_len]
        remaining = n_mask - 1
        for step in range(extraction_steps):
            if remaining <= 0:
                break
            mi = (x_c[0] == mask_id)
            if not mi.any():
                break
            out = model(x_c, attention_mask=attn)
            l2 = prepare_logits(out.logits)
            mp = mi.nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=neg_ent)
            k = min(max(1, remaining // extraction_steps), remaining)
            if step == extraction_steps - 1:
                k = remaining
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            remaining -= len(tk)

    # Step 5: Decode and deduplicate
    candidates = []
    seen = set()
    for bi in range(len(branch_meta)):
        pos_local, prob, tid = branch_meta[bi]
        cand_text = tokenizer.decode(x_all[bi, n_prefix:n_prefix + n_mask].tolist(),
                                     skip_special_tokens=True).strip()
        cand_text = cand_text.split("\n")[0].split(". ")[0].strip()

        if cand_text and len(cand_text) > 1 and cand_text.lower() not in seen:
            seen.add(cand_text.lower())
            candidates.append({"text": cand_text, "init_conf": prob,
                               "position": pos_local})
            if len(candidates) >= n_candidates:
                break

    return candidates



@torch.inference_mode()
def spread_generate_shared(model, tokenizer, context: str, question: str, steps: int = 16,
                           n_tokens: int = 16, temperature: float = 0.1, alpha: float = 0.5):
    device = model.device
    mask_id = get_mask_id(tokenizer)
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
        logits = prepare_logits(out.logits)
        hs = out.hidden_states[-1]

        mask_logits = logits[0, mask_pos]
        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, neg_entropy=_neg_entropy())

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
    mask_id = get_mask_id(tokenizer)
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

        logits_all = prepare_logits(out.logits)
        logits_cond = logits_all[0, mask_pos]
        logits_prior = logits_all[1, mask_pos]

        signal, _, lam, _ = compute_signal_and_scale(
            logits_cond,
            logits_prior,
            lambda_max=lambda_max,
            beta=beta,
            eps=eps,
        )
        guided_logits = logits_prior + lam.unsqueeze(-1) * (logits_cond - logits_prior)

        confidence, x0 = sample_tokens(guided_logits, temperature=temperature, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, len(confidence)))
        x[0, mask_pos[topk]] = x0[topk]
        remaining -= len(topk)
        lambda_traj.append(lam.mean().item())

    answer_tokens = x[0, n_prefix:n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), answer_tokens, {
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


def expand_positions_with_radius(positions: list[int], max_pos: int, radius: int) -> list[int]:
    if radius <= 0:
        return sorted(set(positions))
    expanded = set()
    for pos in positions:
        start = max(0, pos - radius)
        end = min(max_pos, pos + radius)
        expanded.update(range(start, end + 1))
    return sorted(expanded)


def expand_evidence(
    retriever: Wiki18Retriever,
    question: str,
    initial_context: str,
    initial_passages: list[str],
    seed_hypotheses: list[str],
    n_candidates: int = 3,
    expand_top_k: int = 3,
) -> tuple[list[str], list[dict]]:
    candidates = extract_candidates_generic(MODEL_REF, TOKENIZER_REF, initial_context, question, n_candidates) if n_candidates > 0 else []
    all_passages = list(initial_passages)
    queries = []
    seen_queries = set()
    for seed in seed_hypotheses:
        if not seed:
            continue
        seed = seed.strip()
        if len(seed) <= 2:
            continue
        query = f"{question} {seed[:100]}"
        if query not in seen_queries:
            seen_queries.add(query)
            queries.append(query)
    queries.extend(f"{question} {cand['text']}" for cand in candidates)
    if queries:
        for hits in retriever.retrieve_batch(queries, top_k=expand_top_k):
            all_passages.extend(hits)

    return unique_passages(all_passages), candidates


@torch.inference_mode()
def eamd_micro_shared(model, tokenizer, retriever: Wiki18Retriever, question: str,
                      initial_passages: list[str], steps: int = 8, n_tokens: int = 8,
                      temperature: float = 0.05, lambda_max: float = 1.0,
                      beta: float = 0.5, eps: float = 1e-6, expand_top_k: int = 3,
                      pivot_ratio: float = 0.5, top_m: int = 2, budget_min: int = 1,
                      kappa: float = 8.0, tau_q: float = 0.30, eta: float = 0.5,
                      neighbor_radius: int = 0, phase1_guidance: str = "baseline"):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    eos_id = tokenizer.eos_token_id

    old_context = "\n\n".join(initial_passages)
    if phase1_guidance == "aram":
        cond_ids, prior_ids, n_prefix0 = build_short_cond_and_prior(tokenizer, old_context, question, n_tokens)
        x0 = torch.tensor([cond_ids], dtype=torch.long, device=device)
        x0_prior = torch.tensor([prior_ids], dtype=torch.long, device=device)
        attn0 = torch.ones((1, len(cond_ids)), dtype=torch.long, device=device)
        attn0_prior = torch.ones((1, len(prior_ids)), dtype=torch.long, device=device)
    else:
        prefix_ids, n_prefix0 = build_short_prompt(tokenizer, old_context, question)
        canvas = prefix_ids + [mask_id] * n_tokens
        x0 = torch.tensor([canvas], dtype=torch.long, device=device)
        x0_prior = None
        attn0 = torch.ones((1, len(canvas)), dtype=torch.long, device=device)
        attn0_prior = None

    k_per_step = max(1, math.ceil(n_tokens / steps))
    pivot_steps = min(max(1, int(round(steps * pivot_ratio))), max(1, steps - 1))
    remaining = n_tokens
    phase1_conf = []

    for step in range(pivot_steps):
        if remaining <= 0:
            break
        mask_idx = (x0[0] == mask_id)
        mask_idx[:n_prefix0] = False
        if not mask_idx.any():
            break

        mask_pos = mask_idx.nonzero(as_tuple=True)[0]
        if phase1_guidance == "aram":
            x0_prior[0, n_prefix0:] = x0[0, n_prefix0:]
            x_batch = torch.cat([x0, x0_prior], dim=0)
            attn_batch = torch.cat([attn0, attn0_prior], dim=0)
            out = model(x_batch, attention_mask=attn_batch)
            logits_all = prepare_logits(out.logits)
            logits_full = logits_all[0, mask_pos]
            logits_base = logits_all[1, mask_pos]
            signal, _, extra_scale, _ = compute_signal_and_scale(
                logits_full,
                logits_base,
                lambda_max=lambda_max,
                beta=beta,
                eps=eps,
            )
            guided_logits = logits_base + extra_scale.unsqueeze(-1) * (logits_full - logits_base)
        else:
            out = model(x0, attention_mask=attn0)
            logits = prepare_logits(out.logits)
            guided_logits = logits[0, mask_pos]

        confidence, x_pred = sample_tokens(guided_logits, temperature=temperature, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == pivot_steps - 1 and steps == pivot_steps:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, len(confidence)))
        x0[0, mask_pos[topk]] = x_pred[topk]
        phase1_conf.extend(confidence[topk].tolist())
        remaining -= len(topk)

    mask_idx = (x0[0] == mask_id)
    mask_idx[:n_prefix0] = False
    out0 = model(x0, attention_mask=attn0)
    logits0 = prepare_logits(out0.logits)
    answer_state = x0[0, n_prefix0:n_prefix0 + n_tokens].clone()
    masked_local0 = (answer_state == mask_id).nonzero(as_tuple=True)[0]
    provisional_tokens = answer_state.clone()
    if len(masked_local0) > 0:
        provisional_tokens[masked_local0] = torch.argmax(logits0[0, masked_local0 + n_prefix0], dim=-1)
    provisional_answer = decode_answer(tokenizer, provisional_tokens)

    expanded_passages, _ = expand_evidence(
        retriever,
        question,
        old_context,
        initial_passages,
        [provisional_answer],
        n_candidates=0,
        expand_top_k=expand_top_k,
    )
    new_context = "\n\n".join(expanded_passages)

    full_ids, base_ids, n_prefix = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)
    x_base = torch.tensor([base_ids], dtype=torch.long, device=device)
    x_full = torch.tensor([full_ids], dtype=torch.long, device=device)
    x_base[0, n_prefix:n_prefix + n_tokens] = answer_state.to(device)
    x_full[0, n_prefix:n_prefix + n_tokens] = answer_state.to(device)
    attn_base = torch.ones((1, len(base_ids)), dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)

    answer_positions = list(range(n_tokens))
    committed_positions = [pos for pos in answer_positions if answer_state[pos].item() != mask_id]
    remask_positions = []
    g_q = 0.0
    ratio_mean = 0.0
    if committed_positions:
        pos_tensor = torch.tensor([n_prefix + pos for pos in committed_positions], dtype=torch.long, device=device)
        out_old = model(x_base, attention_mask=attn_base)
        out_new = model(x_full, attention_mask=attn_full)
        logits_old = prepare_logits(out_old.logits)[0, pos_tensor]
        logits_new = prepare_logits(out_new.logits)[0, pos_tensor]
        signal, noise, _, _ = compute_signal_and_scale(
            logits_new,
            logits_old,
            lambda_max=lambda_max,
            beta=beta,
            eps=eps,
        )
        ratio = signal / (noise + eps)
        ratio_mean = topm_mean(ratio, top_m).item()
        g_q = torch.sigmoid(torch.tensor(kappa * (ratio_mean - tau_q), device=device)).item()
        budget_floor = min(max(0, budget_min), len(committed_positions))
        budget = int(math.ceil(budget_floor + max(0, len(committed_positions) - budget_floor) * g_q))
        budget = min(max(0, budget), len(committed_positions))
        if budget > 0:
            top_idx = torch.topk(ratio, budget).indices.tolist()
            remask_positions = [committed_positions[i] for i in top_idx]
            remask_positions = expand_positions_with_radius(remask_positions, n_tokens - 1, neighbor_radius)
            x_base[0, [n_prefix + pos for pos in remask_positions]] = mask_id
            x_full[0, [n_prefix + pos for pos in remask_positions]] = mask_id

    remaining = int((x_full[0, n_prefix:n_prefix + n_tokens] == mask_id).sum().item())
    total_to_commit = max(1, remaining) if remaining > 0 else 0
    denom_steps = max(1, steps - pivot_steps)
    k_per_step_2 = max(1, math.ceil(total_to_commit / denom_steps)) if remaining > 0 else 0
    token_confidences = []
    signal_means = []
    scale_means = []

    for step in range(pivot_steps, steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + n_prefix
        base_pos = masked_local + n_prefix
        out_full = model(x_full, attention_mask=attn_full)
        out_base = model(x_base, attention_mask=attn_base)
        logits_full = prepare_logits(out_full.logits)[0, full_pos]
        logits_base = prepare_logits(out_base.logits)[0, base_pos]

        schedule = float(step + 1) / float(steps)
        signal, _, extra_scale, guidance_scale = compute_signal_and_scale(
            logits_full,
            logits_base,
            lambda_max=lambda_max,
            beta=beta,
            eps=eps,
            schedule=schedule,
        )
        extra_scale = torch.clamp(extra_scale * (1.0 + eta * g_q), max=lambda_max)
        guidance_scale = 1.0 + extra_scale
        guided_logits = logits_full + extra_scale.unsqueeze(-1) * (logits_full - logits_base)

        confidence, x_pred = sample_tokens(guided_logits, temperature=temperature, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step_2, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, len(confidence)))
        chosen_local = masked_local[topk]
        chosen_full = chosen_local + n_prefix
        chosen_base = chosen_local + n_prefix
        x_full[0, chosen_full] = x_pred[topk]
        x_base[0, chosen_base] = x_pred[topk]
        token_confidences.extend(confidence[topk].tolist())
        signal_means.append(signal.mean().item())
        scale_means.append(guidance_scale.mean().item())
        remaining -= len(topk)

    answer_tokens = x_full[0, n_prefix:n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), answer_tokens, {
        "pivot_steps": pivot_steps,
        "phase1_guidance": phase1_guidance,
        "provisional_answer": provisional_answer,
        "n_passages_old": len(initial_passages),
        "n_passages_new": len(expanded_passages),
        "remasked_positions": remask_positions,
        "g_q": g_q,
        "ratio_mean": ratio_mean,
        "mean_signal": sum(signal_means) / len(signal_means) if signal_means else 0.0,
        "mean_guidance_scale": sum(scale_means) / len(scale_means) if scale_means else 1.0,
        "avg_conf": sum(token_confidences) / len(token_confidences) if token_confidences else 0.0,
        "phase1_avg_conf": sum(phase1_conf) / len(phase1_conf) if phase1_conf else 0.0,
        "content_positions": content_positions(answer_tokens, eos_id),
    }


@torch.inference_mode()
def eamd_regen_shared(model, tokenizer, question: str, old_context: str, new_context: str,
                      steps: int = 16, n_tokens: int = 16, temperature: float = 0.1,
                      eps: float = 1e-6, gamma_cap: float = 8.0):
    device = model.device

    full_ids, base_ids, n_prefix = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)

    x_base = torch.tensor([base_ids], dtype=torch.long, device=device)
    x_full = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn_base = torch.ones((1, len(base_ids)), dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    token_confidences = []
    ig_means = []
    var_means = []
    gamma_means = []
    w_means = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, n_prefix:] == get_mask_id(tokenizer)).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + n_prefix
        base_pos = masked_local + n_prefix

        # Batched forward pass: base + full in one call (2x speedup)
        x_pair = torch.cat([x_base, x_full], dim=0)
        attn_pair = torch.cat([attn_base, attn_full], dim=0)
        out_pair = model(x_pair, attention_mask=attn_pair)
        logits_pair = prepare_logits(out_pair.logits)
        logits_base = logits_pair[0, base_pos]
        logits_full = logits_pair[1, full_pos]

        mu_t, w_t = compute_w_t(len(masked_local), n_tokens)
        ig, var, gamma, guidance_scale = compute_v2_guidance(
            logits_full,
            logits_base,
            w_t=w_t,
            eps=eps,
            gamma_cap=gamma_cap,
        )
        guided_logits = logits_full + gamma.unsqueeze(-1) * (logits_full - logits_base)

        confidence, x0 = sample_tokens(guided_logits, temperature=temperature, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, len(confidence)))

        chosen_local = masked_local[topk]
        chosen_full = chosen_local + n_prefix
        chosen_base = chosen_local + n_prefix
        x_full[0, chosen_full] = x0[topk]
        x_base[0, chosen_base] = x0[topk]
        token_confidences.extend(confidence[topk].tolist())
        ig_means.append(ig.mean().item())
        var_means.append(var.mean().item())
        gamma_means.append(gamma.mean().item())
        w_means.append(w_t)
        remaining -= len(topk)

    answer_tokens = x_full[0, n_prefix:n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), {
        "mean_ig": sum(ig_means) / len(ig_means) if ig_means else 0.0,
        "mean_var": sum(var_means) / len(var_means) if var_means else 0.0,
        "mean_gamma": sum(gamma_means) / len(gamma_means) if gamma_means else 0.0,
        "mean_guidance_scale": 1.0 + (sum(gamma_means) / len(gamma_means) if gamma_means else 0.0),
        "mean_w_t": sum(w_means) / len(w_means) if w_means else 0.0,
        "avg_conf": sum(token_confidences) / len(token_confidences) if token_confidences else 0.0,
    }


def select_remask_positions(
    positions: list[int],
    rho: torch.Tensor,
    threshold: float,
) -> list[int]:
    return [positions[i] for i in range(len(positions)) if rho[i].item() >= threshold]


@torch.inference_mode()
def eamd_remask_shared(model, tokenizer, question: str, old_context: str, new_context: str,
                       seed_tokens: torch.Tensor, steps: int = 16, temperature: float = 0.1,
                       eps: float = 1e-6, gamma_cap: float = 8.0,
                       tau: float = 0.10, remask_prior: float = 0.10,
                       remask_cost: float = 0.0, remask_threshold: float = 0.5):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    eos_id = tokenizer.eos_token_id
    n_tokens = len(seed_tokens)

    full_ids, base_ids, n_prefix = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)
    x_base = torch.tensor([base_ids], dtype=torch.long, device=device)
    x_full = torch.tensor([full_ids], dtype=torch.long, device=device)
    x_base[0, n_prefix:n_prefix + n_tokens] = seed_tokens.to(device)
    x_full[0, n_prefix:n_prefix + n_tokens] = seed_tokens.to(device)
    attn_base = torch.ones((1, len(base_ids)), dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(full_ids)), dtype=torch.long, device=device)

    positions = content_positions(seed_tokens, eos_id)
    if not positions:
        return decode_answer(tokenizer, seed_tokens), seed_tokens.clone(), {
            "remasked_positions": [],
            "mean_delta_refresh": 0.0,
            "max_delta_refresh": 0.0,
            "mean_rho": 0.0,
            "max_rho": 0.0,
            "mean_gamma": 0.0,
            "mean_ig": 0.0,
            "mean_var": 0.0,
            "mean_w_t": 0.0,
            "avg_conf": 0.0,
        }

    token_pos = torch.tensor([n_prefix + pos for pos in positions], dtype=torch.long, device=device)
    # Batched forward pass: old + new evidence in one call
    x_pair = torch.cat([x_base, x_full], dim=0)
    attn_pair = torch.cat([attn_base, attn_full], dim=0)
    out_pair = model(x_pair, attention_mask=attn_pair)
    logits_pair = prepare_logits(out_pair.logits)
    logits_old = logits_pair[0, token_pos]
    logits_new = logits_pair[1, token_pos]
    delta_refresh, rho = compute_v2_remask_probabilities(
        logits_old,
        logits_new,
        remask_prior=remask_prior,
        remask_cost=remask_cost,
        tau=tau,
        eps=eps,
    )
    committed = [seed_tokens[pos].item() for pos in positions]
    predicted = torch.argmax(logits_new, dim=-1).tolist()
    remask_positions = select_remask_positions(positions, rho, remask_threshold)

    if not remask_positions:
        return decode_answer(tokenizer, seed_tokens), seed_tokens.clone(), {
            "remasked_positions": [],
            "mean_delta_refresh": delta_refresh.mean().item(),
            "max_delta_refresh": delta_refresh.max().item(),
            "mean_rho": rho.mean().item(),
            "max_rho": rho.max().item(),
            "mean_gamma": 0.0,
            "mean_ig": 0.0,
            "mean_var": 0.0,
            "mean_w_t": 0.0,
            "avg_conf": 0.0,
            "top1_changes": sum(int(p != c) for p, c in zip(predicted, committed)),
        }

    x_base[0, [n_prefix + pos for pos in remask_positions]] = mask_id
    x_full[0, [n_prefix + pos for pos in remask_positions]] = mask_id

    remaining = len(remask_positions)
    k_per_step = max(1, math.ceil(remaining / steps))
    token_confidences = []
    ig_means = []
    var_means = []
    gamma_means = []
    w_means = []

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, n_prefix:] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + n_prefix
        base_pos = masked_local + n_prefix

        # Batched forward pass: base + full in one call (2x speedup)
        x_pair = torch.cat([x_base, x_full], dim=0)
        attn_pair = torch.cat([attn_base, attn_full], dim=0)
        out_pair = model(x_pair, attention_mask=attn_pair)
        logits_pair = prepare_logits(out_pair.logits)
        logits_base = logits_pair[0, base_pos]
        logits_full = logits_pair[1, full_pos]

        mu_t, w_t = compute_w_t(len(masked_local), n_tokens)
        ig, var, gamma, guidance_scale = compute_v2_guidance(
            logits_full,
            logits_base,
            w_t=w_t,
            eps=eps,
            gamma_cap=gamma_cap,
        )
        guided_logits = logits_full + gamma.unsqueeze(-1) * (logits_full - logits_base)

        confidence, x0 = sample_tokens(guided_logits, temperature=temperature, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        _, topk = torch.topk(confidence, min(n_commit, len(confidence)))

        chosen_local = masked_local[topk]
        chosen_full = chosen_local + n_prefix
        chosen_base = chosen_local + n_prefix
        x_full[0, chosen_full] = x0[topk]
        x_base[0, chosen_base] = x0[topk]
        token_confidences.extend(confidence[topk].tolist())
        ig_means.append(ig.mean().item())
        var_means.append(var.mean().item())
        gamma_means.append(gamma.mean().item())
        w_means.append(w_t)
        remaining -= len(topk)

    answer_tokens = x_full[0, n_prefix:n_prefix + n_tokens].clone()
    return decode_answer(tokenizer, answer_tokens), answer_tokens, {
        "remasked_positions": remask_positions,
        "mean_delta_refresh": delta_refresh.mean().item(),
        "max_delta_refresh": delta_refresh.max().item(),
        "mean_rho": rho.mean().item(),
        "max_rho": rho.max().item(),
        "mean_gamma": sum(gamma_means) / len(gamma_means) if gamma_means else 0.0,
        "mean_guidance_scale": 1.0 + (sum(gamma_means) / len(gamma_means) if gamma_means else 0.0),
        "mean_ig": sum(ig_means) / len(ig_means) if ig_means else 0.0,
        "mean_var": sum(var_means) / len(var_means) if var_means else 0.0,
        "mean_w_t": sum(w_means) / len(w_means) if w_means else 0.0,
        "avg_conf": sum(token_confidences) / len(token_confidences) if token_confidences else 0.0,
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
    parser.add_argument("--model", default="dream", choices=["dream", "llada"])
    parser.add_argument("--model_type", dest="model", choices=["dream", "llada"], help=argparse.SUPPRESS)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--dataset", default="musique", choices=sorted(QUESTION_FILES))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--questions_file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--methods", default="baseline,spread,aram,pool,eamd_v2_regen,eamd_v2_remask")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--answer_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--seed_query_mode", default="baseline", choices=["aram", "baseline", "dual"])
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma_cap", type=float, default=8.0)
    parser.add_argument("--remask_prior", type=float, default=0.10)
    parser.add_argument("--remask_cost", type=float, default=0.0)
    parser.add_argument("--remask_threshold", type=float, default=0.5)
    parser.add_argument("--refinement_rounds", type=int, default=1)
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

    model_name = args.model_name
    if model_name is None:
        model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"

    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()

    global MODEL_REF, TOKENIZER_REF, MODEL_TYPE_REF
    MODEL_REF = model
    TOKENIZER_REF = tokenizer
    MODEL_TYPE_REF = args.model

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

        baseline_answer, baseline_tokens, baseline_conf = short_generate(
            model,
            tokenizer,
            old_context,
            question,
            steps=args.steps,
            n_tokens=args.answer_tokens,
            temperature=args.temperature,
        )
        baseline = evaluate(baseline_answer, gold)
        baseline["avg_conf"] = baseline_conf
        if log_this and "baseline" in methods:
            print(f"  baseline:       {baseline_answer} | F1={baseline['f1']:.3f}", flush=True)

        need_aram_seed = ("aram" in methods) or (
            args.seed_query_mode in ("aram", "dual")
            and any(name in methods for name in ("eamd_v2_regen", "eamd_v2_remask"))
        )
        aram_answer = None
        aram_tokens = None
        aram_stats = None
        if need_aram_seed:
            aram_answer, aram_tokens, aram_stats = aram_generate_shared(
                model,
                tokenizer,
                old_context,
                question,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                temperature=args.temperature,
                lambda_max=args.lambda_max,
                beta=args.beta,
            )

        need_new_context = any(name in methods for name in ("pool", "eamd_v2_regen", "eamd_v2_remask"))
        expanded_passages, candidates = (initial_passages, [])
        if need_new_context:
            seed_hypotheses = []
            if args.seed_query_mode in ("aram", "dual") and aram_answer is not None:
                seed_hypotheses.append(aram_answer)
            if args.seed_query_mode in ("baseline", "dual"):
                seed_hypotheses.append(baseline_answer)
            if not seed_hypotheses:
                seed_hypotheses.append(baseline_answer)
            expanded_passages, candidates = expand_evidence(
                retriever,
                question,
                old_context,
                initial_passages,
                seed_hypotheses,
                args.n_candidates,
                args.expand_top_k,
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
            spread_answer, spread_stats = spread_generate_shared(
                model,
                tokenizer,
                old_context,
                question,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                temperature=args.temperature,
            )
            spread = evaluate(spread_answer, gold)
            spread["stats"] = spread_stats
            row["spread"] = spread
            if log_this:
                print(f"  spread:         {spread_answer} | F1={spread['f1']:.3f}", flush=True)

        if "aram" in methods and aram_answer is not None and aram_stats is not None:
            aram = evaluate(aram_answer, gold)
            aram["stats"] = aram_stats
            row["aram"] = aram
            if log_this:
                print(f"  aram:           {aram_answer} | F1={aram['f1']:.3f}", flush=True)

        if "pool" in methods:
            pool_answer, _, pool_conf = short_generate(
                model,
                tokenizer,
                new_context,
                question,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                temperature=args.temperature,
            )
            pool = evaluate(pool_answer, gold)
            pool["avg_conf"] = pool_conf
            row["pool"] = pool
            if log_this:
                print(f"  pool:           {pool_answer} | F1={pool['f1']:.3f}", flush=True)

        if "eamd_v2_regen" in methods:
            regen_answer, regen_stats = eamd_regen_shared(
                model,
                tokenizer,
                question,
                old_context,
                new_context,
                steps=args.steps,
                n_tokens=args.answer_tokens,
                temperature=args.temperature,
                gamma_cap=args.gamma_cap,
            )
            regen = evaluate(regen_answer, gold)
            regen["stats"] = regen_stats
            row["eamd_v2_regen"] = regen
            if log_this:
                print(f"  eamd_v2_regen: {regen_answer} | F1={regen['f1']:.3f}", flush=True)

        if "eamd_v2_remask" in methods:
            round_old_context = old_context
            round_new_context = new_context
            round_passages = list(expanded_passages)
            seed_tokens = baseline_tokens.clone()
            remask_answer = decode_answer(tokenizer, seed_tokens)
            round_stats = []
            for round_idx in range(args.refinement_rounds):
                remask_answer, seed_tokens, remask_stats = eamd_remask_shared(
                    model,
                    tokenizer,
                    question,
                    round_old_context,
                    round_new_context,
                    seed_tokens,
                    steps=args.steps,
                    temperature=args.temperature,
                    gamma_cap=args.gamma_cap,
                    tau=args.tau,
                    remask_prior=args.remask_prior,
                    remask_cost=args.remask_cost,
                    remask_threshold=args.remask_threshold,
                )
                remask_stats["round"] = round_idx + 1
                round_stats.append(remask_stats)
                if round_idx + 1 >= args.refinement_rounds:
                    break
                next_passages, _ = expand_evidence(
                    retriever,
                    question,
                    round_new_context,
                    round_passages,
                    [remask_answer],
                    args.n_candidates,
                    args.expand_top_k,
                )
                if len(next_passages) == len(round_passages):
                    break
                round_old_context = round_new_context
                round_new_context = "\n\n".join(next_passages)
                round_passages = next_passages

            remask = evaluate(remask_answer, gold)
            remask["stats"] = round_stats[-1]
            remask["round_stats"] = round_stats
            row["eamd_v2_remask"] = remask
            if log_this:
                print(f"  eamd_v2_remask:{remask_answer} | F1={remask['f1']:.3f}", flush=True)

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
            f"{DEFAULT_OUTPUT_ROOT}/{args.model}/{args.dataset}/shards/"
            f"eamd_v2_wiki18_{args.model}_{args.dataset}_{args.start_idx}_{end_idx}.json"
        )
    payload = {
        "metadata": {
            "dataset": args.dataset,
            "model": args.model,
            "model_name": model_name,
            "questions_file": questions_file,
            "start_idx": args.start_idx,
            "end_idx": end_idx,
            "n_questions": len(results),
            "methods": methods,
            "steps": args.steps,
            "answer_tokens": args.answer_tokens,
            "temperature": args.temperature,
            "n_candidates": args.n_candidates,
            "initial_top_k": args.initial_top_k,
            "expand_top_k": args.expand_top_k,
            "seed_query_mode": args.seed_query_mode,
            "tau": args.tau,
            "lambda_max": args.lambda_max,
            "beta": args.beta,
            "gamma_cap": args.gamma_cap,
            "remask_prior": args.remask_prior,
            "remask_cost": args.remask_cost,
            "remask_threshold": args.remask_threshold,
            "refinement_rounds": args.refinement_rounds,
            "retriever": "intfloat/e5-base-v2",
            "corpus": "wiki18_100w",
            "guidance": "gamma = w_t * IG / Var_{p1}(r)",
            "remask": "rho = sigmoid(logit(pi) + (Delta_refresh - c)/tau)",
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
        print(f"  {key:16s} F1={values['f1']:.3f} EM={values['em']:.3f} contain={values['contain']:.3f}", flush=True)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
