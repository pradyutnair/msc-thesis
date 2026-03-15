#!/usr/bin/env python3
"""Offline GRPO training for the escalation policy.

Trains Qwen3-8B with LoRA using counterfactual trajectories from
collect_escalation_trajectories.py. For each sub-question, we have
results from BOTH structured and agentic workers, giving us exact
counterfactual advantages for the ACCEPT/ESCALATE decision.

The escalation policy learns to decide — after observing a structured
worker's output — whether the result is sufficient (ACCEPT) or whether
an expensive agentic worker should be invoked (ESCALATE).

Usage:
    python scripts/train_escalation_grpo.py \
        --config configs/grpo_training.yaml \
        --trajectories trajectories/escalation_hotpotqa/escalation_trajectories.jsonl \
                       trajectories/escalation_2wikimultihop/escalation_trajectories.jsonl \
                       trajectories/escalation_musique/escalation_trajectories.jsonl \
        --output models/escalation_lora \
        --cost-penalty 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Prompt and completion templates ──────────────────────────────────

ESCALATION_PROMPT_TEMPLATE = """\
You are an escalation policy for a multi-hop question answering system.
A structured retrieval worker has attempted to answer a sub-question.
Decide whether to ACCEPT the structured result or ESCALATE to a more expensive agentic retrieval worker.

Original question: {question}
Sub-question: {sq_text}

Structured retrieval result:
- Answer: "{structured_answer}"
- Evidence passages found: {evidence_count}

Previously resolved answers:
{entity_context}

Decision (ACCEPT or ESCALATE with brief reasoning):
"""


def build_accept_completion(structured_answer: str, evidence_count: int) -> str:
    if evidence_count > 0:
        return f'ACCEPT. Structured retrieval found "{structured_answer}" with {evidence_count} supporting passages.'
    else:
        return f'ACCEPT. Structured retrieval found "{structured_answer}".'


def build_escalate_completion(structured_answer: str) -> str:
    s_usable = _is_usable(structured_answer)
    if not s_usable:
        return "ESCALATE. Structured retrieval failed to find a usable answer. Agentic search needed."
    else:
        return f'ESCALATE. Structured answer "{structured_answer}" may be insufficient. Deeper agentic search needed.'


# ── Utility functions ────────────────────────────────────────────────

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def _is_usable(answer: str) -> bool:
    if not answer:
        return False
    return answer.strip().lower() not in ("unknown", "error", "", "none", "n/a")


def compute_escalation_rewards(
    structured: dict,
    agentic: dict,
    cost_penalty: float = 0.2,
    gold_answer: str = "",
) -> tuple[float, float]:
    """Compute rewards for ACCEPT and ESCALATE given counterfactual results.

    Returns (r_accept, r_escalate).

    Reward logic:
    - ACCEPT is good when structured suffices (saves compute)
    - ESCALATE is good when only agentic finds the answer (avoids wrong answer)
    - Escalation always incurs a cost penalty (extra compute)
    """
    s_answer = structured.get("answer", "")
    a_answer = agentic.get("answer", "")
    s_usable = _is_usable(s_answer)
    a_usable = _is_usable(a_answer)
    answers_match = _normalize(s_answer) == _normalize(a_answer) and s_usable and a_usable

    if s_usable and (not a_usable or answers_match):
        # Structured is sufficient — ACCEPT is clearly correct
        r_accept = 1.0
        r_escalate = -cost_penalty  # wasted compute, no improvement
    elif not s_usable and a_usable:
        # Only agentic found an answer — ESCALATE is clearly correct
        r_accept = -1.0  # would have missed the answer
        r_escalate = 1.0 - cost_penalty  # justified escalation
    elif s_usable and a_usable and not answers_match:
        # Both found different answers — use gold to determine correct one
        if gold_answer:
            gold_norm = _normalize(gold_answer)
            s_match = gold_norm in _normalize(s_answer) or _normalize(s_answer) in gold_norm
            a_match = gold_norm in _normalize(a_answer) or _normalize(a_answer) in gold_norm
            if s_match and not a_match:
                # Structured was correct — ACCEPT is right
                r_accept = 1.0
                r_escalate = -cost_penalty
            elif a_match and not s_match:
                # Agentic was correct — ESCALATE is justified
                r_accept = -0.5
                r_escalate = 1.0 - cost_penalty
            else:
                # Both match or neither — ambiguous
                r_accept = 0.3
                r_escalate = 0.5 - cost_penalty
        else:
            # No gold answer — slight preference for ACCEPT
            r_accept = 0.3
            r_escalate = 0.5 - cost_penalty
    else:
        # Both failed — neither strategy helps
        r_accept = 0.0
        r_escalate = -cost_penalty  # escalation wasted compute
    return r_accept, r_escalate


# ── Dataset ──────────────────────────────────────────────────────────

class EscalationGRPODataset(Dataset):
    """Dataset of (prompt, completion, advantage) tuples for escalation GRPO.

    For each non-skipped sub-question in the counterfactual trajectories,
    generates TWO samples: one for ACCEPT, one for ESCALATE, with advantages
    computed from the counterfactual reward difference.
    """

    def __init__(
        self,
        trajectories_files: list[str],
        tokenizer,
        max_length: int = 512,
        cost_penalty: float = 0.2,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []

        total_loaded = 0
        skipped_both_failed = 0
        skipped_missing = 0

        for tfile in trajectories_files:
            if not Path(tfile).exists():
                logger.warning("Trajectory file not found: %s", tfile)
                continue
            with open(tfile) as f:
                questions = [json.loads(line) for line in f if line.strip()]
            total_loaded += len(questions)

            for q_data in questions:
                question = q_data["question"]
                # Build entity context from prior SQ answers
                entity_lines = []

                for sq_result in q_data["sub_question_results"]:
                    if sq_result.get("skipped"):
                        continue

                    structured = sq_result.get("structured")
                    agentic = sq_result.get("agentic")
                    if not structured or not agentic:
                        skipped_missing += 1
                        continue

                    s_answer = structured.get("answer", "")
                    a_answer = agentic.get("answer", "")

                    # Skip if both produced nothing useful
                    if not _is_usable(s_answer) and not _is_usable(a_answer):
                        skipped_both_failed += 1
                        # Still update entity context (empty) for downstream SQs
                        continue

                    # Build the escalation prompt
                    entity_context = "\n".join(entity_lines) if entity_lines else "(none yet)"
                    prompt = ESCALATION_PROMPT_TEMPLATE.format(
                        question=question,
                        sq_text=sq_result["resolved_text"],
                        structured_answer=s_answer if s_answer else "(no answer)",
                        evidence_count=structured.get("evidence_count", 0),
                        entity_context=entity_context,
                    )

                    # Build completions
                    accept_completion = build_accept_completion(
                        s_answer, structured.get("evidence_count", 0),
                    )
                    escalate_completion = build_escalate_completion(s_answer)

                    # Compute counterfactual rewards (with gold-answer tie-breaking)
                    r_accept, r_escalate = compute_escalation_rewards(
                        structured, agentic, cost_penalty,
                        gold_answer=q_data.get("gold_answer", ""),
                    )

                    # Compute advantages (group mean baseline with group_size=2)
                    mean_r = (r_accept + r_escalate) / 2.0

                    self.samples.append({
                        "prompt": prompt,
                        "completion": accept_completion,
                        "advantage": r_accept - mean_r,
                        "reward": r_accept,
                        "action": "ACCEPT",
                    })
                    self.samples.append({
                        "prompt": prompt,
                        "completion": escalate_completion,
                        "advantage": r_escalate - mean_r,
                        "reward": r_escalate,
                        "action": "ESCALATE",
                    })

                    # Update entity context for downstream SQs
                    # Use the best available answer
                    best = s_answer if _is_usable(s_answer) else a_answer
                    if best:
                        entity_lines.append(
                            f"- SQ-{sq_result['sq_id']}: \"{sq_result['sq_text'][:60]}\" -> {best}"
                        )

        # Log dataset statistics
        n_accept = sum(1 for s in self.samples if s["action"] == "ACCEPT")
        n_escalate = sum(1 for s in self.samples if s["action"] == "ESCALATE")
        advantages = [s["advantage"] for s in self.samples]
        rewards = [s["reward"] for s in self.samples]

        logger.info(
            "Loaded %d samples (%d ACCEPT, %d ESCALATE) from %d questions across %d files",
            len(self.samples), n_accept, n_escalate, total_loaded, len(trajectories_files),
        )
        logger.info(
            "Skipped: %d both_failed, %d missing data", skipped_both_failed, skipped_missing,
        )
        if advantages:
            logger.info(
                "Advantage stats: mean=%.3f, min=%.3f, max=%.3f",
                sum(advantages) / len(advantages), min(advantages), max(advantages),
            )
            logger.info(
                "Reward stats: mean=%.3f, min=%.3f, max=%.3f",
                sum(rewards) / len(rewards), min(rewards), max(rewards),
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_text = sample["prompt"] + sample["completion"]

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_encoding = self.tokenizer(
            sample["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "prompt_length": prompt_len,
            "advantage": sample["advantage"],
        }


# ── Collate and loss (same as train_grpo.py) ─────────────────────────

def collate_fn(batch):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_token_id = 0

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    prompt_lengths = []
    advantages = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        prompt_lengths.append(item["prompt_length"])
        advantages.append(item["advantage"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
        "advantages": torch.tensor(advantages, dtype=torch.float32),
    }


def compute_grpo_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute advantage-weighted language modeling loss on completions only."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out prompt tokens — only compute loss on completion
    completion_mask = shift_mask.clone()
    for i, pl in enumerate(prompt_lengths):
        completion_mask[i, : max(pl - 1, 0)] = 0

    per_sample_loss = -(token_log_probs * completion_mask).sum(dim=1)
    per_sample_count = completion_mask.sum(dim=1).clamp(min=1)
    per_sample_nll = per_sample_loss / per_sample_count

    # Advantage-weighted loss: positive advantages reinforce, negative suppress
    weighted_loss = (per_sample_nll * advantages.to(per_sample_nll.device)).mean()
    return weighted_loss


# ── Training loop ────────────────────────────────────────────────────

def train(
    config: dict,
    trajectories_files: list[str],
    output_dir: str,
    cost_penalty: float,
):
    model_name = config["model_name"]
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lr = config.get("learning_rate", 5e-6)
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("batch_size", 8)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    max_length = config.get("escalation_max_length", 512)
    warmup_ratio = config.get("warmup_ratio", 0.1)
    weight_decay = config.get("weight_decay", 0.01)

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = EscalationGRPODataset(
        trajectories_files=trajectories_files,
        tokenizer=tokenizer,
        max_length=max_length,
        cost_penalty=cost_penalty,
    )

    if len(dataset) == 0:
        logger.error("No training samples found. Exiting.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    total_steps = (len(dataloader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    logger.info(
        "Training: %d samples, %d epochs, %d steps/epoch, batch=%d, effective_batch=%d",
        len(dataset), num_epochs, len(dataloader), batch_size, batch_size * grad_accum,
    )

    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            advantages = batch["advantages"]
            prompt_lengths = batch["prompt_lengths"]

            loss = compute_grpo_loss(
                model, input_ids, attention_mask, prompt_lengths, advantages,
            )
            loss = loss / grad_accum
            loss.backward()
            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    logger.info(
                        "Epoch %d | Step %d/%d | Loss: %.4f | LR: %.2e",
                        epoch + 1, global_step, total_steps,
                        epoch_loss / (step + 1), scheduler.get_last_lr()[0],
                    )

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch + 1, avg_loss)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    logger.info("Saved escalation LoRA adapter to %s", out_path)

    # Save training metadata
    metadata = {
        "model_name": model_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "cost_penalty": cost_penalty,
        "num_samples": len(dataset),
        "trajectories_files": trajectories_files,
    }
    with open(out_path / "escalation_training_config.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Offline GRPO for escalation policy")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument(
        "--trajectories", required=True, nargs="+",
        help="Counterfactual trajectory JSONL files",
    )
    parser.add_argument("--output", required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--cost-penalty", type=float, default=0.2, help="Token cost penalty for escalation")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.trajectories, args.output, args.cost_penalty)


if __name__ == "__main__":
    main()
