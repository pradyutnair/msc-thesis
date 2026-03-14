#!/usr/bin/env python3
"""Offline GRPO training for the orchestration planner.

Trains Qwen3-8B with LoRA using pre-collected trajectories from
collect_trajectories.py. Implements offline GRPO: group completions by
question, compute normalized advantages, and train with weighted SFT loss.

Usage:
    accelerate launch scripts/train_grpo.py \
        --config configs/grpo_training.yaml \
        --trajectories trajectories/hotpotqa/trajectories.jsonl \
        --output models/orchestrator_lora
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class GRPODataset(Dataset):
    """Dataset of (prompt, completion, advantage) tuples for offline GRPO."""

    def __init__(
        self,
        trajectories_file: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        min_group_size: int = 2,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(trajectories_file) as f:
            raw = [json.loads(line) for line in f if line.strip()]

        groups: dict[str, list[dict]] = defaultdict(list)
        for t in raw:
            groups[t["qid"]].append(t)

        self.samples = []
        skipped = 0
        for qid, group in groups.items():
            if len(group) < min_group_size:
                skipped += 1
                continue

            rewards = [t["reward"] for t in group]
            mean_r = sum(rewards) / len(rewards)
            std_r = max((sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5, 1e-6)

            for t in group:
                advantage = (t["reward"] - mean_r) / std_r
                if t.get("decomposition_text"):
                    self.samples.append({
                        "question": t["question"],
                        "completion": t["decomposition_text"],
                        "advantage": advantage,
                        "reward": t["reward"],
                    })

        logger.info("Loaded %d samples from %d groups (skipped %d small groups)",
                     len(self.samples), len(groups) - skipped, skipped)

        advantages = [s["advantage"] for s in self.samples]
        rewards = [s["reward"] for s in self.samples]
        logger.info("Advantage stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                     sum(advantages) / len(advantages),
                     (sum((a - sum(advantages) / len(advantages)) ** 2 for a in advantages) / len(advantages)) ** 0.5,
                     min(advantages), max(advantages))
        logger.info("Reward stats: mean=%.3f, min=%.3f, max=%.3f",
                     sum(rewards) / len(rewards), min(rewards), max(rewards))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = f"Question: {sample['question']}\n\nDecompose this question into sub-questions with retrieval modes."
        full_text = prompt + "\n" + sample["completion"]

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_encoding = self.tokenizer(
            prompt + "\n",
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

    completion_mask = shift_mask.clone()
    for i, pl in enumerate(prompt_lengths):
        completion_mask[i, :max(pl - 1, 0)] = 0

    per_sample_loss = -(token_log_probs * completion_mask).sum(dim=1)
    per_sample_count = completion_mask.sum(dim=1).clamp(min=1)
    per_sample_nll = per_sample_loss / per_sample_count

    weighted_loss = (per_sample_nll * advantages.to(per_sample_nll.device)).mean()
    return weighted_loss


def train(config: dict, trajectories_file: str, output_dir: str):
    model_name = config["model_name"]
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lr = config.get("learning_rate", 5e-6)
    num_epochs = config.get("num_epochs", 2)
    batch_size = config.get("batch_size", 4)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    max_length = config.get("max_length", 2048)
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

    dataset = GRPODataset(
        trajectories_file=trajectories_file,
        tokenizer=tokenizer,
        max_length=max_length,
    )
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

    logger.info("Training: %d epochs, %d steps/epoch, effective batch=%d",
                num_epochs, len(dataloader), batch_size * grad_accum)

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
                    logger.info("Epoch %d | Step %d/%d | Loss: %.4f | LR: %.2e",
                                epoch + 1, global_step, total_steps,
                                epoch_loss / (step + 1), scheduler.get_last_lr()[0])

        avg_loss = epoch_loss / len(dataloader)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch + 1, avg_loss)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    logger.info("Saved LoRA adapter to %s", out_path)

    with open(out_path / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Offline GRPO training")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--trajectories", required=True, help="Trajectories JSONL")
    parser.add_argument("--output", required=True, help="Output directory for LoRA adapter")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.trajectories, args.output)


if __name__ == "__main__":
    main()
