#!/usr/bin/env python3
"""SFT fine-tuning on correct sub-question answers from counterfactual data.

Extracts (sub-question context, correct answer) pairs from counterfactual
trajectories where at least one retrieval strategy succeeded. Fine-tunes
Qwen3-8B with LoRA to improve the base model's QA ability, which benefits
ALL workers (structured summarizer, agentic reasoner, aggregate synthesizer).

This is Stage 1 of the training pipeline:
  Stage 1: SFT on counterfactual SQ answers  -> better base model
  Stage 2: GRPO for planner                  -> better decompositions
  Stage 3: GRPO for escalation policy         -> better strategy selection

Usage:
    python scripts/train_worker_sft.py \
        --config configs/grpo_training.yaml \
        --trajectories trajectories/escalation_hotpotqa/escalation_trajectories.jsonl \
                       trajectories/escalation_2wikimultihop/escalation_trajectories.jsonl \
                       trajectories/escalation_musique/escalation_trajectories.jsonl \
        --output models/sft_base
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Prompt template ──────────────────────────────────────────────────

SQ_ANSWER_PROMPT = """\
Answer the following sub-question for a multi-hop question answering system.
Give a concise, factual answer (1-5 words).

Original question: {question}
Sub-question: {sq_text}

Previously resolved answers:
{entity_context}

Answer: """


# ── Utilities ────────────────────────────────────────────────────────

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


def _pick_best_answer(structured: dict, agentic: dict, gold_answer: str = "") -> str | None:
    """Pick the best available answer from counterfactual pair.

    Uses gold-answer containment check to resolve disagreements between
    structured and agentic workers. This eliminates noisy training signal
    from incorrect answers.
    """
    s_ans = structured.get("answer", "")
    a_ans = agentic.get("answer", "")
    s_ok = _is_usable(s_ans)
    a_ok = _is_usable(a_ans)

    if a_ok and s_ok:
        # Both usable — check if they agree
        if _normalize(s_ans) == _normalize(a_ans):
            return a_ans  # agreement

        # Disagreement: use gold answer to pick the correct one
        if gold_answer:
            gold_norm = _normalize(gold_answer)
            s_match = gold_norm in _normalize(s_ans) or _normalize(s_ans) in gold_norm
            a_match = gold_norm in _normalize(a_ans) or _normalize(a_ans) in gold_norm
            if a_match and not s_match:
                return a_ans
            elif s_match and not a_match:
                return s_ans
            elif not s_match and not a_match:
                return None  # neither matches gold — skip this SQ
        # No gold or both match: prefer agentic
        return a_ans
    elif a_ok:
        return a_ans
    elif s_ok:
        return s_ans
    return None


# ── Dataset ──────────────────────────────────────────────────────────

class SubQuestionSFTDataset(Dataset):
    """Dataset of (prompt, answer) pairs for SFT from counterfactual data."""

    def __init__(
        self,
        trajectories_files: list[str],
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict[str, str]] = []

        total_sqs = 0
        n_usable = 0
        n_skipped = 0
        n_both_failed = 0

        for tfile in trajectories_files:
            if not Path(tfile).exists():
                logger.warning("File not found: %s", tfile)
                continue

            with open(tfile) as f:
                questions = [json.loads(line) for line in f if line.strip()]

            for q_data in questions:
                question = q_data["question"]
                entity_lines: list[str] = []

                for sq_result in q_data["sub_question_results"]:
                    total_sqs += 1

                    if sq_result.get("skipped"):
                        n_skipped += 1
                        continue

                    structured = sq_result.get("structured", {})
                    agentic = sq_result.get("agentic", {})

                    best_answer = _pick_best_answer(structured, agentic, q_data.get("gold_answer", ""))
                    if best_answer is None:
                        n_both_failed += 1
                        continue

                    n_usable += 1

                    entity_context = (
                        "\n".join(entity_lines) if entity_lines
                        else "(none yet)"
                    )
                    prompt = SQ_ANSWER_PROMPT.format(
                        question=question,
                        sq_text=sq_result["resolved_text"],
                        entity_context=entity_context,
                    )

                    self.samples.append({
                        "prompt": prompt,
                        "answer": best_answer,
                    })

                    # Track entity context for downstream SQs
                    entity_lines.append(
                        f"- SQ-{sq_result['sq_id']}: {best_answer}"
                    )

        logger.info(
            "SFT dataset: %d samples from %d SQs "
            "(skipped=%d aggregate, %d both_failed, %d usable, %d files)",
            len(self.samples), total_sqs, n_skipped, n_both_failed,
            n_usable, len(trajectories_files),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        full_text = sample["prompt"] + sample["answer"]

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
        }


# ── Collate ──────────────────────────────────────────────────────────

def collate_fn(batch):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    pad_token_id = 0

    input_ids = torch.full(
        (len(batch), max_len), pad_token_id, dtype=torch.long,
    )
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    prompt_lengths = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        prompt_lengths.append(item["prompt_length"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_lengths": prompt_lengths,
    }


# ── Loss (standard SFT: NLL on completion tokens only) ───────────────

def compute_sft_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
) -> torch.Tensor:
    """Standard language modeling loss on answer tokens only."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Mask out prompt tokens
    completion_mask = shift_mask.clone()
    for i, pl in enumerate(prompt_lengths):
        completion_mask[i, : max(pl - 1, 0)] = 0

    # Cross-entropy loss on completion tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_logits.size(0), shift_logits.size(1))

    masked_loss = (per_token_loss * completion_mask).sum(dim=1)
    token_count = completion_mask.sum(dim=1).clamp(min=1)
    per_sample_loss = masked_loss / token_count

    return per_sample_loss.mean()


# ── Training ─────────────────────────────────────────────────────────

def train(
    config: dict,
    trajectories_files: list[str],
    output_dir: str,
):
    model_name = config["model_name"]
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    lr = config.get("sft_learning_rate", config.get("learning_rate", 2e-5))
    num_epochs = config.get("sft_num_epochs", 3)
    batch_size = config.get("sft_batch_size", config.get("batch_size", 8))
    grad_accum = config.get("gradient_accumulation_steps", 4)
    max_length = config.get("sft_max_length", 512)
    warmup_ratio = config.get("warmup_ratio", 0.1)
    weight_decay = config.get("weight_decay", 0.01)

    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
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

    dataset = SubQuestionSFTDataset(
        trajectories_files=trajectories_files,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    if len(dataset) == 0:
        logger.error("No training samples. Exiting.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )
    total_steps = (len(dataloader) // grad_accum) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    logger.info(
        "SFT training: %d samples, %d epochs, batch=%d, eff_batch=%d, "
        "total_steps=%d",
        len(dataset), num_epochs, batch_size,
        batch_size * grad_accum, total_steps,
    )

    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lengths = batch["prompt_lengths"]

            loss = compute_sft_loss(
                model, input_ids, attention_mask, prompt_lengths,
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
                        epoch + 1,
                        global_step,
                        total_steps,
                        epoch_loss / (step + 1),
                        scheduler.get_last_lr()[0],
                    )

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info("Epoch %d complete. Avg loss: %.4f", epoch + 1, avg_loss)

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Merge LoRA into base model for downstream use
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    logger.info("Saved merged SFT model to %s", out_path)

    # Save training metadata
    metadata = {
        "base_model": model_name,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "num_samples": len(dataset),
        "trajectories_files": trajectories_files,
        "stage": "sft_worker",
    }
    with open(out_path / "sft_training_config.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="SFT on sub-question answers from counterfactual data",
    )
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument(
        "--trajectories",
        required=True,
        nargs="+",
        help="Counterfactual trajectory JSONL files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged SFT model",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.trajectories, args.output)


if __name__ == "__main__":
    main()
