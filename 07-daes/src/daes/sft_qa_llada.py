"""
Masked SFT for Dream-7B on multi-hop QA data.
Adapted from d1 (arxiv 2504.12216) for Dream architecture.

Trains Dream-7B with LoRA to generate: <reasoning>...</reasoning><answer>...</answer>
given context + question. Uses absorbing state diffusion loss from d1.

Key adaptations from d1:
- Model: Dream-7B-Instruct (not LLaDA-8B)
- mask_token_id: 151666 (Dream, not LLaDA's 126336)
- Data: multi-hop QA trajectories (not math)
- Single GPU training (no deepspeed)
"""

import torch
import torch.nn.functional as F
import argparse
import json
import os
import sys
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# d1-style diffusion trainer
# ---------------------------------------------------------------------------

class DiffusionTrainer(Trainer):
    """Absorbing state diffusion loss (from d1)."""
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        t = inputs.pop("t")
        num_prompt_tokens = inputs.pop("num_prompt_tokens")

        outputs = model(**inputs)
        logits = outputs.logits
        # LLaDA: NO AR-shift needed (position i predicts position i directly)

        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)

        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            valid = (labels != -100).sum()
            if valid > 0:
                self.log({"unscaled_loss": (unscaled_loss.sum() / valid).item()})

        loss = unscaled_loss / t
        total_tokens = inputs["input_ids"].numel() - num_prompt_tokens
        loss = loss.sum() / max(total_tokens, 1)
        return loss if not return_outputs else (loss, outputs)


class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, eval=False):
        self.data = data
        self.eval = eval
        if eval:
            self.t = torch.linspace(0, 1, len(data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class DiffusionDataCollator(DefaultDataCollator):
    """Forward noising: randomly mask tokens, preserve prompt tokens."""
    def __init__(self, tokenizer, mask_token_id, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.max_length = max_length

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]
        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)
        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0

        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()

        batch["input_ids"] = noisy_batch.long()
        return batch


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant that answers questions.
INSTRUCTIONS:
1. First, check if the answer is in the provided context passages
2. If the answer is in the context, use it
3. Always provide a direct, concise answer (typically 1-10 words)
4. Do NOT include explanations or reasoning
5. Never say no answer found - always attempt to answer"""


def preprocess_data(data_path, tokenizer, max_length=4096, test_split=0.02):
    """Load and preprocess QA data in d1 format."""
    data = [json.loads(line) for line in open(data_path)]
    print(f"Loaded {len(data)} examples from {data_path}")

    preprocessed = []
    skipped = 0
    for ex in tqdm(data, desc="Preprocessing"):
        question = ex["question"]
        reasoning = ex["thinking_trajectories"][0]
        answer = ex["attempt"]

        # Truncate long answers
        answer = answer[:200]
        reasoning = reasoning[:1500]

        prompt_content = SYSTEM_PROMPT + "\n\n" + question
        response_content = f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{answer}\n</answer>"

        prompt = [{"role": "user", "content": prompt_content}]
        response = [{"role": "assistant", "content": response_content}]

        try:
            full_text = tokenizer.apply_chat_template(prompt + response, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False,
                                                        add_generation_prompt=True)

            tokenized = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=max_length, padding="max_length"
            ).input_ids.squeeze(0)

            prompt_tokens = tokenizer(
                prompt_text, return_tensors="pt", truncation=True,
                max_length=max_length
            )

            preprocessed.append({
                "input_ids": tokenized,
                "prompt_lengths": prompt_tokens.attention_mask.sum(-1),
            })
        except Exception as e:
            skipped += 1
            if skipped < 5:
                print(f"Skip: {e}")

    print(f"Preprocessed: {len(preprocessed)}, Skipped: {skipped}")

    random.shuffle(preprocessed)
    n_test = max(1, int(len(preprocessed) * test_split))
    return preprocessed[n_test:], preprocessed[:n_test]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data_path", default="/projects/prjs1800/msc-thesis/07-daes/data/d1_qa_sft_answer_only.jsonl")
    parser.add_argument("--output_dir", default="/projects/prjs1800/msc-thesis/07-daes/models/llada_qa_sft")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    args = parser.parse_args()

    init_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    print(f"Loading {args.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    print(f"Model loaded. mask_token_id={tokenizer.mask_token_id}", flush=True)

    # LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()

    # Data
    train_data, eval_data = preprocess_data(args.data_path, tokenizer, args.max_length)
    train_dataset = QADataset(train_data)
    eval_dataset = QADataset(eval_data, eval=True)
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}", flush=True)

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        learning_rate=args.lr,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        data_collator=DiffusionDataCollator(
            tokenizer=tokenizer,
            mask_token_id=126336,
            max_length=args.max_length,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...", flush=True)
    trainer.train()
    print("Training complete!", flush=True)

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
