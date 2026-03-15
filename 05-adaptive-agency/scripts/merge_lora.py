#!/usr/bin/env python3
"""Merge a LoRA adapter into a base model and save the result.

Usage:
    python scripts/merge_lora.py \
        --base models/sft_base \
        --lora models/planner_lora \
        --output models/sft_planner_base
"""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Path to base model")
    parser.add_argument("--lora", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    args = parser.parse_args()

    logger.info("Loading base model: %s", args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    logger.info("Loading LoRA adapter: %s", args.lora)
    model = PeftModel.from_pretrained(model, args.lora)

    logger.info("Merging weights...")
    merged = model.merge_and_unload()

    logger.info("Saving merged model to: %s", args.output)
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
