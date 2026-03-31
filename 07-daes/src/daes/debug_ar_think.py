"""Quick debug: what does Qwen3-8B actually output with thinking enabled?"""
import sys, os
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()

prompt = """You are a helpful assistant.
Answer the question using the context when possible.
Give a direct concise answer in 1 to 6 words.
Do not explain.
You may think first, but end with exactly one line formatted as 'Final answer: <short phrase>'.

Context:
The Green family includes Tina Green and Sir Philip Green.

Question: Who is Tina Green's spouse?
Answer:"""

messages = [{"role": "user", "content": prompt}]

# Try with thinking enabled
try:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
except:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)

# Decode WITHOUT skip_special_tokens to see raw output
raw_full = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=False)
raw_clean = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

print("=== RAW (skip_special_tokens=False) ===")
print(repr(raw_full[:500]))
print()
print("=== CLEAN (skip_special_tokens=True) ===")
print(repr(raw_clean[:500]))
