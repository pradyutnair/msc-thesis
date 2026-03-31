"""Quick 10q LLaDA validation with ARAM-matched settings.
32 tokens, 32 steps, low-confidence unmasking (neg_entropy=False).
"""
import json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, compute_em,
    short_user_prompt, extract_candidates_generic, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import eamd_v2_wiki18
from transformers import AutoTokenizer, AutoModel


@torch.inference_mode()
def simple_decode_llada(model, tokenizer, context, question, steps=32, n_tokens=32):
    """Decode with low-confidence unmasking (LLaDA native)."""
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)
    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break
        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, masked_local + n_prefix]
        # KEY: neg_entropy=False for LLaDA low-confidence unmasking
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=False)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)
    return decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])


def main():
    t_start = time.time()
    print("=== LLaDA 10q Validation (32 tokens, 32 steps, low-confidence) ===", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().eval()
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = "llada"
    print(f"  Loaded in {time.time() - t_start:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    # Use HotpotQA to match ARAM's reported 20.7% EM
    questions = json.load(open(QUESTION_FILES["hotpotqa"]))[:10]

    queries = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(queries, 5)
    print(f"  Retrieved in {time.time() - t_start:.1f}s", flush=True)

    total_f1 = 0; total_em = 0
    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        ctx = "\n\n".join(all_initial[qi])

        ans = simple_decode_llada(model, tokenizer, ctx, qtext, steps=32, n_tokens=32)
        f1_result = compute_f1(ans, gold)
        f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[2]
        em = compute_em(ans, gold)
        total_f1 += f1; total_em += em
        print(f"  [{qi+1}/10] gold=[{gold[:25]}] pred=[{ans[:25]}] F1={f1:.3f} EM={em:.0f}", flush=True)

    print(f"\nMean F1={total_f1/10:.4f} Mean EM={total_em/10:.4f}")
    print(f"Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
