"""Diagnose token distribution at intermediate denoising steps on LLaDA vs Dream.
Key question: does LLaDA have more diversity at earlier steps?"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy, Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18


def diagnose_temporal(model, tokenizer, context, question, total_steps=32, n_tokens=12, checkpoints=None):
    """Run denoising and snapshot the distribution at multiple timesteps."""
    if checkpoints is None:
        checkpoints = [0, 2, 4, 8, 16, total_steps]  # steps at which to snapshot

    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    k_per_step = max(1, math.ceil(n_tokens / total_steps))
    remaining = n_tokens
    neg_ent = _neg_entropy()

    snapshots = []

    for step in range(total_steps):
        if remaining <= 0:
            break
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)

        # Snapshot at checkpoints
        if step in checkpoints:
            answer_logits = logits[0, n_prefix:n_prefix + n_tokens]
            probs = torch.softmax(answer_logits / 0.3, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

            n_masked = len(masked_local)
            pos_data = []
            for pos in range(min(n_tokens, 6)):
                is_masked = pos in masked_local.tolist()
                top_probs, top_ids = torch.topk(probs[pos], 5)
                top_tokens = [tokenizer.decode([tid.item()]) for tid in top_ids]
                pos_data.append({
                    "pos": pos,
                    "masked": is_masked,
                    "entropy": round(entropy[pos].item(), 3),
                    "top_tokens": list(zip(top_tokens, [round(p.item(), 4) for p in top_probs]))
                })

            snapshots.append({
                "step": step,
                "n_masked": n_masked,
                "n_committed": n_tokens - n_masked,
                "mean_entropy_masked": round(entropy[masked_local].mean().item(), 3) if n_masked > 0 else 0,
                "max_entropy_masked": round(entropy[masked_local].max().item(), 3) if n_masked > 0 else 0,
                "positions": pos_data,
            })

        # Commit tokens (standard denoising)
        token_logits = logits[0, masked_local + n_prefix]
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=neg_ent)
        n_commit = min(k_per_step, remaining)
        if step == total_steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

    # Final snapshot
    final_text = decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])

    return {
        "snapshots": snapshots,
        "final_answer": final_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_questions", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded: {args.model}", flush=True)

    retriever = Wiki18Retriever()
    questions = json.load(open(QUESTION_FILES[args.dataset]))[:args.n_questions]
    query_texts = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, 5)
    print(f"Retrieved for {len(questions)} questions", flush=True)

    results = []
    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        context = "\n\n".join(all_initial[qi])

        diag = diagnose_temporal(model, tokenizer, context, qtext)
        diag["question"] = qtext
        diag["gold"] = gold
        results.append(diag)

        print(f"\n[{qi+1}] {qtext[:70]}", flush=True)
        print(f"  Gold: {gold}", flush=True)
        print(f"  Final: {diag['final_answer']}", flush=True)
        for snap in diag["snapshots"]:
            print(f"\n  Step {snap['step']} ({snap['n_masked']} masked, {snap['n_committed']} committed):", flush=True)
            print(f"    Mean entropy (masked): {snap['mean_entropy_masked']}, Max: {snap['max_entropy_masked']}", flush=True)
            for pos in snap["positions"][:4]:
                marker = "M" if pos["masked"] else "C"
                top3 = pos["top_tokens"][:3]
                print(f"    Pos {pos['pos']} [{marker}] H={pos['entropy']:.2f}: {top3}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
