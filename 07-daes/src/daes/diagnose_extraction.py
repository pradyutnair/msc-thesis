"""Diagnose bridge extraction quality: Dream vs LLaDA side by side."""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy, Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, short_generate, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18


def diagnose_extraction(model, tokenizer, context, question, n_positions=4, n_branch=3, n_mask=12):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_mask
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    out = model(x, attention_mask=attn)
    logits = prepare_logits(out.logits)
    answer_logits = logits[0, n_prefix:n_prefix + n_mask]

    probs = torch.softmax(answer_logits / 0.3, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    positions = []
    for pos in range(min(n_mask, 6)):
        top_probs, top_ids = torch.topk(probs[pos], 10)
        top_tokens = [tokenizer.decode([tid.item()]) for tid in top_ids]
        positions.append({
            "pos": pos,
            "entropy": round(entropy[pos].item(), 3),
            "top_tokens": list(zip(top_tokens, [round(p.item(), 4) for p in top_probs]))
        })

    entropy_positions = torch.topk(entropy, min(n_positions, n_mask)).indices.tolist()
    top_positions = [0]
    for p in entropy_positions:
        if p not in top_positions:
            top_positions.append(p)
        if len(top_positions) >= n_positions + 1:
            break

    candidates = []
    neg_ent = _neg_entropy()
    for pos_local in top_positions:
        pos_global = n_prefix + pos_local
        pos_probs = torch.softmax(answer_logits[pos_local] / 0.3, dim=-1)
        top_probs, top_ids = torch.topk(pos_probs, n_branch)
        for i in range(len(top_probs)):
            x_c = torch.tensor([canvas], dtype=torch.long, device=device)
            x_c[0, pos_global] = top_ids[i].item()
            seed_token = tokenizer.decode([top_ids[i].item()])
            remaining = n_mask - 1
            for step in range(12):
                if remaining <= 0:
                    break
                mi = (x_c[0] == mask_id)
                if not mi.any():
                    break
                o2 = model(x_c, attention_mask=attn)
                l2 = prepare_logits(o2.logits)
                mp = mi.nonzero(as_tuple=True)[0]
                c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=neg_ent)
                k = min(max(1, remaining // 12), remaining)
                if step == 11:
                    k = remaining
                _, tk = torch.topk(c2, min(k, len(c2)))
                x_c[0, mp[tk]] = x02[tk]
                remaining -= len(tk)
            cand_text = tokenizer.decode(x_c[0, n_prefix:n_prefix + n_mask].tolist(),
                                         skip_special_tokens=True).strip()
            cand_text = cand_text.split("\n")[0].split(". ")[0].strip()
            candidates.append({
                "position": pos_local,
                "seed_token": seed_token,
                "seed_prob": round(top_probs[i].item(), 4),
                "full_text": cand_text
            })

    return {
        "positions": positions,
        "selected_positions": top_positions,
        "candidates": candidates,
        "mean_entropy": round(entropy.mean().item(), 3),
        "max_entropy": round(entropy.max().item(), 3),
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
        diag = diagnose_extraction(model, tokenizer, context, qtext)
        diag["question"] = qtext
        diag["gold"] = gold
        diag["id"] = q.get("qid") or q.get("id", f"dev_{qi}")
        results.append(diag)

        print(f"\n[{qi+1}] {qtext[:80]}", flush=True)
        print(f"  Gold: {gold}", flush=True)
        print(f"  Mean entropy: {diag['mean_entropy']}, Max: {diag['max_entropy']}", flush=True)
        print(f"  Selected positions: {diag['selected_positions']}", flush=True)
        for pos in diag["positions"][:4]:
            top3 = pos["top_tokens"][:3]
            print(f"  Pos {pos['pos']} (H={pos['entropy']:.2f}): {top3}", flush=True)
        print(f"  Candidates:", flush=True)
        for c in diag["candidates"][:6]:
            print(f"    pos={c['position']} seed='{c['seed_token']}' ({c['seed_prob']:.3f}) -> '{c['full_text'][:50]}'", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
