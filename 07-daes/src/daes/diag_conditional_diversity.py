"""Diagnostic: Is LLaDAs conditional posterior more diverse than its marginal?"""
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

@torch.inference_mode()
def full_denoise(model, tokenizer, context, question, steps=32, n_tokens=32):
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
        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=_neg_entropy())
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)
    return x, n_prefix, n_tokens

def find_entity_spans(tokenizer, answer_ids, min_len=1, max_len=4):
    spans = []
    tokens = answer_ids.tolist()
    texts = [tokenizer.decode([t]).strip() for t in tokens]
    stopwords = set("the a an is was are were of in to and or not no yes".split())
    stopwords.update([",", ".", "!", "?", "'", "-", ":", ";", "(", ")"])
    i = 0
    while i < len(texts):
        if texts[i].lower() in stopwords or len(texts[i]) < 2:
            i += 1
            continue
        j = i + 1
        while j < min(i + max_len, len(texts)):
            if texts[j].lower() in stopwords or len(texts[j]) < 1:
                break
            j += 1
        if j - i >= min_len:
            span_text = tokenizer.decode(tokens[i:j]).strip()
            if len(span_text) > 1:
                spans.append({"start": i, "end": j, "text": span_text})
        i = j if j > i + 1 else i + 1
    return spans

@torch.inference_mode()
def diagnose_question(model, tokenizer, context, question, steps=32, n_tokens=32):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)

    canvas_marginal = prefix_ids + [mask_id] * n_tokens
    x_marginal = torch.tensor([canvas_marginal], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas_marginal)), dtype=torch.long, device=device)
    out_marginal = model(x_marginal, attention_mask=attn)
    logits_marginal = prepare_logits(out_marginal.logits)
    probs_marginal = torch.softmax(logits_marginal[0, n_prefix:n_prefix + n_tokens], dim=-1)
    entropy_marginal = -(probs_marginal * torch.log(probs_marginal + 1e-10)).sum(dim=-1)

    x_committed, _, _ = full_denoise(model, tokenizer, context, question, steps, n_tokens)
    answer_ids = x_committed[0, n_prefix:n_prefix + n_tokens]
    answer_text = decode_answer(tokenizer, answer_ids)
    entity_spans = find_entity_spans(tokenizer, answer_ids)

    span_results = []
    for span in entity_spans[:5]:
        x_remasked = x_committed.clone()
        for pos in range(span["start"], span["end"]):
            x_remasked[0, n_prefix + pos] = mask_id
        out_cond = model(x_remasked, attention_mask=attn)
        logits_cond = prepare_logits(out_cond.logits)
        span_diag = {"span_text": span["text"], "start": span["start"], "end": span["end"], "positions": []}
        for pos in range(span["start"], span["end"]):
            probs_cond = torch.softmax(logits_cond[0, n_prefix + pos], dim=-1)
            h_cond = -(probs_cond * torch.log(probs_cond + 1e-10)).sum().item()
            h_marg = entropy_marginal[pos].item()
            top_p_c, top_i_c = torch.topk(probs_cond, 5)
            top_tok_c = [(tokenizer.decode([t.item()]).strip(), round(p.item(), 4)) for t, p in zip(top_i_c, top_p_c)]
            top_p_m, top_i_m = torch.topk(probs_marginal[pos], 5)
            top_tok_m = [(tokenizer.decode([t.item()]).strip(), round(p.item(), 4)) for t, p in zip(top_i_m, top_p_m)]
            orig_tok = tokenizer.decode([answer_ids[pos].item()]).strip()
            orig_p = probs_cond[answer_ids[pos]].item()
            span_diag["positions"].append({
                "pos": pos, "orig_token": orig_tok, "orig_prob_cond": round(orig_p, 4),
                "H_marg": round(h_marg, 3), "H_cond": round(h_cond, 3),
                "ratio": round(h_cond / max(h_marg, 1e-6), 2),
                "top5_cond": top_tok_c, "top5_marg": top_tok_m,
            })
        span_results.append(span_diag)
    return {"answer": answer_text, "mean_H_marg": round(entropy_marginal.mean().item(), 3),
            "n_spans": len(entity_spans), "spans": span_results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--n_questions", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    t0 = time.time()
    print(f"=== Conditional Diversity Diagnostic ({args.model}) ===", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.model == "dream":
        ma = SimpleNamespace(model_name_or_path="Dream-org/Dream-v0-Instruct-7B")
        model = dllm.utils.get_model(model_args=ma).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=ma)
    else:
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    retriever = Wiki18Retriever()
    questions = json.load(open(QUESTION_FILES[args.dataset]))[:args.n_questions]
    all_initial = retriever.retrieve_batch([f"query: {q['question']}" for q in questions], 5)
    print(f"Loaded in {time.time() - t0:.1f}s", flush=True)
    results = []
    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        context = "\n\n".join(all_initial[qi])
        diag = diagnose_question(model, tokenizer, context, qtext)
        diag["question"] = qtext
        diag["gold"] = gold
        results.append(diag)
        print(f"\n[{qi+1}/{len(questions)}] {qtext[:70]}", flush=True)
        print(f"  Gold: {gold} | Answer: {diag['answer'][:50]}", flush=True)
        for sp in diag["spans"][:3]:
            for p in sp["positions"]:
                print(f"  '{sp['span_text']}' p{p['pos']}: H_m={p['H_marg']:.3f} H_c={p['H_cond']:.3f} ({p['ratio']:.1f}x) orig='{p['orig_token']}'({p['orig_prob_cond']:.2f}) top3_c={p['top5_cond'][:3]}", flush=True)
    all_r = [p["ratio"] for r in results for sp in r["spans"] for p in sp["positions"] if p["H_marg"] > 0.01]
    print(f"\n{'='*60}", flush=True)
    if all_r:
        import statistics
        print(f"H_cond/H_marg: mean={statistics.mean(all_r):.2f} median={statistics.median(all_r):.2f} min={min(all_r):.2f} max={max(all_r):.2f}", flush=True)
        print(f"Cond MORE diverse: {sum(1 for r in all_r if r > 1)}/{len(all_r)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"Done in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
