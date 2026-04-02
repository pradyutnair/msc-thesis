"""Annealed Branch-and-Repair Denoising (ABRD) Pilot.

4 arms:
1. baseline: standard confidence-based denoising
2. taps: TAPS perturbation (early semantic branching)
3. p2: P2-Self remasking (consistency-based error correction)
4. abrd: TAPS + P2 combined

All arms use the same fixed retrieval pool (top-5 passages from original question).
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    _neg_entropy, Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, decode_answer, compute_f1, QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel
import eamd_v2_wiki18


@torch.inference_mode()
def abrd_decode(model, tokenizer, context, question, steps=32, n_tokens=32,
                use_taps=False, use_p2=False,
                taps_sigma=1.0, taps_anneal_frac=0.5,
                p2_check_step_frac=0.5, p2_remask_frac=0.3):
    """Decode with optional TAPS perturbation and P2-Self remasking."""
    device = model.device
    mask_id = get_mask_id(tokenizer)
    neg_ent = _neg_entropy()

    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    p2_check_step = int(steps * p2_check_step_frac)
    p2_triggered = False
    remask_count = 0

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        out = model(x, attention_mask=attn)
        logits = prepare_logits(out.logits)
        token_logits = logits[0, masked_local + n_prefix]

        # TAPS: add decaying perturbation to logits at early steps
        if use_taps:
            progress = step / max(1, steps - 1)
            if progress < taps_anneal_frac:
                noise_scale = taps_sigma * (1.0 - progress / taps_anneal_frac)
                noise = torch.randn_like(token_logits) * noise_scale
                token_logits = token_logits + noise

        confidence, x0 = sample_tokens(token_logits, temperature=0.0, neg_entropy=neg_ent)

        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        x[0, masked_local[topk] + n_prefix] = x0[topk]
        remaining -= len(topk)

        # P2-Self: at the check step, evaluate committed tokens and remask inconsistent ones
        if use_p2 and step == p2_check_step and not p2_triggered:
            p2_triggered = True
            committed_local = (x[0, n_prefix:n_prefix + n_tokens] != mask_id).nonzero(as_tuple=True)[0]
            if len(committed_local) > 2:
                # Forward pass to check consistency
                out2 = model(x, attention_mask=attn)
                logits2 = prepare_logits(out2.logits)
                # For each committed position, check if model still agrees
                inconsistency = []
                for cl in committed_local:
                    pos_global = n_prefix + cl
                    committed_token = x[0, pos_global].item()
                    predicted_token = logits2[0, pos_global].argmax().item()
                    if committed_token != predicted_token:
                        # Model changed its mind — compute score
                        logit_diff = (logits2[0, pos_global, predicted_token] - logits2[0, pos_global, committed_token]).item()
                        inconsistency.append((cl.item(), logit_diff))

                if inconsistency:
                    # Sort by inconsistency magnitude, remask top fraction
                    inconsistency.sort(key=lambda x: x[1], reverse=True)
                    n_remask = max(1, int(len(committed_local) * p2_remask_frac))
                    n_remask = min(n_remask, len(inconsistency))
                    for i in range(n_remask):
                        pos_local = inconsistency[i][0]
                        x[0, n_prefix + pos_local] = mask_id
                        remaining += 1
                        remask_count += 1

    answer = decode_answer(tokenizer, x[0, n_prefix:n_prefix + n_tokens])
    return answer, {"remask_count": remask_count, "p2_triggered": p2_triggered}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--taps_sigma", type=float, default=1.0)
    parser.add_argument("--taps_anneal_frac", type=float, default=0.5)
    parser.add_argument("--p2_check_frac", type=float, default=0.5)
    parser.add_argument("--p2_remask_frac", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== ABRD Pilot ({args.model}) ===", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == "dream":
        ma = SimpleNamespace(model_name_or_path="Dream-org/Dream-v0-Instruct-7B")
        model = dllm.utils.get_model(model_args=ma).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=ma)
    else:
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).cuda().eval()
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    query_texts = ["query: " + q["question"] for q in questions]
    all_initial = retriever.retrieve_batch(query_texts, 5)
    print(f"Loaded {len(questions)} questions in {time.time() - t0:.1f}s", flush=True)

    methods = ["baseline", "taps", "p2", "abrd"]
    totals = {m: {"f1": 0.0, "em": 0.0, "contain": 0.0} for m in methods}
    results = []

    for qi, q in enumerate(questions):
        tq = time.time()
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        context = "\n\n".join(all_initial[qi])

        row = {"id": q.get("id", "dev_" + str(qi)), "question": qtext, "gold": gold}

        for method in methods:
            use_taps = method in ("taps", "abrd")
            use_p2 = method in ("p2", "abrd")
            ans, stats = abrd_decode(
                model, tokenizer, context, qtext,
                use_taps=use_taps, use_p2=use_p2,
                taps_sigma=args.taps_sigma, taps_anneal_frac=args.taps_anneal_frac,
                p2_check_step_frac=args.p2_check_frac, p2_remask_frac=args.p2_remask_frac,
            )
            f1_result = compute_f1(ans, gold)
            f1 = f1_result[2] if isinstance(f1_result, tuple) else f1_result
            em = float(ans.strip().lower() == gold.strip().lower())
            contain = float(gold.strip().lower() in ans.strip().lower())
            totals[method]["f1"] += f1
            totals[method]["em"] += em
            totals[method]["contain"] += contain
            row[method] = {"answer": ans[:80], "f1": round(f1, 4), "em": em, "contain": contain}
            row[method + "_stats"] = stats

        elapsed = time.time() - tq
        results.append(row)

        if (qi + 1) % 5 == 0 or qi == 0 or qi == len(questions) - 1:
            n_done = qi + 1
            line = " | ".join(str(m) + "=" + format(totals[m]["f1"]/n_done, ".3f") for m in methods)
            print(f"[{n_done}/{len(questions)}] ({elapsed:.1f}s) {line}", flush=True)
            print(f"  Gold: {gold}", flush=True)
            for m in methods:
                ans_short = row[m]["answer"][:60]
                print(f"  {m:8s}: {ans_short}", flush=True)

    n = len(results)
    summary = {m: {k: round(v / max(1, n), 4) for k, v in totals[m].items()} for m in methods}
    print(chr(10) + "=" * 60, flush=True)
    print(f"ABRD Pilot | {args.model} | {args.dataset} | N={n}", flush=True)
    for m in methods:
        s = summary[m]
        delta = s["f1"] - summary["baseline"]["f1"]
        sf1,sem,sco = s["f1"],s["em"],s["contain"]; print(f"  {m:8s} F1={sf1:.4f} EM={sem:.4f} contain={sco:.4f} delta={delta:+.4f}", flush=True)
    print(f"Total: {time.time() - t0:.1f}s ({(time.time() - t0) / max(1, n):.1f}s/q)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                   "timing": {"total_sec": round(time.time() - t0, 1)}}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
