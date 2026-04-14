"""EAMD v2 Guidance Sweep — test gamma multipliers + gated guidance.
Loads model ONCE, pre-computes retrieval ONCE, sweeps configs per question.

Run: python -u src/daes/eamd_v2_sweep.py --model dream --dataset musique --n_questions 50
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, compute_signal_and_scale, build_short_pair, prepare_logits,
    get_mask_id, decode_answer, compute_f1, compute_w_t,
    compute_v2_guidance, short_generate, expand_evidence,
    QUESTION_FILES,
)
from dllm.pipelines.dream.sampler import sample_tokens
import dllm
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModel


SWEEP_CONFIGS = [
    {"name": "pool",          "gamma_mult": 0.0,  "gated": False, "v1": False},
    {"name": "v1_tanh",       "gamma_mult": 1.0,  "gated": False, "v1": True},
    {"name": "v2_canonical",  "gamma_mult": 1.0,  "gated": False, "v1": False},
    {"name": "v2_gated",      "gamma_mult": 1.0,  "gated": True,  "v1": False},
    {"name": "v2_gamma_0.5",  "gamma_mult": 0.5,  "gated": False, "v1": False},
    {"name": "v1_gated",      "gamma_mult": 1.0,  "gated": True,  "v1": True},
]


@torch.inference_mode()
def sweep_denoise(model, tokenizer, question, old_context, new_context,
                  steps=32, n_tokens=6, temperature=0.0, gamma_cap=8.0,
                  gamma_mult=1.0, gated=False, cfg_v1=False):
    """Single denoising run with configurable gamma multiplier and gating.

    If gated=True: at each masked position, only apply guidance when:
      - argmax(p_full) != argmax(p_base)  (evidence changes the prediction)
      - entropy(p_full) < entropy(p_base)  (new evidence reduces uncertainty)
    Otherwise use p_full logits directly (= Pool).
    """
    device = model.device
    mask_id = get_mask_id(tokenizer)

    c1_ids, c0_ids, n_prefix = build_short_pair(tokenizer, new_context, old_context, question, n_tokens)

    x_full = torch.tensor([c1_ids], dtype=torch.long, device=device)
    x_base = torch.tensor([c0_ids], dtype=torch.long, device=device)
    attn_full = torch.ones((1, len(c1_ids)), dtype=torch.long, device=device)
    attn_base = torch.ones((1, len(c0_ids)), dtype=torch.long, device=device)

    remaining = n_tokens
    k_per_step = max(1, math.ceil(n_tokens / steps))
    stats = {"gammas": [], "igs": [], "gated_count": 0, "total_positions": 0}

    for step in range(steps):
        if remaining <= 0:
            break
        masked_local = (x_full[0, n_prefix:n_prefix + n_tokens] == mask_id).nonzero(as_tuple=True)[0]
        if len(masked_local) == 0:
            break

        full_pos = masked_local + n_prefix

        # Batched forward pass
        x_pair = torch.cat([x_base, x_full], dim=0)
        attn_pair = torch.cat([attn_base, attn_full], dim=0)
        out = model(x_pair, attention_mask=attn_pair)
        logits_pair = prepare_logits(out.logits)
        logits_base = logits_pair[0, full_pos]
        logits_full = logits_pair[1, full_pos]

        if gamma_mult == 0.0:
            # Pure Pool: just use C1 logits
            guided = logits_full
        elif cfg_v1:
            # V1 tanh heuristic: lambda_max * tanh(beta * symKL / (H + eps)) * w_t
            _, w_t = compute_w_t(len(masked_local), n_tokens)
            signal, noise, extra, _ = compute_signal_and_scale(
                logits_full, logits_base,
                lambda_max=1.0, beta=1.0, eps=1e-6, schedule=w_t,
            )
            gamma = extra * gamma_mult
            if gated:
                argmax_full = logits_full.argmax(dim=-1)
                argmax_base = logits_base.argmax(dim=-1)
                entropy_full = -(F.softmax(logits_full, dim=-1) * F.log_softmax(logits_full, dim=-1)).sum(dim=-1)
                entropy_base = -(F.softmax(logits_base, dim=-1) * F.log_softmax(logits_base, dim=-1)).sum(dim=-1)
                gate = (argmax_full != argmax_base) & (entropy_full < entropy_base)
                gamma = torch.where(gate, gamma, torch.zeros_like(gamma))
                stats["gated_count"] += gate.sum().item()
                stats["total_positions"] += len(gate)
            guided = logits_full + gamma.unsqueeze(-1) * (logits_full - logits_base)
            stats["gammas"].append(gamma.mean().item())
        else:
            _, w_t = compute_w_t(len(masked_local), n_tokens)
            ig, var, gamma, _ = compute_v2_guidance(
                logits_full, logits_base, w_t=w_t, gamma_cap=gamma_cap,
            )
            gamma = gamma * gamma_mult
            if gated:
                argmax_full = logits_full.argmax(dim=-1)
                argmax_base = logits_base.argmax(dim=-1)
                entropy_full = -(F.softmax(logits_full, dim=-1) * F.log_softmax(logits_full, dim=-1)).sum(dim=-1)
                entropy_base = -(F.softmax(logits_base, dim=-1) * F.log_softmax(logits_base, dim=-1)).sum(dim=-1)
                gate = (argmax_full != argmax_base) & (entropy_full < entropy_base)
                gamma = torch.where(gate, gamma, torch.zeros_like(gamma))
                stats["gated_count"] += gate.sum().item()
                stats["total_positions"] += len(gate)
            guided = logits_full + gamma.unsqueeze(-1) * (logits_full - logits_base)
            stats["gammas"].append(gamma.mean().item())
            stats["igs"].append(ig.mean().item())

        confidence, x0 = sample_tokens(guided, temperature=temperature, neg_entropy=True)
        n_commit = min(k_per_step, remaining)
        if step == steps - 1:
            n_commit = remaining
        n_commit = min(n_commit, len(confidence))
        _, topk = torch.topk(confidence, n_commit)
        chosen = masked_local[topk]
        x_full[0, chosen + n_prefix] = x0[topk]
        x_base[0, chosen + n_prefix] = x0[topk]
        remaining -= len(topk)

    answer_tokens = x_full[0, n_prefix:n_prefix + n_tokens]
    answer = decode_answer(tokenizer, answer_tokens)

    summary_stats = {
        "mean_gamma": sum(stats["gammas"]) / len(stats["gammas"]) if stats["gammas"] else 0.0,
        "mean_ig": sum(stats["igs"]) / len(stats["igs"]) if stats["igs"] else 0.0,
    }
    if gated:
        summary_stats["gate_rate"] = stats["gated_count"] / max(1, stats["total_positions"])

    return answer, summary_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dream", "llada"])
    parser.add_argument("--dataset", default="musique")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--n_questions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--answer_tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_candidates", type=int, default=3)
    parser.add_argument("--initial_top_k", type=int, default=5)
    parser.add_argument("--expand_top_k", type=int, default=3)
    parser.add_argument("--gamma_cap", type=float, default=8.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== EAMD v2 Guidance Sweep ===", flush=True)
    print(f"Model: {args.model}, Steps: {args.steps}, Tokens: {args.answer_tokens}", flush=True)
    print(f"Configs: {[c['name'] for c in SWEEP_CONFIGS]}", flush=True)

    # Load model
    print(f"Loading {args.model}...", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_name = "Dream-org/Dream-v0-Instruct-7B" if args.model == "dream" else "GSAI-ML/LLaDA-8B-Instruct"
    if args.model == "dream":
        model_args = SimpleNamespace(model_name_or_path=model_name)
        model = dllm.utils.get_model(model_args=model_args).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()

    import eamd_v2_wiki18
    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"  Loaded in {time.time() - t_start:.1f}s", flush=True)

    # Load retriever + questions
    retriever = Wiki18Retriever()
    all_questions = json.load(open(QUESTION_FILES[args.dataset]))
    questions = all_questions[args.start_idx:args.start_idx + args.n_questions]
    print(f"Loaded {len(questions)} questions", flush=True)

    # Phase 1: Batch retrieval
    print("Phase 1: Batch retrieval...", flush=True)
    t1 = time.time()
    queries = [f"query: {q['question']}" for q in questions]
    all_initial = retriever.retrieve_batch(queries, args.initial_top_k)
    print(f"  Done in {time.time() - t1:.1f}s", flush=True)

    # Phase 2: Evidence expansion (quick seeds)
    print("Phase 2: Evidence expansion...", flush=True)
    t2 = time.time()
    qdata = []
    for qi, q in enumerate(questions):
        qtext = q["question"]
        gold = q.get("answer") or (q.get("golden_answers") or [""])[0]
        initial = all_initial[qi]
        old_ctx = "\n\n".join(initial)

        seed_ans, _, _ = short_generate(model, tokenizer, old_ctx, qtext,
                                         steps=16, n_tokens=16, temperature=args.temperature)
        expanded, _ = expand_evidence(retriever, qtext, old_ctx, initial,
                                       [seed_ans], args.n_candidates, args.expand_top_k)
        new_ctx = "\n\n".join(expanded)
        qdata.append({"question": qtext, "gold": gold, "old_context": old_ctx,
                       "new_context": new_ctx, "qid": q.get("qid") or q.get("id", f"dev_{qi}")})
        if (qi + 1) % 10 == 0:
            print(f"  Expanded {qi+1}/{len(questions)}", flush=True)
    print(f"  Done in {time.time() - t2:.1f}s", flush=True)

    # Phase 3: Sweep
    print(f"Phase 3: Guidance sweep ({len(SWEEP_CONFIGS)} configs x {len(qdata)} questions)...", flush=True)
    t3 = time.time()

    config_totals = {c["name"]: {"f1": 0.0, "em": 0.0, "contain": 0.0, "answers": []} for c in SWEEP_CONFIGS}
    results = []

    for qi, qd in enumerate(qdata):
        row = {"id": qd["qid"], "question": qd["question"], "gold": qd["gold"]}
        for cfg in SWEEP_CONFIGS:
            ans, stats = sweep_denoise(
                model, tokenizer, qd["question"], qd["old_context"], qd["new_context"],
                steps=args.steps, n_tokens=args.answer_tokens, temperature=args.temperature,
                gamma_cap=args.gamma_cap, gamma_mult=cfg["gamma_mult"], gated=cfg["gated"], cfg_v1=cfg.get("v1", False),
            )
            f1_result = compute_f1(ans, qd["gold"])
            f1 = f1_result if isinstance(f1_result, (int, float)) else f1_result[0]
            em = float(ans.strip().lower() == qd["gold"].strip().lower())
            contain = float(qd["gold"].strip().lower() in ans.strip().lower())
            config_totals[cfg["name"]]["f1"] += f1
            config_totals[cfg["name"]]["em"] += em
            config_totals[cfg["name"]]["contain"] += contain
            row[cfg["name"]] = {"answer": ans, "f1": round(f1, 4), "em": em, "stats": stats}

        results.append(row)
        if (qi + 1) % 10 == 0 or qi == 0 or qi == len(qdata) - 1:
            print(f"  [{qi+1}/{len(qdata)}] {qd['qid']}", flush=True)

    n = len(qdata)
    t_sweep = time.time() - t3
    t_total = time.time() - t_start

    # Summary
    print(f"\nSweep: {t_sweep:.1f}s ({t_sweep/n:.1f}s/q)")
    print(f"Total: {t_total:.1f}s")
    print(f"\n{'Config':<20s} {'F1':>6s} {'EM':>6s} {'Contain':>8s}")
    print("-" * 42)
    summary = {}
    for cfg in SWEEP_CONFIGS:
        name = cfg["name"]
        s = {k: round(v / n, 4) for k, v in config_totals[name].items() if k != "answers"}
        summary[name] = s
        print(f"{name:<20s} {s['f1']:>6.3f} {s['em']:>6.3f} {s['contain']:>8.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "results": results, "config": vars(args),
                    "timing": {"sweep_sec": t_sweep, "total_sec": t_total}}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
