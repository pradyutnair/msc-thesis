"""Logit lens diagnostic: do intermediate layers expose bridge entities on LLaDA?"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/src/daes")

from eamd_v2_wiki18 import (
    Wiki18Retriever, build_short_prompt, prepare_logits,
    get_mask_id, QUESTION_FILES,
)
import eamd_v2_wiki18
from transformers import AutoTokenizer, AutoModel

MUSIQUE_DEV = "/projects/prjs1800/datasets/musique/musique_full_v1.0_dev.jsonl"

def load_musique(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize_answer(s):
    import re, string
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def get_gold_bridges(example):
    gold = normalize_answer(example.get("answer", ""))
    decomposition = example.get("question_decomposition") or []
    answers, seen = [], set()
    for step in decomposition:
        ans = (step.get("answer") or "").strip()
        if not ans:
            continue
        key = normalize_answer(ans)
        if key in seen:
            continue
        seen.add(key)
        answers.append(ans)
    bridges = [a for a in answers if normalize_answer(a) != gold]
    if not bridges and len(answers) > 1:
        bridges = answers[:-1]
    return bridges

@torch.inference_mode()
def logit_lens_diagnostic(model, tokenizer, context, question, gold_bridges,
                          layers_to_probe=None, n_tokens=32):
    device = model.device
    mask_id = get_mask_id(tokenizer)
    prefix_ids, n_prefix = build_short_prompt(tokenizer, context, question)
    canvas = prefix_ids + [mask_id] * n_tokens
    x = torch.tensor([canvas], dtype=torch.long, device=device)
    attn = torch.ones((1, len(canvas)), dtype=torch.long, device=device)

    # Forward pass collecting hidden states from all layers
    out = model(x, attention_mask=attn, output_hidden_states=True)

    # Get unembedding matrix
    if hasattr(model, "lm_head"):
        unembed = model.lm_head.weight  # [V, D]
    elif hasattr(model, "model") and hasattr(model.model, "transformer"):
        unembed = model.model.transformer.ff_out.weight
    else:
        # Try to find it
        for name, param in model.named_parameters():
            if "lm_head" in name or "ff_out" in name or "embed_tokens" in name:
                if param.shape[0] > 10000:  # vocab-sized
                    unembed = param
                    break

    hidden_states = out.hidden_states  # tuple of [1, seq_len, D] for each layer
    n_layers = len(hidden_states) - 1  # exclude embedding layer

    if layers_to_probe is None:
        # Probe: layer 0 (embed), L/4, L/2, 3L/4, L (final)
        layers_to_probe = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]

    # Tokenize gold bridges for matching
    bridge_tokens = set()
    for b in gold_bridges:
        toks = tokenizer.encode(b, add_special_tokens=False)
        for t in toks:
            bridge_tokens.add(t)
        # Also add lowercased version
        toks_lower = tokenizer.encode(b.lower(), add_special_tokens=False)
        for t in toks_lower:
            bridge_tokens.add(t)

    layer_results = []
    for layer_idx in layers_to_probe:
        h = hidden_states[layer_idx][0]  # [seq_len, D]
        # Project through unembedding
        logits = h @ unembed.T  # [seq_len, V]
        # Look at answer positions
        answer_logits = logits[n_prefix:n_prefix + min(6, n_tokens)]  # first 6 positions
        probs = torch.softmax(answer_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        pos_data = []
        for pos in range(min(6, n_tokens)):
            top_p, top_i = torch.topk(probs[pos], 10)
            top_tokens = [(tokenizer.decode([t.item()]).strip(), round(p.item(), 4)) for t, p in zip(top_i, top_p)]

            # Check if any bridge token appears in top-20
            top20_ids = torch.topk(probs[pos], 20).indices.tolist()
            bridge_in_top20 = [tokenizer.decode([t]).strip() for t in top20_ids if t in bridge_tokens]

            pos_data.append({
                "pos": pos,
                "entropy": round(entropy[pos].item(), 3),
                "top5": top_tokens[:5],
                "bridge_in_top20": bridge_in_top20,
            })

        layer_results.append({
            "layer": layer_idx,
            "layer_frac": round(layer_idx / max(1, n_layers), 2),
            "mean_entropy": round(entropy.mean().item(), 3),
            "positions": pos_data,
        })

    return {"n_layers": n_layers, "layers": layer_results, "gold_bridges": gold_bridges}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llada", choices=["dream", "llada"])
    parser.add_argument("--n_questions", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== Logit Lens Diagnostic ({args.model}) ===", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.model == "llada":
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
        model = AutoModel.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
                                          torch_dtype=torch.bfloat16).cuda().eval()
    else:
        import dllm
        from types import SimpleNamespace
        ma = SimpleNamespace(model_name_or_path="Dream-org/Dream-v0-Instruct-7B")
        model = dllm.utils.get_model(model_args=ma).eval()
        tokenizer = dllm.utils.get_tokenizer(model_args=ma)

    eamd_v2_wiki18.MODEL_REF = model
    eamd_v2_wiki18.TOKENIZER_REF = tokenizer
    eamd_v2_wiki18.MODEL_TYPE_REF = args.model
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    retriever = Wiki18Retriever()

    # Use MuSiQue for gold bridges
    musique = load_musique(MUSIQUE_DEV)[:args.n_questions]
    # Get wiki18 questions for retrieval
    wiki18_qs = json.load(open(QUESTION_FILES["musique"]))[:args.n_questions]
    query_texts = [f"query: {q['question']}" for q in wiki18_qs]
    all_initial = retriever.retrieve_batch(query_texts, 5)

    results = []
    for qi in range(min(len(musique), len(wiki18_qs))):
        q_musique = musique[qi]
        q_wiki18 = wiki18_qs[qi]
        qtext = q_wiki18["question"]
        gold = q_wiki18.get("answer") or ""
        gold_bridges = get_gold_bridges(q_musique)
        context = "\n\n".join(all_initial[qi])

        print(f"\n[{qi+1}] {qtext[:60]}", flush=True)
        print(f"  Gold: {gold} | Bridges: {gold_bridges}", flush=True)

        diag = logit_lens_diagnostic(model, tokenizer, context, qtext, gold_bridges)
        diag["question"] = qtext
        diag["gold"] = gold
        results.append(diag)

        for lr in diag["layers"]:
            bridge_hits = []
            for p in lr["positions"][:3]:
                if p["bridge_in_top20"]:
                    bridge_hits.extend(p["bridge_in_top20"])
            layer_num = lr["layer"]
            n_lay = diag["n_layers"]
            lfrac = lr["layer_frac"]
            mh = lr["mean_entropy"]
            p0t3 = lr["positions"][0]["top5"][:3]
            print(f"  Layer {layer_num}/{n_lay} ({lfrac:.0%}): mean_H={mh:.2f} p0_top3={p0t3} bridge_hits={bridge_hits if bridge_hits else []}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"\nDone in {time.time() - t0:.1f}s. Saved to {args.output}")

if __name__ == "__main__":
    main()
