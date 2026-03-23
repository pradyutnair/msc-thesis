"""
Reproduce SPREAD Table 1 baseline: Dream-7B MuSiQue F1=30.56, CR=77.65.
Test different prompt formats to find what produces long, context-copying outputs.
"""
import json, sys, torch, pickle, numpy as np, re, string, time
import torch.nn.functional as F
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

# --- Retriever ---
with open("/projects/prjs1800/external/arag/data/musique/index_e5_musique_full/sentence_index.pkl", "rb") as f:
    idx = pickle.load(f)
st = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

def retrieve(q, top_k=5):
    q_emb = st.encode([q], normalize_embeddings=True)[0]
    sims = np.dot(idx["embeddings"], q_emb)
    top = np.argsort(sims)[::-1][:top_k*3]
    cb = {}
    for i in top:
        cid = idx["sentence_to_chunk"][i]
        if cid not in cb or sims[i] > cb[cid]: cb[cid] = float(sims[i])
    ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [idx["chunks"][cid]["text"][:2000] for cid, _ in ranked]

# --- Model ---
@dataclass
class MA:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
model = dllm.utils.get_model(model_args=MA()).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=MA())
sampler = DreamSampler(model=model, tokenizer=tokenizer)
mask_id = tokenizer.mask_token_id

# --- Metrics ---
def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def f1_score(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return 0, 0, 0
    common = set(pt) & set(gt)
    if not common: return 0, 0, 0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return p, r, 2*p*r/(p+r)

def copy_rate(pred, context):
    """Fraction of prediction words that appear in the context."""
    pred_words = set(normalize(pred).split())
    ctx_words = set(normalize(context).split())
    if not pred_words: return 0
    return len(pred_words & ctx_words) / len(pred_words)

# --- Generation variants ---
def generate_with_dllm_sample(context, question, use_chat_template=True):
    """Use dllm's native sample() method."""
    if use_chat_template:
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    else:
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        inputs = tokenizer.encode(prompt, add_special_tokens=False)

    config = DreamSamplerConfig(steps=128, max_new_tokens=512, temperature=0.1, alg="entropy")
    output = sampler.sample([inputs], config)
    if hasattr(output, 'sequences'):
        seq = output.sequences[0]
    else:
        seq = output[0]
    gen = seq[len(inputs):]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def generate_with_infill(context, question, use_chat_template=True, n_mask=512, steps=128):
    """Use infill() — our custom denoising loop."""
    if use_chat_template:
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    else:
        prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False)

    canvas = prefix_ids + [mask_id] * n_mask
    config = DreamSamplerConfig(steps=steps, temperature=0.1, alg="entropy")
    output = sampler.infill([canvas], config)
    if hasattr(output, 'sequences'):
        seq = output.sequences[0]
    else:
        seq = output[0]
    gen = seq[len(prefix_ids):]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# --- Run experiments ---
qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[:20]
print(f"Testing 4 generation variants on 20 MuSiQue questions", flush=True)
print("=" * 70, flush=True)

variants = [
    ("sample+chat", lambda ctx, q: generate_with_dllm_sample(ctx, q, use_chat_template=True)),
    ("sample+raw", lambda ctx, q: generate_with_dllm_sample(ctx, q, use_chat_template=False)),
    ("infill+chat", lambda ctx, q: generate_with_infill(ctx, q, use_chat_template=True)),
    ("infill+raw", lambda ctx, q: generate_with_infill(ctx, q, use_chat_template=False)),
]

for vname, gen_fn in variants:
    sum_f1, sum_cr, sum_words = 0, 0, 0
    print(f"\n--- {vname} ---", flush=True)
    for i, q in enumerate(qs):
        passages = retrieve(q["question"])
        context = "\n\n".join(passages)
        t0 = time.time()
        try:
            answer = gen_fn(context, q["question"])
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            answer = ""
        elapsed = time.time() - t0
        _, _, f = f1_score(answer, q["answer"])
        cr = copy_rate(answer, context)
        words = len(answer.split())
        sum_f1 += f; sum_cr += cr; sum_words += words
        if i < 3:
            print(f"  [{i+1}] ({elapsed:.1f}s) F1={f:.2f} CR={cr:.2f} words={words}", flush=True)
            print(f"    Gold: {q['answer']}", flush=True)
            print(f"    Pred: {answer[:150]}", flush=True)

    n = len(qs)
    print(f"  SUMMARY: F1={sum_f1/n*100:.1f}%  CR={sum_cr/n*100:.1f}%  Avg words={sum_words/n:.0f}", flush=True)
    print(f"  TARGET:  F1=30.6%  CR=77.7%", flush=True)

print("\n" + "=" * 70, flush=True)
print("Done!", flush=True)
