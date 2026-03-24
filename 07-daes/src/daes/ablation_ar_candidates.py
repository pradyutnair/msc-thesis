"""AR candidate comparison: are dLLM candidates better than AR model candidates?
Compares: dLLM candidates (Dream-7B distribution) vs AR candidates (Qwen3-8B generation).
Both use E5 retrieval, same pipeline."""
import argparse, json, sys, time, re, string, pickle, random, numpy as np, torch
sys.path.insert(0, os.environ.get("DLLM_PATH", "dllm"))
import os
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer as ARTokenizer
from dataclasses import dataclass

random.seed(42)

# --- Config ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/projects/prjs1800/external/arag/data")
parser.add_argument("--index_path", default=None)
parser.add_argument("--n_questions", type=int, default=50)
args = parser.parse_args()

DATA_DIR = args.data_dir
INDEX_PATH = args.index_path or os.path.join(DATA_DIR, "musique/index_e5_musique_full/sentence_index.pkl")
QUESTIONS_PATH = os.path.join(DATA_DIR, "musique/questions.json")

# --- Retriever ---
with open(INDEX_PATH, "rb") as f:
    idx = pickle.load(f)
e5 = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

def retrieve(query, top_k=5):
    q_emb = e5.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(idx["embeddings"], q_emb)
    top = np.argsort(sims)[::-1][:top_k * 3]
    cb = {}
    for i in top:
        cid = idx["sentence_to_chunk"][i]
        if cid not in cb or sims[i] > cb[cid]:
            cb[cid] = float(sims[i])
    ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [idx["chunks"][cid]["text"][:2000] for cid, _ in ranked]

# --- Dream-7B ---
@dataclass
class MA:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
dream_model = dllm.utils.get_model(model_args=MA()).eval()
dream_tokenizer = dllm.utils.get_tokenizer(model_args=MA())
dream_sampler = DreamSampler(model=dream_model, tokenizer=dream_tokenizer)
dream_config = DreamSamplerConfig(steps=128, max_new_tokens=512, temperature=0.1, alg="entropy", return_dict=True)
mask_id = dream_tokenizer.mask_token_id
print("Dream-7B loaded.", flush=True)

# --- Qwen3-8B (AR) ---
ar_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto")
ar_tokenizer = ARTokenizer.from_pretrained("Qwen/Qwen3-8B")
print("Qwen3-8B loaded.", flush=True)

# --- Metrics ---
def normalize(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())
def f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt: return 0
    common = set(pt) & set(gt)
    if not common: return 0
    p, r = len(common)/len(pt), len(common)/len(gt)
    return 2*p*r/(p+r)

# --- Dream generation ---
def dream_generate(context, question):
    prompt = context + "\n\nQuestion: " + question + "\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    inputs = dream_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    output = dream_sampler.sample([inputs], dream_config)
    return dream_tokenizer.decode(output.sequences[0][len(inputs):], skip_special_tokens=True).strip()

# --- dLLM candidates ---
def get_dllm_candidates(context, question, n=3):
    prompt = context + "\n\nQuestion: " + question + "\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = dream_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = dream_tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    canvas = prefix_ids + [mask_id] * 20
    x = torch.tensor([canvas], dtype=torch.long, device=dream_model.device)
    attn = torch.ones_like(x)
    with torch.no_grad():
        out = dream_model(x, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    probs = torch.softmax(logits[0, n_prefix] / 0.3, dim=-1)
    top_probs, top_ids = torch.topk(probs, n * 3)
    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        x_c = torch.tensor([canvas], dtype=torch.long, device=dream_model.device)
        x_c[0, n_prefix] = tid
        rem = 19
        for step in range(16):
            if rem <= 0: break
            mi = (x_c == mask_id)
            if not mi.any(): break
            with torch.no_grad():
                o2 = dream_model(x_c, attention_mask=attn)
            l2 = torch.cat([o2.logits[:, :1], o2.logits[:, :-1]], dim=1)
            mp = mi[0].nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=True)
            k = min(max(1, rem // 16), rem)
            if step == 15: k = rem
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            rem -= len(tk)
        text = dream_tokenizer.decode(x_c[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append(text)
            if len(candidates) >= n: break
    return candidates

# --- AR candidates ---
def get_ar_candidates(context, question, n=3):
    """Generate n candidate answers using Qwen3-8B (AR model)."""
    prompt = context + "\n\nQuestion: " + question + "\n\nAnswer briefly:"
    messages = [{"role": "user", "content": prompt}]
    input_text = ar_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = ar_tokenizer.encode(input_text, return_tensors="pt").to(ar_model.device)

    candidates = []
    seen = set()
    for _ in range(n * 2):
        with torch.no_grad():
            output = ar_model.generate(
                input_ids, max_new_tokens=30, temperature=0.7,
                do_sample=True, top_p=0.9, pad_token_id=ar_tokenizer.eos_token_id,
            )
        text = ar_tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Extract first phrase
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append(text)
            if len(candidates) >= n:
                break
    return candidates

# --- Run pipeline ---
def run_branch(question, candidates):
    passages = retrieve(question, top_k=5)
    context = "\n\n".join(passages)
    if not candidates:
        return dream_generate(context, question)
    c = random.choice(candidates)
    hop2 = retrieve(question + " " + c, top_k=3)
    expanded = context + "\n\n" + "\n\n".join(hop2)
    return dream_generate(expanded, question)

# --- Main ---
qs = json.load(open(QUESTIONS_PATH))[:args.n_questions]
out_path = "results/ablation_ar_candidates.jsonl"
os.makedirs("results", exist_ok=True)
open(out_path, "w").close()

for mode in ["no_branch", "dllm_candidates", "ar_candidates"]:
    sf, cn = 0, 0
    for i, q in enumerate(qs):
        t0 = time.time()
        passages = retrieve(q["question"], top_k=5)
        context = "\n\n".join(passages)
        if mode == "no_branch":
            answer = dream_generate(context, q["question"])
        elif mode == "dllm_candidates":
            cands = get_dllm_candidates(context, q["question"])
            answer = run_branch(q["question"], cands)
        elif mode == "ar_candidates":
            cands = get_ar_candidates(context, q["question"])
            answer = run_branch(q["question"], cands)
        elapsed = time.time() - t0
        sc = f1(answer, q["answer"])
        sf += sc
        if q["answer"].lower() in answer.lower():
            cn += 1
        result = {"mode": mode, "id": q["id"], "question": q["question"],
                  "gold_answer": q["answer"], "pred_answer": answer,
                  "f1": round(sc, 4), "contain": q["answer"].lower() in answer.lower(),
                  "time": round(elapsed, 1)}
        with open(out_path, "a") as _fw:
            _fw.write(json.dumps(result) + "\n")
        if i < 3 or i == args.n_questions - 1:
            print("[%s %d/%d] (%.1fs) F1=%.2f %s" % (mode, i+1, args.n_questions, elapsed, sc, answer[:50]), flush=True)
    print("\n%s: F1=%.1f%% Contain=%d/%d=%.1f%%\n" % (mode, sf/len(qs)*100, cn, len(qs), cn/len(qs)*100), flush=True)
