"""Candidate source ablation: are dLLM candidates special?
1. dllm_candidates: top-3 from Dream-7B distribution (our method)
2. question_entities: extract entities from the question itself
3. random_words: random words from vocabulary
4. no_branch: baseline top-5 (control)
"""
import argparse, json, sys, time, re, string, pickle, random, numpy as np, torch
sys.path.insert(0, "/projects/prjs1800/msc-thesis/07-daes/dllm")
import dllm
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig, sample_tokens
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

random.seed(42)

with open("/projects/prjs1800/external/arag/data/musique/index_e5_musique_full/sentence_index.pkl", "rb") as f:
    idx = pickle.load(f)
e5 = SentenceTransformer("intfloat/e5-base-v2", device="cpu")

def retrieve(query, top_k=5):
    q_emb = e5.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(idx["embeddings"], q_emb)
    top = np.argsort(sims)[::-1][:top_k * 3]
    cb = {}
    for i in top:
        cid = idx["sentence_to_chunk"][i]
        if cid not in cb or sims[i] > cb[cid]: cb[cid] = float(sims[i])
    ranked = sorted(cb.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [idx["chunks"][cid]["text"][:2000] for cid, _ in ranked]

@dataclass
class MA:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
model = dllm.utils.get_model(model_args=MA()).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=MA())
sampler = DreamSampler(model=model, tokenizer=tokenizer)
config = DreamSamplerConfig(steps=128, max_new_tokens=512, temperature=0.1, alg="entropy", return_dict=True)
mask_id = tokenizer.mask_token_id
print("Ready.", flush=True)

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

def generate(context, question):
    prompt = context + "\n\nQuestion: " + question + "\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    output = sampler.sample([inputs], config)
    return tokenizer.decode(output.sequences[0][len(inputs):], skip_special_tokens=True).strip()

def get_dllm_candidates(context, question, n=3):
    """Our method: read top-k from dLLM distribution."""
    prompt = context + "\n\nQuestion: " + question + "\n\nThe answer is:"
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(input_text, add_special_tokens=False)
    n_prefix = len(prefix_ids)
    canvas = prefix_ids + [mask_id] * 20
    x = torch.tensor([canvas], dtype=torch.long, device=model.device)
    attn = torch.ones_like(x)
    with torch.no_grad():
        out = model(x, attention_mask=attn)
    logits = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
    probs = torch.softmax(logits[0, n_prefix] / 0.3, dim=-1)
    top_probs, top_ids = torch.topk(probs, n * 3)
    candidates = []
    seen = set()
    for prob, tid in zip(top_probs, top_ids):
        x_c = torch.tensor([canvas], dtype=torch.long, device=model.device)
        x_c[0, n_prefix] = tid
        rem = 19
        for step in range(16):
            if rem <= 0: break
            mi = (x_c == mask_id)
            if not mi.any(): break
            with torch.no_grad():
                o2 = model(x_c, attention_mask=attn)
            l2 = torch.cat([o2.logits[:, :1], o2.logits[:, :-1]], dim=1)
            mp = mi[0].nonzero(as_tuple=True)[0]
            c2, x02 = sample_tokens(l2[0, mp], temperature=0.1, neg_entropy=True)
            k = min(max(1, rem // 16), rem)
            if step == 15: k = rem
            _, tk = torch.topk(c2, min(k, len(c2)))
            x_c[0, mp[tk]] = x02[tk]
            rem -= len(tk)
        text = tokenizer.decode(x_c[0, n_prefix:].tolist(), skip_special_tokens=True).strip()
        text = text.split("\n")[0].split(". ")[0].strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            candidates.append(text)
            if len(candidates) >= n: break
    return candidates

def get_question_entities(question):
    """Extract content words from question as pseudo-candidates."""
    stop = {"what", "when", "where", "who", "which", "how", "is", "are", "was",
            "were", "the", "a", "an", "of", "in", "on", "at", "to", "for", "by",
            "with", "from", "that", "this", "did", "does", "do", "has", "have",
            "had", "be", "been", "being", "it", "its", "and", "or", "but", "not",
            "no", "if", "than", "then", "so", "as", "can", "could", "would",
            "should", "will", "shall", "may", "might", "must", "about", "into",
            "after", "before", "between", "during", "called", "known", "named"}
    words = [w for w in question.split() if w.lower().strip("?,.'\"") not in stop and len(w) > 2]
    # Take up to 3 content words/phrases
    candidates = []
    for w in words[:6]:
        w = w.strip("?,.'\"")
        if w and w.lower() not in [c.lower() for c in candidates]:
            candidates.append(w)
        if len(candidates) >= 3:
            break
    return candidates

def get_random_words(n=3):
    """Random common English words."""
    common = ["city", "country", "person", "year", "time", "world", "river",
              "mountain", "president", "director", "film", "book", "company",
              "university", "war", "battle", "king", "queen", "capital", "island"]
    return random.sample(common, min(n, len(common)))

def run_branch(question, candidates):
    """Branch with given candidates: retrieve for each, pick random, generate."""
    passages = retrieve(question, top_k=5)
    context = "\n\n".join(passages)
    if not candidates:
        return generate(context, question)
    c = random.choice(candidates)
    hop2 = retrieve(question + " " + c, top_k=3)
    expanded = context + "\n\n" + "\n\n".join(hop2)
    return generate(expanded, question)

qs = json.load(open("/projects/prjs1800/external/arag/data/musique/questions.json"))[:50]

for mode in ["no_branch", "dllm_candidates", "question_entities", "random_words"]:
    sf, cn = 0, 0
    for i, q in enumerate(qs):
        t0 = time.time()
        if mode == "no_branch":
            answer = generate("\n\n".join(retrieve(q["question"], top_k=5)), q["question"])
        elif mode == "dllm_candidates":
            passages = retrieve(q["question"], top_k=5)
            cands = get_dllm_candidates("\n\n".join(passages), q["question"])
            answer = run_branch(q["question"], cands)
        elif mode == "question_entities":
            cands = get_question_entities(q["question"])
            answer = run_branch(q["question"], cands)
        elif mode == "random_words":
            cands = get_random_words()
            answer = run_branch(q["question"], cands)
        elapsed = time.time() - t0
        sc = f1(answer, q["answer"])
        sf += sc
        if q["answer"].lower() in answer.lower(): cn += 1
        if i < 2 or i == 49:
            cand_str = ""
            if mode != "no_branch":
                cand_str = " cands=%s" % str(cands[:3] if 'cands' in dir() else [])
            print("[%s %d/50] (%.1fs) F1=%.2f%s %s" % (mode, i+1, elapsed, sc, cand_str, answer[:40]), flush=True)
    print("\n%s: F1=%.1f%% Contain=%d/50=%.1f%%\n" % (mode, sf/50*100, cn, cn/50*100), flush=True)
