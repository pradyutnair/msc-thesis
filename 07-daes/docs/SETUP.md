# RunPod / Remote GPU Setup Guide

Quick-start guide for running DNMR experiments on rented GPUs. Learned the hard way — follow this exactly.

## Requirements

- 1x A100 80GB (model 16GB + FAISS index 61GB = 77GB VRAM)
- 120GB+ system RAM (FAISS read_index needs to load 61GB into CPU RAM before GPU transfer)
- Container storage with: code repo, HF model cache, FAISS index, corpus, questions

## Step 1: SSH Access

RunPod provides two SSH endpoints:

- **Proxy SSH** (always works): `ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod`
  - Does NOT support non-PTY connections (breaks scp, Cursor Remote, piped commands)
- **Direct SSH** (use this): `ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519_runpod`
  - Get IP/port from RunPod dashboard under "TCP Port Mapping"
  - Supports everything: scp, non-interactive commands, background jobs

## Step 2: Workspace Setup

Run `setup_workspace.sh` from the thesis repo root:

```bash
# On the pod (first time only)
bash setup_workspace.sh --workspace /workspace
```

This creates:

- `/data/` — corpus (wiki18_100w.jsonl), questions, id_offset
- `/indexes/` — e5_Flat.index (61GB)
- `/hf_cache/` — cached HF models (LLaDA, Dream, E5)
- `/code/repo/` — git clone of msc-thesis
- `/msc-thesis/` — symlink or copy of repo
- `/init.sh` — environment activation script
- `/code/repo/.venv/` — Python venv (uv managed)

Also, we need to clone and install `dllm`

```bash
cd msc-thesis/07-daes/
git clone https://github.com/ZHZisZZ/dllm.git
cd msc-thesis/07-daes/dllm
uv pip install -e .
```

## Step 3: Critical Optimizations

### 3a. Copy FAISS index to RAM disk (eliminates disk I/O bottleneck)

```bash
cp /indexes/e5_Flat.index /dev/shm/e5_Flat.index
```

### 3b. Install faiss-gpu and move index to GPU (eliminates CPU search bottleneck)

Without this: ~66s/q (CPU brute-force on 21M vectors). With this: ~9s/q.

```bash
# Uninstall faiss-cpu (conflicts with faiss-gpu)
/code/repo/.venv/bin/pip uninstall -y faiss-cpu

# Install faiss-gpu
/code/repo/.venv/bin/pip install faiss-gpu-cu12
```

### 3c. Fix hardcoded paths

Update `eamd_v2_wiki18.py` — replace Snellius paths with RunPod paths:

```bash
cd /msc-thesis/07-daes/src/daes

# Data paths
sed -i 's|/projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl|/data/retrieval-corpus/wiki18_100w.jsonl|' eamd_v2_wiki18.py
sed -i 's|/projects/prjs1800/msc-thesis/01-arag-reproduction/data/index/wiki18_id_offset.json|/data/wiki18_id_offset.json|' eamd_v2_wiki18.py
sed -i 's|/projects/prjs1800/msc-thesis/01-arag-reproduction/data/questions_wiki18|/data/questions|' eamd_v2_wiki18.py

# FAISS index -> /dev/shm (RAM disk)
sed -i 's|/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index|/dev/shm/e5_Flat.index|' eamd_v2_wiki18.py

# FAISS CPU -> GPU transfer
sed -i 's|self.index = faiss.read_index(FAISS_INDEX)|cpu_index = faiss.read_index(FAISS_INDEX); res = faiss.StandardGpuResources(); self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index); self._gpu_res = res; del cpu_index|' eamd_v2_wiki18.py
```

### One-liner setup script (run all 3 steps at once)

```bash
# Copy index to RAM
cp /indexes/e5_Flat.index /dev/shm/e5_Flat.index

# Install faiss-gpu
/code/repo/.venv/bin/pip uninstall -y faiss-cpu && /code/repo/.venv/bin/pip install faiss-gpu-cu12

# Fix all paths + GPU FAISS in one go
cd /msc-thesis/07-daes/src/daes
sed -i \
  -e 's|/projects/prjs1800/datasets/flashrag/retrieval-corpus/wiki18_100w.jsonl|/data/retrieval-corpus/wiki18_100w.jsonl|' \
  -e 's|/projects/prjs1800/msc-thesis/01-arag-reproduction/data/index/wiki18_id_offset.json|/data/wiki18_id_offset.json|' \
  -e 's|/projects/prjs1800/msc-thesis/01-arag-reproduction/data/questions_wiki18|/data/questions|' \
  -e 's|/projects/prjs1800/datasets/flashrag/indexes/e5_Flat.index|/dev/shm/e5_Flat.index|' \
  -e 's|self.index = faiss.read_index(FAISS_INDEX)|cpu_index = faiss.read_index(FAISS_INDEX); res = faiss.StandardGpuResources(); self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index); self._gpu_res = res; del cpu_index|' \
  eamd_v2_wiki18.py
```

## Step 4: Run Experiments

```bash
cd /msc-thesis/07-daes/src/daes

# Foreground (see output live)
PYTHONPATH=/msc-thesis/07-daes/dllm:/msc-thesis/07-daes/src/daes \
  /code/repo/.venv/bin/python -u dnmr_pool_v2_lean.py \
    --model llada \
    --dataset musique \
    --n_questions 1000 \
    --output /results/pool_v2_llada_musique_1000q.json \
    2>&1 | tee /results/pool_v2_musique.log

# Background (survives SSH disconnect)
PYTHONPATH=/msc-thesis/07-daes/dllm:/msc-thesis/07-daes/src/daes \
  nohup /code/repo/.venv/bin/python -u dnmr_pool_v2_lean.py \
    --model llada \
    --dataset musique \
    --n_questions 1000 \
    --output /results/pool_v2_llada_musique_1000q.json \
    > /results/pool_v2_musique.log 2>&1 &
```

Monitor:

```bash
grep '^\[' /results/pool_v2_musique.log | tail -5   # progress
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader  # VRAM
ps aux | grep dnmr_pool  # process alive?
```

## Datasets

```bash
--dataset musique         --output /results/pool_v2_llada_musique_1000q.json
--dataset hotpotqa        --output /results/pool_v2_llada_hotpotqa_1000q.json
--dataset 2wikimultihopqa --output /results/pool_v2_llada_2wikimultihopqa_1000q.json
```

## Performance Reference


| Setup                     | Batch Retrieval (1000q) | Per-Question | Total 1000q | Cost (A100 $1.5/h) |
| ------------------------- | ----------------------- | ------------ | ----------- | ------------------ |
| Container disk, CPU FAISS | ~13 min                 | ~66s/q       | ~18h        | $27                |
| /dev/shm, CPU FAISS       | ~13 min                 | ~50s/q       | ~14h        | $21                |
| /dev/shm, GPU FAISS       | ~2 min                  | ~9s/q        | ~2.7h       | **$4**             |


**Always use GPU FAISS.**

## Gotchas

- **faiss-cpu and faiss-gpu conflict**: uninstall faiss-cpu BEFORE installing faiss-gpu, or `index_cpu_to_gpu` won't exist
- **RunPod proxy SSH rejects non-PTY**: use direct IP:port for scp, Cursor Remote, programmatic access
- **FAISS read_index takes ~2 min**: parsing 61GB from /dev/shm into memory structures. One-time cost.
- **Results save every 10 questions**: incremental JSON, won't lose progress on crash
- **PYTHONPATH required**: dllm and daes are not pip-installed, need explicit path
- **77GB VRAM usage**: model (16GB) + index (61GB). Only 3GB headroom on 80GB A100. Don't load anything else.

