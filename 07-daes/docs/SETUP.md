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

## Step 2: Workspace setup (one script)

From the thesis repo root, run `msc-thesis/setup/setup_workspace.sh`. It creates the workspace layout (data, indexes, HF cache, venv, results), syncs the Python env from `uv.lock` (including **lm-eval**), clones **dLLM** into `07-daes/dllm` if missing and installs it editable, downloads models/data, and writes `init.sh` + `workspace_paths.py`.

```bash
# Standard (CPU FAISS from workspace disk)
bash msc-thesis/setup/setup_workspace.sh --workspace /workspace

# A100 + GPU FAISS: swaps faiss-cpu → faiss-gpu-cu12, copies the index to /dev/shm when possible
bash msc-thesis/setup/setup_workspace.sh --workspace /workspace --with-gpu-faiss
```

Useful flags: `--skip-env`, `--skip-models`, `--skip-data`, `--skip-dllm`, `--show-disk-usage`.

After it finishes:

```bash
source /workspace/init.sh
```

`init.sh` exports paths (`CORPUS_JSONL`, `QUESTIONS_DIR`, `FAISS_INDEX`, …) and `PYTHONPATH` (`07-daes/dllm`, `07-daes/src`). `07-daes/src/daes/eamd_v2_wiki18.py` reads those env vars (defaults remain Snellius paths), so you do **not** need manual `sed` on that file when using this workspace.

## Step 3: What the script used to do manually

The following are now covered by Step 2 (or by sourcing `init.sh`):

- Clone + editable install of **dLLM** (`--skip-dllm` to opt out).
- **GPU FAISS** + optional **RAM-disk index** copy: use `--with-gpu-faiss` (requires `nvidia-smi` and a venv). Without it, the workflow uses faiss-cpu and the index path under `${WORKSPACE}/indexes/`.
- **lm-eval** is a project dependency in `pyproject.toml` / `uv.lock`; it is installed with `uv sync`.

If you must do the GPU FAISS swap by hand (same as the script):

```bash
# Uninstall CPU build, install GPU build (conflicts if both present)
/path/to/venv/bin/pip uninstall -y faiss-cpu
/path/to/venv/bin/pip install faiss-gpu-cu12

# Optional: copy index to shared memory (large, fast I/O)
cp "${FAISS_INDEX}" /dev/shm/e5_Flat.index
export DAES_FAISS_GPU=1
export DAES_FAISS_INDEX=/dev/shm/e5_Flat.index
```

## Step 4: Run experiments

```bash
cd "${REPO_ROOT}/07-daes/src/daes"
# After: source "${WORKSPACE}/init.sh"

python -u dnmr_pool_v2_lean.py \
  --model llada \
  --dataset musique \
  --n_questions 1000 \
  --output "${RESULTS_DIR}/pool_v2_llada_musique_1000q.json" \
  2>&1 | tee "${RESULTS_DIR}/pool_v2_musique.log"
```

Background:

```bash
nohup python -u dnmr_pool_v2_lean.py \
  --model llada \
  --dataset musique \
  --n_questions 1000 \
  --output "${RESULTS_DIR}/pool_v2_llada_musique_1000q.json" \
  > "${RESULTS_DIR}/pool_v2_musique.log" 2>&1 &
```

Monitor:

```bash
grep '^\[' "${RESULTS_DIR}/pool_v2_musique.log" | tail -5
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
ps aux | grep dnmr_pool
```

## Datasets

```bash
--dataset musique         --output "${RESULTS_DIR}/pool_v2_llada_musique_1000q.json"
--dataset hotpotqa        --output "${RESULTS_DIR}/pool_v2_llada_hotpotqa_1000q.json"
--dataset 2wikimultihopqa --output "${RESULTS_DIR}/pool_v2_llada_2wikimultihopqa_1000q.json"
```

## Performance Reference

| Setup                     | Batch Retrieval (1000q) | Per-Question | Total 1000q | Cost (A100 $1.5/h) |
| ------------------------- | ----------------------- | ------------ | ----------- | ------------------ |
| Container disk, CPU FAISS | ~13 min                 | ~66s/q       | ~18h        | $27                |
| /dev/shm, CPU FAISS       | ~13 min                 | ~50s/q       | ~14h        | $21                |
| /dev/shm, GPU FAISS       | ~2 min                  | ~9s/q        | ~2.7h       | **$4**             |

**Always use GPU FAISS** on A100 for production runs (`--with-gpu-faiss` once the index exists).

## Gotchas

- **faiss-cpu and faiss-gpu conflict**: uninstall faiss-cpu before faiss-gpu, or `index_cpu_to_gpu` may be missing.
- **RunPod proxy SSH rejects non-PTY**: use direct IP:port for scp, Cursor Remote, programmatic access.
- **FAISS read_index takes ~2 min**: parsing 61GB from /dev/shm into memory structures. One-time cost.
- **Results save every 10 questions**: incremental JSON, won't lose progress on crash.
- **PYTHONPATH**: `init.sh` sets `dllm` + `daes`; run experiments after `source init.sh`.
- **77GB VRAM usage**: model (16GB) + index (61GB). Only 3GB headroom on 80GB A100. Don't load anything else.
