# Vast.ai Experiment Setup

## What needs to run on GPU:
1. **AR candidate comparison** (CRITICAL): Qwen3-8B generates candidates instead of Dream-7B. 50q MuSiQue. ~30 min.
2. **maskgit-plus baseline**: Dream-7B with alg="maskgit_plus". 50q MuSiQue. ~10 min.
3. **Candidate count ablation**: n=1,2,3,5 candidates. 50q MuSiQue. ~40 min total.
4. **Self-consistency**: Generate 3 answers (one per candidate), majority vote. 50q MuSiQue. ~20 min.

Total: ~2 hours of A100 time.

## Models to download (~30GB total):
- Dream-7B-Instruct: `Dream-org/Dream-v0-Instruct-7B` (~14GB)
- E5-base-v2: `intfloat/e5-base-v2` (~0.5GB)
- Qwen3-8B: `Qwen/Qwen3-8B` (~16GB) — for AR comparison

## Data to copy from Snellius:
- `/projects/prjs1800/external/arag/data/musique/questions.json` (1000 questions)
- `/projects/prjs1800/external/arag/data/musique/index_e5_musique_full/sentence_index.pkl` (~300MB)
- `/projects/prjs1800/external/arag/data/hotpotqa/questions.json`
- `/projects/prjs1800/external/arag/data/hotpotqa/index_e5_full/sentence_index.pkl`
- `/projects/prjs1800/external/arag/data/2wikimultihop/questions.json`
- `/projects/prjs1800/external/arag/data/2wikimultihop/index_e5_full/sentence_index.pkl`

## Environment:
```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.0 accelerate peft datasets sentencepiece
pip install omegaconf tyro tqdm rich einops sentence-transformers
git clone https://github.com/ZHZisZZ/dllm.git
# Patch dllm: remove non-Dream imports in pipelines/__init__.py and utils/models.py
```

## Scripts needed:
- ablation_candidate_source.py (modified for AR comparison)
- track2_bv_nvembed.py (main experiment runner)
- spread_reproduce.py (for maskgit-plus baseline)
