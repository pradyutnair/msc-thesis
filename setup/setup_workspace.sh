#!/bin/bash
set -euo pipefail

# =============================================================================
# Portable Workspace Setup for Diffusion-Native Multi-Hop QA Experiments
# =============================================================================
# Usage: bash setup_workspace.sh [--workspace /path] [--skip-models] [--skip-data]
#
# Works on: IVI, RunPod, Vast.ai, or any Linux box with NVIDIA GPUs + conda
#
# Clones GIT_REPO into WORKSPACE/msc-thesis (repo root: uv.lock, pyproject.toml).
# Experiment code lives under msc-thesis/07-daes/, setup scripts under msc-thesis/setup/.
# =============================================================================

# === CONFIGURABLE PARAMETERS ===
WORKSPACE="${WORKSPACE:-/tmp/pnair}"
GIT_REPO="${GIT_REPO:-https://github.com/pradyutnair/msc-thesis.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"
# Provide HF_TOKEN via the environment
HF_TOKEN="${HF_TOKEN:-}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

# Models to download (HuggingFace model IDs)
MODELS=(
    "Dream-org/Dream-v0-Instruct-7B"
    "GSAI-ML/LLaDA-8B-Instruct"
    "intfloat/e5-base-v2"
)

# Skip flags
SKIP_MODELS=false
SKIP_DATA=false
SKIP_ENV=false
SKIP_CODE=false

# === DERIVED PATHS (change these if your layout differs) ===
# Repo root: clone target for https://github.com/.../msc-thesis (code under 07-daes/, setup/, etc.)
REPO_ROOT="${WORKSPACE}/msc-thesis"
DATA_DIR="${WORKSPACE}/data"
INDEX_DIR="${WORKSPACE}/indexes"
HF_CACHE="${WORKSPACE}/hf_cache"
ENV_DIR="${WORKSPACE}/env"
RESULTS_DIR="${WORKSPACE}/results"
TRITON_CACHE="${WORKSPACE}/triton_cache"

# Data paths (used by experiment scripts)
CORPUS_JSONL="${DATA_DIR}/retrieval-corpus/wiki18_100w.jsonl"
ID_OFFSET_JSON="${DATA_DIR}/wiki18_id_offset.json"
FAISS_INDEX="${INDEX_DIR}/e5_Flat.index"
QUESTIONS_DIR="${DATA_DIR}/questions"

# === PARSE ARGS ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --git-repo)  GIT_REPO="$2"; shift 2 ;;
        --skip-models) SKIP_MODELS=true; shift ;;
        --skip-data)   SKIP_DATA=true; shift ;;
        --skip-env)    SKIP_ENV=true; shift ;;
        --skip-code)   SKIP_CODE=true; shift ;;
        --help) echo "Usage: $0 [--workspace DIR] [--git-repo URL] [--skip-models] [--skip-data] [--skip-env] [--skip-code]"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Re-derive paths after potential workspace override
REPO_ROOT="${WORKSPACE}/msc-thesis"
DATA_DIR="${WORKSPACE}/data"
INDEX_DIR="${WORKSPACE}/indexes"
HF_CACHE="${WORKSPACE}/hf_cache"
ENV_DIR="${WORKSPACE}/env"
RESULTS_DIR="${WORKSPACE}/results"
TRITON_CACHE="${WORKSPACE}/triton_cache"
CORPUS_JSONL="${DATA_DIR}/retrieval-corpus/wiki18_100w.jsonl"
ID_OFFSET_JSON="${DATA_DIR}/wiki18_id_offset.json"
FAISS_INDEX="${INDEX_DIR}/e5_Flat.index"
QUESTIONS_DIR="${DATA_DIR}/questions"

echo "============================================"
echo "  Workspace Setup"
echo "============================================"
echo "  WORKSPACE:  ${WORKSPACE}"
echo "  REPO_ROOT:  ${REPO_ROOT}"
echo "  GIT_REPO:   ${GIT_REPO}"
echo "============================================"

# === 1. CREATE DIRECTORY STRUCTURE ===
echo -e "\n[1/7] Creating directories..."
mkdir -p "${DATA_DIR}/questions" "${DATA_DIR}/retrieval-corpus" \
         "${INDEX_DIR}" "${HF_CACHE}" "${ENV_DIR}" "${RESULTS_DIR}" "${TRITON_CACHE}"

# === 2. CLONE CODE (moved before env so uv sync can use uv.lock) ===
if [ "$SKIP_CODE" = false ]; then
    echo -e "\n[2/7] Cloning code..."
    if [ -d "${REPO_ROOT}/.git" ]; then
        echo "  Repo already cloned, pulling latest..."
        git -C "${REPO_ROOT}" pull --ff-only 2>/dev/null || true
    else
        git clone --branch "${GIT_BRANCH}" "${GIT_REPO}" "${REPO_ROOT}"
    fi
else
    echo -e "\n[2/7] Skipping code clone"
fi

# === 3. ENVIRONMENT (uv sync from repo's uv.lock) ===
if [ "$SKIP_ENV" = false ]; then
    echo -e "\n[3/7] Setting up Python environment with uv..."

    # Install uv if not available
    if ! command -v uv &>/dev/null; then
        echo "  Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Sync from uv.lock in the repo
    if [ -f "${REPO_ROOT}/uv.lock" ]; then
        echo "  Running uv sync..."
        cd "${REPO_ROOT}"
        uv sync --no-install-package vllm --no-install-project
        cd -
        ENV_DIR="${REPO_ROOT}/.venv"
    else
        echo "  WARNING: No uv.lock found in repo, falling back to pip"
        if [ ! -f "${ENV_DIR}/bin/python" ]; then
            python3 -m venv "${ENV_DIR}"
        fi
        source "${ENV_DIR}/bin/activate"
        pip install -q torch torchvision --index-url "${CUDA_INDEX}"
        pip install -q transformers sentence-transformers faiss-cpu accelerate huggingface-hub
    fi

    # Activate
    if [ -f "${ENV_DIR}/bin/activate" ]; then
        source "${ENV_DIR}/bin/activate"
    fi
    echo "  Environment ready"
else
    echo -e "\n[3/7] Skipping environment setup"
    # Try to activate existing env
    if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
        ENV_DIR="${REPO_ROOT}/.venv"
        source "${ENV_DIR}/bin/activate"
    elif [ -f "${ENV_DIR}/bin/activate" ]; then
        source "${ENV_DIR}/bin/activate"
    fi
fi

# (Code clone moved to step 2, before env setup)

# === 4. DOWNLOAD MODELS ===
export HF_HOME="${HF_CACHE}"
export HF_TOKEN="${HF_TOKEN}"

if [ "$SKIP_MODELS" = false ]; then
    echo -e "\n[4/6] Downloading models..."
    for model_id in "${MODELS[@]}"; do
        model_dir="${HF_CACHE}/hub/models--${model_id//\//__}"
        if [ -d "${model_dir}" ]; then
            echo "  ${model_id} already cached, skipping"
        else
            echo "  Downloading ${model_id}..."
            python -c "from huggingface_hub import snapshot_download; snapshot_download('${model_id}')" &
        fi
    done
    wait
    echo "  All models downloaded"
else
    echo -e "\n[4/6] Skipping model downloads"
fi

# === 5. DOWNLOAD CORPUS ===
if [ "$SKIP_DATA" = false ]; then
    echo -e "\n[5/6] Downloading corpus + data..."
    if [ -f "${CORPUS_JSONL}" ]; then
        echo "  Corpus already exists, skipping"
    else
        echo "  Downloading wiki18_100w from HuggingFace..."
        python -c "
from huggingface_hub import hf_hub_download
import zipfile, os
path = hf_hub_download(repo_id='RUC-NLPIR/FlashRAG_datasets',
                       filename='retrieval-corpus/wiki18_100w.zip',
                       repo_type='dataset',
                       local_dir='${DATA_DIR}')
print('  Extracting...')
with zipfile.ZipFile(path) as z:
    z.extractall('${DATA_DIR}/retrieval-corpus/')
os.remove(path)
# Clean up HF cache copy
import shutil
cache_dir = '${DATA_DIR}/.cache'
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
print('  Corpus ready')
"
    fi

    # === DOWNLOAD FAISS INDEX ===
    echo "  Downloading FAISS index..."
    if [ -f "${FAISS_INDEX}" ]; then
        echo "  Index already exists, skipping"
    else
        echo "  Downloading e5_Flat index from HuggingFace (~61GB, may take a while)..."
        python -c "
from huggingface_hub import snapshot_download
import os, shutil
dl_dir = '${INDEX_DIR}/download'
snapshot_download(repo_id='PeterJinGo/wiki-18-e5-index',
                  repo_type='dataset',
                  local_dir=dl_dir)
print('  Concatenating parts...')
# Remove cache to free space before concatenation
cache_dir = os.path.join(dl_dir, '.cache')
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
# Concatenate: mv part_aa, append part_ab, delete parts
os.rename(os.path.join(dl_dir, 'part_aa'), '${FAISS_INDEX}')
with open('${FAISS_INDEX}', 'ab') as out:
    with open(os.path.join(dl_dir, 'part_ab'), 'rb') as inp:
        while True:
            chunk = inp.read(100 * 1024 * 1024)  # 100MB chunks
            if not chunk:
                break
            out.write(chunk)
os.remove(os.path.join(dl_dir, 'part_ab'))
# Clean up download dir
for f in os.listdir(dl_dir):
    os.remove(os.path.join(dl_dir, f))
os.rmdir(dl_dir)
print('  Index ready')
"
    fi

    # === BUILD ID OFFSET MAP ===
    echo "  Building id_offset map..."
    if [ -f "${ID_OFFSET_JSON}" ]; then
        echo "  id_offset already exists, skipping"
    else
        python -c "
import json
offsets = {}
with open('${CORPUS_JSONL}', 'rb') as f:
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            break
        row = json.loads(line)
        offsets[row['id']] = pos
with open('${ID_OFFSET_JSON}', 'w') as f:
    json.dump(offsets, f)
print(f'  Built offset map for {len(offsets)} passages')
"
    fi

    # === DOWNLOAD QUESTION FILES ===
    echo "  Downloading question files..."
    if [ -f "${QUESTIONS_DIR}/hotpotqa.json" ]; then
        echo "  Questions already exist, skipping"
    else
        python -c "
from huggingface_hub import hf_hub_download
import json
datasets = {
    'hotpotqa': 'hotpotqa/dev.jsonl',
    'musique': 'musique/dev.jsonl',
    '2wikimultihopqa': '2wikimultihopqa/dev.jsonl',
}
for name, path in datasets.items():
    print(f'  Downloading {name}...')
    local = hf_hub_download(repo_id='RUC-NLPIR/FlashRAG_datasets',
                            filename=path, repo_type='dataset')
    # Convert JSONL to JSON array (match Snellius format)
    records = []
    with open(local) as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                'id': obj.get('id', ''),
                'qid': obj.get('id', ''),
                'question': obj.get('question', ''),
                'answer': obj.get('golden_answers', [''])[0] if isinstance(obj.get('golden_answers'), list) else obj.get('answer', ''),
                'golden_answers': obj.get('golden_answers', [obj.get('answer', '')]),
            })
    with open('${QUESTIONS_DIR}/' + name + '.json', 'w') as f:
        json.dump(records, f)
    print(f'    {len(records)} questions')
print('  Questions ready')
"
    fi
else
    echo -e "\n[5/6] Skipping data downloads"
fi

# === 7. GENERATE INIT SCRIPT ===
echo -e "\n[6/6] Generating init script..."
cat > "${WORKSPACE}/init.sh" << INITEOF
#!/bin/bash
# Source this file to set up the environment:
#   source ${WORKSPACE}/init.sh

# Load CUDA (IVI/HPC only — comment out on RunPod/Vast.ai)
if command -v module &>/dev/null; then
    source /etc/profile.d/modules.sh 2>/dev/null
    module load cuda12.6/toolkit/12.6 2>/dev/null
fi

# Activate Python env (uv creates .venv in repo root)
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
elif [ -f "${ENV_DIR}/bin/activate" ]; then
    source "${ENV_DIR}/bin/activate"
fi

# Environment variables
export HF_HOME="${HF_CACHE}"
export HF_TOKEN="${HF_TOKEN}"
# daes package: 07-daes/src; repo root for other imports
export PYTHONPATH="${REPO_ROOT}/07-daes/src:${REPO_ROOT}:\${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Data paths (importable via: from workspace_paths import *)
export WORKSPACE="${WORKSPACE}"
export CORPUS_JSONL="${CORPUS_JSONL}"
export ID_OFFSET_JSON="${ID_OFFSET_JSON}"
export FAISS_INDEX="${FAISS_INDEX}"
export QUESTIONS_DIR="${QUESTIONS_DIR}"
export RESULTS_DIR="${RESULTS_DIR}"

echo "Environment ready: \$(python --version), CUDA: \$(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), \"GPUs\")' 2>/dev/null || echo 'not checked')"
INITEOF
chmod +x "${WORKSPACE}/init.sh"

# Also generate a Python paths module
cat > "${WORKSPACE}/workspace_paths.py" << PYEOF
"""Auto-generated workspace paths. Import in experiment scripts."""
import os

WORKSPACE = "${WORKSPACE}"
CORPUS_JSONL = "${CORPUS_JSONL}"
ID_OFFSET_JSON = "${ID_OFFSET_JSON}"
FAISS_INDEX = "${FAISS_INDEX}"
QUESTIONS_DIR = "${QUESTIONS_DIR}"
RESULTS_DIR = "${RESULTS_DIR}"
QUESTION_FILES = {
    "hotpotqa": os.path.join(QUESTIONS_DIR, "hotpotqa.json"),
    "musique": os.path.join(QUESTIONS_DIR, "musique.json"),
    "2wikimultihopqa": os.path.join(QUESTIONS_DIR, "2wikimultihopqa.json"),
}
PYEOF

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo "  Activate with:  source ${WORKSPACE}/init.sh"
echo "  Workspace size:  $(du -sh ${WORKSPACE} 2>/dev/null | cut -f1)"
echo "  Disk free:       $(df -h ${WORKSPACE} 2>/dev/null | tail -1 | awk '{print $4}')"
echo "============================================"
