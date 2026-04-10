#!/bin/bash
set -euo pipefail

# =============================================================================
# Workspace Setup for Diffusion-Native Multi-Hop QA Experiments
# =============================================================================
# Usage:
#   bash setup_workspace.sh [--workspace /path] [--repo-root /path] \
#       [--skip-env] [--skip-models] [--skip-data] [--skip-dllm] [--with-gpu-faiss]
#
# Notes:
# - This script is intended for HPC / shared filesystems.
# - All writable runtime state lives under WORKSPACE.
# - Hugging Face caches are explicitly separated:
#     HF_HOME/
#       hub/
#       xet/
#       assets/
# - Xet is disabled by default for stability on cluster filesystems.
# =============================================================================

# -----------------------------
# Defaults
# -----------------------------
WORKSPACE="${WORKSPACE:-/tmp/pnair}"
PYTHON_VERSION="${WORKSPACE_PYTHON_VERSION:-3.10.18}"
if [[ "${PYTHON_VERSION}" == "3.10" ]]; then
  PYTHON_VERSION="3.10.18"
fi
HF_TOKEN="${HF_TOKEN:-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}"

SKIP_ENV=false
SKIP_MODELS=false
SKIP_DATA=false
SKIP_DLLM=false
WITH_GPU_FAISS=true
SHOW_DISK_USAGE=false

DLLM_GIT_URL="${DLLM_GIT_URL:-https://github.com/ZHZisZZ/dllm.git}"
SHM_FAISS_INDEX="/dev/shm/e5_Flat.index"

MODELS=(
  "Dream-org/Dream-v0-Instruct-7B"
  "GSAI-ML/LLaDA-8B-Instruct"
  "intfloat/e5-base-v2"
)

# -----------------------------
# Infer repo root from script path
# -----------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# -----------------------------
# Parse args
# -----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "ERROR: --workspace requires a value"
        exit 1
      fi
      WORKSPACE="$2"
      shift 2
      ;;
    --repo-root)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "ERROR: --repo-root requires a value"
        exit 1
      fi
      REPO_ROOT="$2"
      shift 2
      ;;
    --skip-env)
      SKIP_ENV=true
      shift
      ;;
    --skip-models)
      SKIP_MODELS=true
      shift
      ;;
    --skip-data)
      SKIP_DATA=true
      shift
      ;;
    --skip-dllm)
      SKIP_DLLM=true
      shift
      ;;
    --with-gpu-faiss)
      WITH_GPU_FAISS=true
      shift
      ;;
    --show-disk-usage)
      SHOW_DISK_USAGE=true
      shift
      ;;
    --help)
      cat <<EOF
Usage: $0 [--workspace DIR] [--repo-root DIR] [--skip-env] [--skip-models] [--skip-data]
           [--skip-dllm] [--with-gpu-faiss] [--show-disk-usage]

Examples:
  bash $0 --workspace /workspace/pnair
  bash $0 --workspace /workspace --with-gpu-faiss
  WORKSPACE=/workspace/pnair HF_TOKEN=... bash $0
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# -----------------------------
# Derived paths
# -----------------------------
DLLM_DIR="${REPO_ROOT}/07-daes/dllm"

DATA_DIR="${WORKSPACE}/data"
QUESTIONS_DIR="${DATA_DIR}/questions"
CORPUS_DIR="${DATA_DIR}/retrieval-corpus"
INDEX_DIR="${WORKSPACE}/indexes"
RESULTS_DIR="${WORKSPACE}/results"
TRITON_CACHE="${WORKSPACE}/triton_cache"
MANIFEST_DIR="${WORKSPACE}/manifests"
TMP_DIR="${WORKSPACE}/tmp"
VENV_DIR="${WORKSPACE}/venv"
UV_CACHE_DIR="${WORKSPACE}/uv_cache"

HF_HOME_DIR="${WORKSPACE}/hf_home"
HF_HUB_CACHE_DIR="${HF_HOME_DIR}/hub"
HF_XET_CACHE_DIR="${HF_HOME_DIR}/xet"
HF_ASSETS_CACHE_DIR="${HF_HOME_DIR}/assets"

CORPUS_JSONL="${CORPUS_DIR}/wiki18_100w.jsonl"
ID_OFFSET_JSON="${DATA_DIR}/wiki18_id_offset.json"
FAISS_INDEX="${INDEX_DIR}/e5_Flat.index"
MODEL_MANIFEST_JSON="${MANIFEST_DIR}/model_paths.json"

# -----------------------------
# Helpers
# -----------------------------
section() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $1"
    exit 1
  fi
}

setup_hf_env() {
  export HF_HOME="${HF_HOME_DIR}"
  export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
  export HF_XET_CACHE="${HF_XET_CACHE_DIR}"
  export HF_ASSETS_CACHE="${HF_ASSETS_CACHE_DIR}"
  export TRANSFORMERS_CACHE="${HF_HUB_CACHE_DIR}"
  export HF_HUB_DISABLE_XET=1
  export HF_TOKEN="${HF_TOKEN}"

  mkdir -p \
    "${HF_HOME_DIR}" \
    "${HF_HUB_CACHE_DIR}" \
    "${HF_XET_CACHE_DIR}" \
    "${HF_ASSETS_CACHE_DIR}"
}

# -----------------------------
# Validate repo
# -----------------------------
section "Workspace setup"

echo "Repo root   : ${REPO_ROOT}"
echo "Workspace   : ${WORKSPACE}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: repo root does not exist: ${REPO_ROOT}"
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/pyproject.toml" && ! -f "${REPO_ROOT}/uv.lock" ]]; then
  echo "ERROR: ${REPO_ROOT} does not look like your repo root (missing pyproject.toml / uv.lock)"
  exit 1
fi

# -----------------------------
# Create directories
# -----------------------------
section "Creating workspace directories"

mkdir -p \
  "${DATA_DIR}" \
  "${QUESTIONS_DIR}" \
  "${CORPUS_DIR}" \
  "${INDEX_DIR}" \
  "${RESULTS_DIR}" \
  "${TRITON_CACHE}" \
  "${MANIFEST_DIR}" \
  "${TMP_DIR}" \
  "${VENV_DIR}" \
  "${UV_CACHE_DIR}"

setup_hf_env

# -----------------------------
# Environment setup
# -----------------------------
if [[ "${SKIP_ENV}" == false ]]; then
  section "Setting up Python environment"

  require_cmd python3
  require_cmd curl

  if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
  export UV_CACHE_DIR="${UV_CACHE_DIR}"
  export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
  export UV_PYTHON_INSTALL_DIR="${WORKSPACE}/uv_python"
  export UV_PYTHON_PREFERENCE="managed"

  mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

  unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PYTHON_VERSION UV_PROJECT_ENVIRONMENT UV_PYTHON_INSTALL_DIR UV_PYTHON_PREFERENCE
  hash -r

  if [[ -f "${REPO_ROOT}/uv.lock" ]]; then
    echo "Using uv.lock from repo"
    cd "${REPO_ROOT}"
    uv python install "${PYTHON_VERSION}"
    rm -rf "${VENV_DIR}"
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    uv sync --python "${VENV_DIR}/bin/python" --no-install-package vllm --no-install-project
  else
    echo "uv.lock not found; falling back to venv + pip"
    uv python install "${PYTHON_VERSION}"
    rm -rf "${VENV_DIR}"
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install \
      torch \
      torchvision \
      transformers \
      sentence-transformers \
      faiss-cpu \
      accelerate \
      huggingface-hub \
      "lm-eval>=0.4.8"
  fi
      huggingface-hub \
      "lm-eval>=0.4.8"
  fi

  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  else
    echo "ERROR: venv activation script missing: ${VENV_DIR}/bin/activate"
    exit 1
  fi

  echo "Python: $(python --version)"
else
  section "Skipping environment setup"

  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  else
    echo "WARNING: --skip-env passed but no venv found at ${VENV_DIR}"
  fi
fi

require_cmd python
setup_hf_env

# -----------------------------
# dLLM (editable install; clone if missing)
# -----------------------------
if [[ "${SKIP_DLLM}" == false ]]; then
  section "dLLM package"

  if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
    echo "WARNING: no venv python at ${VENV_DIR}; skipping dLLM"
  elif ! command -v git >/dev/null 2>&1; then
    echo "WARNING: git not found; cannot clone dLLM"
  else
    if [[ ! -d "${DLLM_DIR}/.git" ]]; then
      echo "Cloning dLLM into ${DLLM_DIR}..."
      mkdir -p "$(dirname "${DLLM_DIR}")"
      git clone --depth 1 "${DLLM_GIT_URL}" "${DLLM_DIR}"
    else
      echo "dLLM repo already present: ${DLLM_DIR}"
    fi

    if [[ -f "${DLLM_DIR}/pyproject.toml" ]]; then
      if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${VENV_DIR}/bin/python" -e "${DLLM_DIR}"
      else
        "${VENV_DIR}/bin/pip" install -e "${DLLM_DIR}"
      fi
    else
      echo "ERROR: ${DLLM_DIR} missing pyproject.toml after clone"
      exit 1
    fi
  fi
else
  section "Skipping dLLM install (--skip-dllm)"
fi

# -----------------------------
# Download models
# -----------------------------
if [[ "${SKIP_MODELS}" == false ]]; then
  section "Downloading / resolving model cache paths"

  python - "${MODEL_MANIFEST_JSON}" "${MODELS[@]}" <<'PY'
import json
import os
import sys
from huggingface_hub import snapshot_download

manifest_path = sys.argv[1]
model_ids = sys.argv[2:]

cache_dir = os.environ["HF_HUB_CACHE"]
manifest = {}

for model_id in model_ids:
    print(f"[model] {model_id}")
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        max_workers=2,
        token=os.environ.get("HF_TOKEN") or None,
    )
    manifest[model_id] = path

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nWrote model manifest to: {manifest_path}")
PY
else
  section "Skipping model downloads"

  if [[ ! -f "${MODEL_MANIFEST_JSON}" ]]; then
    echo "WARNING: model manifest does not exist yet: ${MODEL_MANIFEST_JSON}"
  fi
fi

# -----------------------------
# Download data
# -----------------------------
if [[ "${SKIP_DATA}" == false ]]; then
  section "Downloading corpus / index / questions"

  if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
  fi
  setup_hf_env

  # 1) wiki18 corpus
  if [[ -f "${CORPUS_JSONL}" ]]; then
    echo "[corpus] Exists: ${CORPUS_JSONL}"
  else
    echo "[corpus] Downloading wiki18_100w.zip and extracting..."
    python - <<PY
from huggingface_hub import hf_hub_download
import zipfile
import os
import shutil

data_dir = r"${DATA_DIR}"
corpus_dir = r"${CORPUS_DIR}"
cache_dir = os.environ["HF_HUB_CACHE"]

zip_path = hf_hub_download(
    repo_id="RUC-NLPIR/FlashRAG_datasets",
    filename="retrieval-corpus/wiki18_100w.zip",
    repo_type="dataset",
    local_dir=data_dir,
    cache_dir=cache_dir,
    token=os.environ.get("HF_TOKEN") or None,
)

print(f"Downloaded zip to: {zip_path}")

with zipfile.ZipFile(zip_path) as z:
    z.extractall(corpus_dir)

if os.path.exists(zip_path):
    os.remove(zip_path)

cache_meta_dir = os.path.join(data_dir, ".cache")
if os.path.exists(cache_meta_dir):
    shutil.rmtree(cache_meta_dir)

print(f"Extracted corpus into: {corpus_dir}")
PY
  fi

  # 2) FAISS index
  if [[ -f "${FAISS_INDEX}" ]]; then
    echo "[index] Exists: ${FAISS_INDEX}"
  else
    echo "[index] Downloading e5_Flat index and concatenating parts..."
    python - <<PY
from huggingface_hub import snapshot_download
import os
import shutil

index_dir = r"${INDEX_DIR}"
download_dir = os.path.join(index_dir, "download")
final_index = r"${FAISS_INDEX}"
cache_dir = os.environ["HF_HUB_CACHE"]

if os.path.exists(download_dir):
    shutil.rmtree(download_dir)

snapshot_download(
    repo_id="PeterJinGo/wiki-18-e5-index",
    repo_type="dataset",
    local_dir=download_dir,
    cache_dir=cache_dir,
    max_workers=2,
    token=os.environ.get("HF_TOKEN") or None,
)

part_aa = os.path.join(download_dir, "part_aa")
part_ab = os.path.join(download_dir, "part_ab")

if not os.path.exists(part_aa) or not os.path.exists(part_ab):
    raise FileNotFoundError(
        f"Expected part_aa and part_ab in {download_dir}, but did not find them."
    )

print("Concatenating part_aa + part_ab -> e5_Flat.index")
os.replace(part_aa, final_index)

with open(final_index, "ab") as out_f, open(part_ab, "rb") as in_f:
    while True:
        chunk = in_f.read(100 * 1024 * 1024)
        if not chunk:
            break
        out_f.write(chunk)

os.remove(part_ab)

for name in os.listdir(download_dir):
    path = os.path.join(download_dir, name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

os.rmdir(download_dir)

print(f"Index ready: {final_index}")
PY
  fi

  # 3) id -> file offset map
  if [[ -f "${ID_OFFSET_JSON}" ]]; then
    echo "[offsets] Exists: ${ID_OFFSET_JSON}"
  else
    echo "[offsets] Building id_offset map..."
    python - <<PY
import json

corpus_jsonl = r"${CORPUS_JSONL}"
offset_json = r"${ID_OFFSET_JSON}"

offsets = {}
with open(corpus_jsonl, "rb") as f:
    while True:
        pos = f.tell()
        line = f.readline()
        if not line:
            break
        row = json.loads(line)
        offsets[row["id"]] = pos

with open(offset_json, "w") as f:
    json.dump(offsets, f)

print(f"Built offset map for {len(offsets)} passages -> {offset_json}")
PY
  fi

  # 4) question files
  if [[ -f "${QUESTIONS_DIR}/hotpotqa.json" && -f "${QUESTIONS_DIR}/musique.json" && -f "${QUESTIONS_DIR}/2wikimultihopqa.json" ]]; then
    echo "[questions] Already exist in ${QUESTIONS_DIR}"
  else
    echo "[questions] Downloading and converting JSONL -> JSON array..."
    python - <<PY
from huggingface_hub import hf_hub_download
import json
import os

questions_dir = r"${QUESTIONS_DIR}"
cache_dir = os.environ["HF_HUB_CACHE"]

datasets = {
    "hotpotqa": "hotpotqa/dev.jsonl",
    "musique": "musique/dev.jsonl",
    "2wikimultihopqa": "2wikimultihopqa/dev.jsonl",
}

os.makedirs(questions_dir, exist_ok=True)

for name, path in datasets.items():
    print(f"Downloading {name}...")
    local = hf_hub_download(
        repo_id="RUC-NLPIR/FlashRAG_datasets",
        filename=path,
        repo_type="dataset",
        cache_dir=cache_dir,
        token=os.environ.get("HF_TOKEN") or None,
    )

    records = []
    with open(local) as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                "id": obj.get("id", ""),
                "qid": obj.get("id", ""),
                "question": obj.get("question", ""),
                "answer": (
                    obj.get("golden_answers", [""])[0]
                    if isinstance(obj.get("golden_answers"), list)
                    else obj.get("answer", "")
                ),
                "golden_answers": obj.get("golden_answers", [obj.get("answer", "")]),
            })

    out_path = os.path.join(questions_dir, f"{name}.json")
    with open(out_path, "w") as f:
        json.dump(records, f)

    print(f"  wrote {len(records)} questions -> {out_path}")

print("Questions ready")
PY
  fi
else
  section "Skipping data downloads"
fi

# -----------------------------
# Optional GPU FAISS
# -----------------------------
DAES_FAISS_GPU_VAL="0"
DAES_FAISS_INDEX_VAL="${FAISS_INDEX}"

if [[ "${WITH_GPU_FAISS}" == true ]]; then
  if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
    echo "WARNING: --with-gpu-faiss set but no venv; skipping GPU FAISS swap"
  elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: --with-gpu-faiss set but nvidia-smi not found; skipping GPU FAISS swap"
  else
    section "GPU FAISS (faiss-gpu-cu12)"

    PYEXE="${VENV_DIR}/bin/python"
    if command -v uv >/dev/null 2>&1; then
      uv pip uninstall --python "${PYEXE}" -y faiss-cpu 2>/dev/null || true
      uv pip install --python "${PYEXE}" faiss-gpu-cu12
    else
      "${PYEXE}" -m pip uninstall -y faiss-cpu 2>/dev/null || true
      "${PYEXE}" -m pip install faiss-gpu-cu12
    fi

    DAES_FAISS_GPU_VAL="1"

    if [[ -f "${FAISS_INDEX}" && -d /dev/shm ]]; then
      echo "Copying ${FAISS_INDEX} -> ${SHM_FAISS_INDEX}..."
      if cp -f "${FAISS_INDEX}" "${SHM_FAISS_INDEX}"; then
        DAES_FAISS_INDEX_VAL="${SHM_FAISS_INDEX}"
      else
        echo "WARNING: copy to /dev/shm failed; using index on disk"
      fi
    fi
  fi
fi

# -----------------------------
# Generate init.sh
# -----------------------------
section "Generating init.sh and workspace_paths.py"

cat > "${WORKSPACE}/init.sh" <<EOF
#!/bin/bash

if command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load CUDA/12.8.0 2>/dev/null || true
fi

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

export WORKSPACE="${WORKSPACE}"
export REPO_ROOT="${REPO_ROOT}"

export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HUB_CACHE_DIR}"
export HF_XET_CACHE="${HF_XET_CACHE_DIR}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE_DIR}"
export HF_HUB_DISABLE_XET=1
export HF_TOKEN="${HF_TOKEN}"

export TRITON_CACHE_DIR="${TRITON_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${REPO_ROOT}/07-daes/dllm:${REPO_ROOT}/07-daes/src:${REPO_ROOT}:\${PYTHONPATH:-}"

export DATA_DIR="${DATA_DIR}"
export QUESTIONS_DIR="${QUESTIONS_DIR}"
export CORPUS_JSONL="${CORPUS_JSONL}"
export ID_OFFSET_JSON="${ID_OFFSET_JSON}"
export FAISS_INDEX="${FAISS_INDEX}"
export RESULTS_DIR="${RESULTS_DIR}"
export MODEL_MANIFEST_JSON="${MODEL_MANIFEST_JSON}"

export DAES_FAISS_GPU="${DAES_FAISS_GPU_VAL}"
export DAES_FAISS_INDEX="${DAES_FAISS_INDEX_VAL}"

echo "Repo root        : \$REPO_ROOT"
echo "Workspace        : \$WORKSPACE"
echo "Python           : \$(python --version 2>/dev/null || echo 'missing')"
echo "CUDA check       : \$(python - <<'PY'
try:
    import torch
    print(torch.cuda.is_available(), torch.cuda.device_count(), "GPUs")
except Exception as e:
    print("torch check failed:", e)
PY
)"
EOF

chmod +x "${WORKSPACE}/init.sh"

cat > "${WORKSPACE}/workspace_paths.py" <<EOF
"""Auto-generated workspace paths."""
from pathlib import Path

WORKSPACE = Path(r"${WORKSPACE}")
REPO_ROOT = Path(r"${REPO_ROOT}")

DATA_DIR = Path(r"${DATA_DIR}")
QUESTIONS_DIR = Path(r"${QUESTIONS_DIR}")
CORPUS_JSONL = Path(r"${CORPUS_JSONL}")
ID_OFFSET_JSON = Path(r"${ID_OFFSET_JSON}")
FAISS_INDEX = Path(r"${FAISS_INDEX}")

HF_HOME = Path(r"${HF_HOME_DIR}")
HF_HUB_CACHE = Path(r"${HF_HUB_CACHE_DIR}")
HF_XET_CACHE = Path(r"${HF_XET_CACHE_DIR}")
HF_ASSETS_CACHE = Path(r"${HF_ASSETS_CACHE_DIR}")

VENV_DIR = Path(r"${VENV_DIR}")
RESULTS_DIR = Path(r"${RESULTS_DIR}")
TRITON_CACHE = Path(r"${TRITON_CACHE}")
MODEL_MANIFEST_JSON = Path(r"${MODEL_MANIFEST_JSON}")

QUESTION_FILES = {
    "hotpotqa": QUESTIONS_DIR / "hotpotqa.json",
    "musique": QUESTIONS_DIR / "musique.json",
    "2wikimultihopqa": QUESTIONS_DIR / "2wikimultihopqa.json",
}
EOF

# -----------------------------
# Final summary
# -----------------------------
section "Setup complete"

echo "Source env with:"
echo "  source ${WORKSPACE}/init.sh"
echo ""

echo "Important paths:"
echo "  REPO_ROOT        = ${REPO_ROOT}"
echo "  WORKSPACE        = ${WORKSPACE}"
echo "  VENV_DIR         = ${VENV_DIR}"
echo "  HF_HOME          = ${HF_HOME_DIR}"
echo "  HF_HUB_CACHE     = ${HF_HUB_CACHE_DIR}"
echo "  HF_XET_CACHE     = ${HF_XET_CACHE_DIR}"
echo "  HF_ASSETS_CACHE  = ${HF_ASSETS_CACHE_DIR}"
echo "  DATA_DIR         = ${DATA_DIR}"
echo "  QUESTIONS_DIR    = ${QUESTIONS_DIR}"
echo "  CORPUS_JSONL     = ${CORPUS_JSONL}"
echo "  ID_OFFSET_JSON   = ${ID_OFFSET_JSON}"
echo "  INDEX_DIR        = ${INDEX_DIR}"
echo "  FAISS_INDEX      = ${FAISS_INDEX}"
echo "  RESULTS_DIR      = ${RESULTS_DIR}"
echo "  TRITON_CACHE     = ${TRITON_CACHE}"
echo "  INIT_SCRIPT      = ${WORKSPACE}/init.sh"
echo "  PY_PATHS_MODULE  = ${WORKSPACE}/workspace_paths.py"
echo "  MODEL_MANIFEST   = ${MODEL_MANIFEST_JSON}"
echo ""

if [[ -f "${MODEL_MANIFEST_JSON}" ]]; then
  echo "Resolved model cache paths:"
  python - <<PY
import json
manifest_path = r"${MODEL_MANIFEST_JSON}"
with open(manifest_path) as f:
    manifest = json.load(f)
for model_id, path in manifest.items():
    print(f"  {model_id} -> {path}")
PY
  echo ""
fi

if [[ "${SHOW_DISK_USAGE}" == true ]]; then
  echo "Disk usage:"
  du -sh "${WORKSPACE}" 2>/dev/null || true
else
  echo "Disk usage: skipped full scan (re-run with --show-disk-usage)"
fi

echo ""
echo "Free space:"
df -hP "${WORKSPACE}" 2>/dev/null | awk 'NR==2 { print; exit }' || true
