#!/bin/bash
set -e

echo "=== Step 1: Install uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== Step 2: Create venv ==="
uv venv .venv --python 3.12
source .venv/bin/activate

echo "=== Step 3: Install deps ==="
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install transformers==4.57.0 accelerate peft datasets sentencepiece \
    omegaconf tyro tqdm rich einops sentence-transformers

echo "=== Step 4: Clone + patch dllm ==="
git clone https://github.com/ZHZisZZ/dllm.git
sed -i 's/from . import a2d, bert, dream, editflow, fastdllm, llada, llada2/from . import dream/' dllm/dllm/pipelines/__init__.py
sed -i 's/from . import eval, samplers, schedulers, trainers/from . import samplers, schedulers/' dllm/dllm/core/__init__.py
python3 -c "
content = open('dllm/dllm/utils/models.py').read()
import re
content = re.sub(r'from dllm\.pipelines\.a2d.*?\)', 'class A2DLlamaLMHeadModel: pass\n    class A2DQwen2LMHeadModel: pass\n    class A2DQwen3LMHeadModel: pass', content, flags=re.DOTALL)
content = content.replace('from dllm.pipelines.llada2.models.modeling_llada2_moe import LLaDA2MoeModelLM', 'class LLaDA2MoeModelLM: pass')
content = content.replace('from dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM', 'class LLaDAModelLM: pass')
content = content.replace('from dllm.pipelines.llada.models.modeling_lladamoe import LLaDAMoEModelLM', 'class LLaDAMoEModelLM: pass')
open('dllm/dllm/utils/models.py', 'w').write(content)
print('dllm patched')
"
export PYTHONPATH=$(pwd)/dllm:$PYTHONPATH

echo "=== Step 5: Clone thesis repo ==="
git clone https://github.com/pradyutnair/msc-thesis.git

echo "=== Step 6: Download models ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Dream-org/Dream-v0-Instruct-7B')
snapshot_download('Qwen/Qwen3-8B')
snapshot_download('intfloat/e5-base-v2')
print('All models cached')
"

echo "=== Step 7: Copy data from Snellius ==="
mkdir -p data/musique data/hotpotqa data/2wikimultihop
echo "Run these from your local machine:"
echo "  scp snellius:/projects/prjs1800/external/arag/data/musique/questions.json data/musique/"
echo "  scp snellius:/projects/prjs1800/external/arag/data/musique/index_e5_musique_full/sentence_index.pkl data/musique/"
echo "  scp snellius:/projects/prjs1800/external/arag/data/hotpotqa/questions.json data/hotpotqa/"
echo "  scp snellius:/projects/prjs1800/external/arag/data/hotpotqa/index_e5_full/sentence_index.pkl data/hotpotqa/"
echo "  scp snellius:/projects/prjs1800/external/arag/data/2wikimultihop/questions.json data/2wikimultihop/"
echo "  scp snellius:/projects/prjs1800/external/arag/data/2wikimultihop/index_e5_full/sentence_index.pkl data/2wikimultihop/"

echo "=== Setup complete ==="
