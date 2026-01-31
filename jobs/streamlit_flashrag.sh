#!/bin/bash
# Run FlashRAG Streamlit demo. Use with an interactive GPU job or on login node.
# Interactive job: srun --partition=gpu_a100 --gpus=1 --mem=32G --time=2:00:00 --pty bash
# Then: cd /projects/prjs1800/msc-thesis && ./jobs/streamlit_flashrag.sh

set -e
PORT="${STREAMLIT_PORT:-8501}"

module load 2025
module load Anaconda3/2025.06-1
module load CUDA/12.8.0

cd /projects/prjs1800
source venvs/FlashRAG-venv/bin/activate

export HF_HOME="/projects/prjs1800/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export PYTHONPATH="/projects/prjs1800/external/FlashRAG:$PYTHONPATH"

FLASHRAG_EXAMPLES="/projects/prjs1800/external/FlashRAG/examples/quick_start"
FLASHRAG_METHODS="/projects/prjs1800/external/FlashRAG/examples/methods"
[ ! -f "$FLASHRAG_EXAMPLES/my_config.yaml" ] && cp "$FLASHRAG_METHODS/my_config.yaml" "$FLASHRAG_EXAMPLES/my_config.yaml"
cd "$FLASHRAG_EXAMPLES"

echo "=========================================="
echo "FlashRAG Streamlit UI"
echo "Port: $PORT"
echo "On Snellius: from your laptop run:"
echo "  ssh -L ${PORT}:localhost:${PORT} $USER@snellius.surf.nl"
echo "Then open: http://localhost:${PORT}"
echo "=========================================="

exec streamlit run demo_en.py \
  --server.port="$PORT" \
  --server.address=0.0.0.0 \
  --server.headless=true
