#\!/bin/bash
#SBATCH --job-name=d4_analysis
#SBATCH --partition=rome
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/projects/prjs1800/msc-thesis/results/day4/logs/analysis_%j.log

module purge
module load 2025
module load Anaconda3/2025.06-1

source /projects/prjs1800/venvs/FlashRAG-venv/bin/activate
export PYTHONPATH="/projects/prjs1800/external/FlashRAG:$PYTHONPATH"

cd /projects/prjs1800/msc-thesis

IRCOT_HQA=$(ls -td /projects/prjs1800/results/day4/hotpotqa_*ircot*hotpotqa*/intermediate_data.json 2>/dev/null | head -1)
IRCOT_MSQ=$(ls -td /projects/prjs1800/results/day4/musique_*ircot*musique*/intermediate_data.json 2>/dev/null | head -1)
BASELINE_HQA="/projects/prjs1800/results/day1/hotpotqa_2026_02_06_13_47_standard_rag_qwen25_hotpotqa/intermediate_data.json"
BASELINE_MSQ="/projects/prjs1800/results/day1/musique_2026_02_06_14_08_standard_rag_qwen25_musique/intermediate_data.json"
FLARE_HQA=$(ls -td /projects/prjs1800/results/day4/hotpotqa_*flare*hotpotqa*/intermediate_data.json 2>/dev/null | head -1)
FLARE_MSQ=$(ls -td /projects/prjs1800/results/day4/musique_*flare*musique*/intermediate_data.json 2>/dev/null | head -1)

echo "IRCoT HotpotQA: $IRCOT_HQA"
echo "IRCoT MuSiQue: $IRCOT_MSQ"

CMD="python scripts/analyze_ircot_retrieval.py"
if [ -n "$IRCOT_HQA" ]; then CMD="$CMD --ircot_hotpotqa $IRCOT_HQA"; fi
if [ -n "$IRCOT_MSQ" ]; then CMD="$CMD --ircot_musique $IRCOT_MSQ"; fi
if [ -n "$FLARE_HQA" ]; then CMD="$CMD --flare_hotpotqa $FLARE_HQA"; fi
if [ -n "$FLARE_MSQ" ]; then CMD="$CMD --flare_musique $FLARE_MSQ"; fi
CMD="$CMD --baseline_hotpotqa $BASELINE_HQA"
CMD="$CMD --baseline_musique $BASELINE_MSQ"
CMD="$CMD --output_dir /projects/prjs1800/msc-thesis/analysis/outputs"

echo "Running: $CMD"
eval $CMD
