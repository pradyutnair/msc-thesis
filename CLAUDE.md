# MSc Thesis: Multi-Agent RAG

## Project Structure
- Base directory: `/projects/prjs1800/msc-thesis/`
- A-RAG reproduction: `01-arag-reproduction/`
- Models: `/projects/prjs1800/models/`
- Compute: Snellius Dutch National Supercomputer (A100/H100 GPUs)

## Evaluation Pipeline (MANDATORY)

When implementing or testing any new architecture, ALWAYS follow this
progressive evaluation pipeline. Never skip stages.

### Stage 1: Smoke Test (5-10 questions)
- Run on 10 handpicked questions from each dataset before anything else
- Purpose: verify the code runs end-to-end without errors
- Check: predictions file is generated, format is correct, no crashes
- If smoke test fails, fix before proceeding

### Stage 2: Pilot (10 questions)
- Run on first 10 questions of each dataset
- Compare against the user-specified baseline (ask which baseline if not stated)
- Only proceed to Stage 3 if pilot results > baseline on all 3 datasets

### Stage 3: Medium evaluation (100 questions)  
- Run on first 100 questions of each dataset
- Compare against baseline on LLM-Acc and Contain-Acc
- Only proceed to Stage 4 if results > baseline on all 3 datasets

### Stage 4: Full evaluation (1000 questions)
- Uses the fixed ARAG 1000-question packs per dataset:
  - `/projects/prjs1800/external/arag/data/hotpotqa/questions.json`
  - `/projects/prjs1800/external/arag/data/musique/questions.json`
  - `/projects/prjs1800/external/arag/data/2wikimultihop/questions.json`
- Submit as parallel batch jobs: ~200 questions per job, 5 jobs per dataset
- Merge predictions after all jobs complete
- Evaluate merged predictions

## Evaluation Configuration (ALL experiments)

- **LLM Judge**: DeepSeek-R1-Distill-Qwen-32B (ALWAYS, no exceptions)
- **Metrics**: LLM-Accuracy, Contain-Accuracy (primary); EM/F1 for non-agentic only
- **Embeddings**: E5-base-v2 (default unless testing alternatives)
- **Index**: ARAG per-dataset chunked indices

## SLURM Job Submission

For Stage 4 batch jobs:
```bash
# Template: submit_batch.sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
```

Split questions with: `--start-idx N --end-idx M`
Merge with: `scripts/merge_predictions.py results/*/predictions_batch_*.jsonl`

## Current Baselines (reference numbers, n=1000)

All agentic runs (E1–E4) have EM=0 due to verbose output format; use LLM-Acc/Contain as primary metrics.
SAGE (04-sage) produces clean span answers and yields valid EM scores.

### Non-agentic baselines (B0) — EM/F1 valid

| Config | Dataset | EM | F1 | LLM-Acc | Contain |
|--------|---------|-----|-----|---------|---------|
| B0-7B (Qwen2.5-7B) | hotpotqa | 45.0% | 56.6% | 52.6% | 51.2% |
| B0-7B | musique | 15.1% | 24.2% | 24.8% | 19.3% |
| B0-7B | 2wikimultihop | 29.9% | 38.5% | 36.2% | 39.7% |
| B0-7B **mean** | — | **30.0%** | **39.7%** | 37.9% | 36.7% |
| B0-8B (Qwen3-8B) | hotpotqa | 41.4% | 56.7% | 59.7% | 58.2% |
| B0-8B | musique | 14.1% | 24.9% | 26.8% | 22.4% |
| B0-8B | 2wikimultihop | 34.1% | 46.6% | 43.7% | 50.7% |
| B0-8B **mean** | — | **29.9%** | **42.7%** | **43.4%** | **43.8%** |
| B0-30B (Qwen3-30B) | hotpotqa | 34.0% | 52.1% | 58.8% | 58.7% |
| B0-30B | musique | 8.2% | 19.8% | 28.3% | 23.9% |
| B0-30B | 2wikimultihop | 18.3% | 36.3% | 40.0% | 51.2% |
| B0-30B **mean** | — | **20.2%** | **36.1%** | 42.4% | 44.6% |

### Agentic ARAG (01-arag-reproduction) — LLM-Acc/Contain primary

| Config | Dataset | LLM-Acc | Contain | avg_loops |
|--------|---------|---------|---------|-----------|
| E1 (Qwen2.5-7B + E5) | hotpotqa | 65.9% | 68.3% | 3.33 |
| E1 | musique | 31.9% | 32.2% | 3.99 |
| E1 | 2wikimultihop | 48.4% | 53.1% | 3.48 |
| E1 **mean** | — | **48.7%** | **51.2%** | — |
| E2 (Qwen3-8B + E5) | hotpotqa | 70.0% | 59.4% | 2.44 |
| E2 | musique | 46.2% | 27.1% | 2.65 |
| E2 | 2wikimultihop | 63.6% | 54.4% | 2.78 |
| E2 **mean** | — | **45.7%** | **47.0%** | — |
| E3 (Qwen3-8B + QwenEmb) | hotpotqa | — | 59.0% | 2.53 |
| E3 | musique | — | 24.6% | 2.63 |
| E3 | 2wikimultihop | — | 52.2% | 2.84 |
| E3 **mean** | — | **40.1%** | **42.0%** | — |
| E4 (Qwen3-30B + E5) | hotpotqa | 77.1% | 67.7% | 2.66 |
| E4 | musique | 53.3% | 34.4% | 2.98 |
| E4 | 2wikimultihop | 70.7% | 63.9% | 3.05 |
| E4 **mean** | — | **53.7%** | **55.3%** | — |

### SAGE multi-agent (04-sage-autonomous) — EM/F1 valid

| Config | Dataset | EM | F1 | LLM-Acc | Contain |
|--------|---------|-----|-----|---------|---------|
| SAGE v3r2 | hotpotqa | **58.8%** | **72.5%** | 73.3% | 79.2% |
| SAGE v3r2 | musique | **35.5%** | **47.3%** | 50.6% | 53.8% |
| SAGE v3r2 | 2wikimultihop | **67.5%** | **74.7%** | 76.8% | 81.7% |
| SAGE v3r2 **mean** | — | **54.0%** | **64.8%** | **66.9%** | **71.6%** |
| auto_1k_final | hotpotqa | **58.1%** | **71.7%** | 73.5% | 77.3% |
| auto_1k_final | musique | **37.2%** | **47.6%** | 49.5% | 49.7% |
| auto_1k_final | 2wikimultihop | **65.6%** | **73.6%** | 77.6% | 79.2% |
| auto_1k_final **mean** | — | **53.6%** | **64.3%** | **66.9%** | **68.7%** |

### Summary (mean across 3 datasets)

| Config | EM | LLM-Acc | Contain |
|--------|----|---------|---------|
| B0-8B (non-agentic) | 29.9% | 43.4% | 43.8% |
| E2 (agentic Q3-8B) | — | 45.7% | 47.0% |
| E4 (agentic Q3-30B) | — | 53.7% | 55.3% |
| Paper (GPT-4o-mini) | — | 61.1% | 59.1% |
| **SAGE v3r2** | **54.0%** | **66.9%** | **71.6%** |
| **auto_1k_final** | **53.6%** | **66.9%** | **68.7%** |

## Code Standards
- Always include SHA256 fingerprint of question sets in results README
- Save all predictions to `predictions.jsonl` (one JSON object per line)
- Strip `<think>` and `<thnk>` tags from model outputs before evaluation
- Use vLLM for all inference (OpenAI-compatible endpoint)