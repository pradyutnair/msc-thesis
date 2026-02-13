#!/bin/bash
#SBATCH --job-name=download_nq
#SBATCH --partition=cbuild
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=jobs/output/download_nq_%j.log
#SBATCH --error=jobs/output/download_nq_%j.log

# Natural Questions Dataset Download and Preparation
# Dataset: Google's question answering dataset from real queries

echo "=========================================="
echo "Starting Natural Questions Dataset Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load 2023
module load Miniconda3/23.5.2-0

# Activate conda environment
source activate /projects/prjs1800/conda_envs/multi_agentic_rag

# Hugging Face cache: all datasets download only to this directory
HF_CACHE="/projects/prjs1800/.cache/huggingface"
export HF_HOME="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"
export TRANSFORMERS_CACHE="$HF_CACHE/transformers"
mkdir -p "$HF_DATASETS_CACHE"
echo "Using HF cache: $HF_DATASETS_CACHE"

DATASET_DIR="/projects/prjs1800/datasets/natural_questions"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

echo "Downloading Natural Questions (train, validation, test) to $HF_DATASETS_CACHE..."

# Create download script: train, validation, test; cache under /projects/prjs1800/.cache/huggingface
cat > download_nq.py << 'EOF'
from datasets import load_dataset
import json
import os

# Cache is set via HF_DATASETS_CACHE=/projects/prjs1800/.cache/huggingface/datasets
print("Loading Natural Questions (train, validation, test)...")

dataset = load_dataset("google-research-datasets/natural_questions")
splits = list(dataset.keys())
print(f"Splits available: {splits}")

def to_item(item, split_name, idx):
    q = item.get("question")
    text = q.get("text", "") if isinstance(q, dict) else str(q or "")
    doc = item.get("document") or {}
    return {
        "id": item.get("id", f"{split_name}_{idx}"),
        "question": text,
        "annotations": item.get("annotations", []),
        "document_title": doc.get("title", "") if isinstance(doc, dict) else "",
        "document_url": doc.get("url", "") if isinstance(doc, dict) else "",
    }

for split_name in splits:
    split_data = dataset[split_name]
    n = len(split_data)
    print(f"\nProcessing {split_name}: {n} examples...")
    data_list = []
    for idx, item in enumerate(split_data):
        data_list.append(to_item(item, split_name, idx))
        if (idx + 1) % 10000 == 0:
            print(f"  {split_name}: {idx + 1}/{n}")
    out_file = f"natural_questions_{split_name}.json"
    with open(out_file, "w") as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved {out_file} ({len(data_list)} examples)")

# Small and tiny from validation (or first available split)
first_split = dataset["validation"] if "validation" in dataset else dataset[splits[0]]
data_small = [to_item(first_split[i], "small", i) for i in range(min(1000, len(first_split)))]
with open("natural_questions_small.json", "w") as f:
    json.dump(data_small[:100], f, indent=2)
with open("natural_questions_tiny.json", "w") as f:
    json.dump(data_small[:10], f, indent=2)
print("Created natural_questions_small.json (100) and natural_questions_tiny.json (10)")

print("\nNatural Questions download complete!")
EOF

# Run download script
python download_nq.py

# Create dataset info file
cat > dataset_info.txt << EOF
Natural Questions Dataset Information
======================================

Dataset Location: $DATASET_DIR
Hugging Face cache: $HF_CACHE

Files:
- natural_questions_train.json: Training set
- natural_questions_validation.json: Validation set
- natural_questions_test.json: Test set (if available)
- natural_questions_small.json: Small split (100 examples)
- natural_questions_tiny.json: Tiny split (10 examples)

Dataset Description:
Natural Questions (NQ) is a question answering dataset containing real anonymized, 
aggregated queries issued to the Google search engine. Each question is paired with 
a Wikipedia page and annotations indicating short and long answer spans.

Features:
- Real user queries from Google Search
- 307K+ training examples
- Wikipedia articles as context
- Short and long answer annotations
- Some questions may not have answers in the provided context

Answer Types:
- Short answers: Entity or phrase
- Long answers: Paragraph or table
- No answer: Question not answerable from context

Citation:
Kwiatkowski et al. (2019). Natural Questions: A Benchmark for Question Answering Research.
TACL 2019.

Downloaded: $(date)
EOF

echo "=========================================="
echo "Dataset download complete!"
echo "Location: $DATASET_DIR"
echo "See dataset_info.txt for details"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
