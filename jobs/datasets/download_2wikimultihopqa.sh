#!/bin/bash
#SBATCH --job-name=download_2wikimhqa
#SBATCH --partition=cbuild
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=jobs/logs/download_2wikimhqa_%j.out
#SBATCH --error=jobs/logs/download_2wikimhqa_%j.err

# 2WikiMultiHopQA Dataset Download and Preparation
# Dataset: Multi-hop question answering with reasoning chains

echo "=========================================="
echo "Starting 2WikiMultiHopQA Dataset Download"
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

# Set dataset directory
DATASET_DIR="/projects/prjs1800/datasets/2wikimultihopqa"
mkdir -p $DATASET_DIR

cd $DATASET_DIR

echo "Downloading 2WikiMultiHopQA dataset from Hugging Face..."

# Create download script
cat > download_2wiki.py << 'EOF'
from datasets import load_dataset
import json
import os

print("Loading 2WikiMultiHopQA dataset from Hugging Face...")

# Load dataset
dataset = load_dataset("THUDM/2WikiMultihopQA")

print(f"\nDataset splits: {list(dataset.keys())}")

# Save each split
for split_name, split_data in dataset.items():
    print(f"\nProcessing {split_name} split...")
    print(f"Number of examples: {len(split_data)}")
    
    # Convert to list of dictionaries
    data_list = []
    for item in split_data:
        data_list.append(dict(item))
    
    # Save as JSON
    output_file = f"2wikimultihopqa_{split_name}.json"
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=2)
    
    print(f"Saved to: {output_file}")
    
    # Print sample
    if data_list:
        print(f"\nSample from {split_name}:")
        sample = data_list[0]
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Answer: {sample.get('answer', 'N/A')}")
        print(f"Type: {sample.get('type', 'N/A')}")

# Create smaller splits for testing
print("\n========================================")
print("Creating test splits...")
print("========================================")

# Load train or dev split for creating smaller versions
if 'train' in dataset:
    data = [dict(item) for item in dataset['train']]
elif 'dev' in dataset:
    data = [dict(item) for item in dataset['dev']]
else:
    data = [dict(item) for item in list(dataset.values())[0]]

# Small split (100 examples)
small_split = data[:100]
with open('2wikimultihopqa_small.json', 'w') as f:
    json.dump(small_split, f, indent=2)
print(f"Created small split: 100 examples")

# Tiny split (10 examples)
tiny_split = data[:10]
with open('2wikimultihopqa_tiny.json', 'w') as f:
    json.dump(tiny_split, f, indent=2)
print(f"Created tiny split: 10 examples")

print("\n========================================")
print("2WikiMultiHopQA download complete!")
print("========================================")
EOF

# Run download script
python download_2wiki.py

# Create dataset info file
cat > dataset_info.txt << EOF
2WikiMultiHopQA Dataset Information
===================================

Dataset Location: $DATASET_DIR

Files:
- 2wikimultihopqa_train.json: Training set
- 2wikimultihopqa_dev.json: Development set
- 2wikimultihopqa_test.json: Test set
- 2wikimultihopqa_small.json: Small split (100 examples)
- 2wikimultihopqa_tiny.json: Tiny split (10 examples)

Dataset Description:
2WikiMultiHopQA is a multi-hop question answering dataset with explicit 
reasoning chains. It contains questions that require reasoning over multiple 
Wikipedia paragraphs with annotated reasoning paths.

Question Types:
- Inference: Questions requiring multi-step inference
- Comparison: Questions comparing entities
- Bridge Entity: Questions with bridge entities

Features:
- Explicit reasoning chains
- Supporting facts annotations
- Evidence paragraphs

Citation:
Ho et al. (2020). Constructing A Multi-hop QA Dataset for Comprehensive Evaluation 
of Reasoning Steps. COLING 2020.

Downloaded: $(date)
EOF

echo "=========================================="
echo "Dataset download complete!"
echo "Location: $DATASET_DIR"
echo "See dataset_info.txt for details"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
