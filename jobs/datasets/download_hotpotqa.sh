#!/bin/bash
#SBATCH --job-name=download_hotpotqa
#SBATCH --partition=cbuild
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=jobs/output/download_hotpotqa_%j.log
#SBATCH --error=jobs/output/download_hotpotqa_%j.log

# HotpotQA Dataset Download and Preparation
# Dataset: Multi-hop question answering dataset

echo "=========================================="
echo "Starting HotpotQA Dataset Download"
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
DATASET_DIR="/projects/prjs1800/datasets/hotpotqa"
mkdir -p $DATASET_DIR

cd $DATASET_DIR

echo "Downloading HotpotQA dataset..."

# Download training set
echo "Downloading training set..."
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json

# Download dev set (distractor)
echo "Downloading dev set (distractor)..."
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json

# Download dev set (fullwiki)
echo "Downloading dev set (fullwiki)..."
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json

# Download test set (fullwiki)
echo "Downloading test set (fullwiki)..."
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json

echo "=========================================="
echo "Processing and validating dataset..."
echo "=========================================="

# Create a Python script to process and validate the dataset
cat > process_hotpotqa.py << 'EOF'
import json
import os
from collections import Counter

def load_and_analyze(filepath):
    """Load and analyze a HotpotQA file."""
    print(f"\nAnalyzing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # Analyze question types
    if data and isinstance(data[0], dict):
        if 'type' in data[0]:
            types = Counter([item['type'] for item in data])
            print(f"Question types: {dict(types)}")
        
        if 'level' in data[0]:
            levels = Counter([item['level'] for item in data])
            print(f"Difficulty levels: {dict(levels)}")
        
        # Sample question
        print(f"\nSample question: {data[0].get('question', 'N/A')}")
        print(f"Sample answer: {data[0].get('answer', 'N/A')}")
    
    return data

def create_splits(data, output_dir):
    """Create smaller splits for testing."""
    print(f"\nCreating test splits...")
    
    # Create a small dev set (100 examples)
    small_dev = data[:100]
    with open(os.path.join(output_dir, 'hotpot_dev_small.json'), 'w') as f:
        json.dump(small_dev, f, indent=2)
    print(f"Created small dev set: 100 examples")
    
    # Create a tiny dev set (10 examples) for quick testing
    tiny_dev = data[:10]
    with open(os.path.join(output_dir, 'hotpot_dev_tiny.json'), 'w') as f:
        json.dump(tiny_dev, f, indent=2)
    print(f"Created tiny dev set: 10 examples")

# Process all datasets
files = [
    'hotpot_train_v1.1.json',
    'hotpot_dev_distractor_v1.json',
    'hotpot_dev_fullwiki_v1.json',
    'hotpot_test_fullwiki_v1.json'
]

for filename in files:
    data = load_and_analyze(filename)
    
    # Create splits for dev distractor set
    if filename == 'hotpot_dev_distractor_v1.json' and data:
        create_splits(data, '.')

print("\n========================================")
print("HotpotQA dataset processing complete!")
print("========================================")
EOF

# Run processing script
python process_hotpotqa.py

# Create dataset info file
cat > dataset_info.txt << EOF
HotpotQA Dataset Information
============================

Dataset Location: $DATASET_DIR

Files:
- hotpot_train_v1.1.json: Training set
- hotpot_dev_distractor_v1.json: Dev set with distractor paragraphs
- hotpot_dev_fullwiki_v1.json: Dev set with full Wikipedia
- hotpot_test_fullwiki_v1.json: Test set with full Wikipedia
- hotpot_dev_small.json: Small dev set (100 examples)
- hotpot_dev_tiny.json: Tiny dev set (10 examples)

Dataset Description:
HotpotQA is a multi-hop question answering dataset with 113k Wikipedia-based 
question-answer pairs. Questions require reasoning over multiple documents.

Question Types:
- Bridge: Questions that require finding a bridge entity
- Comparison: Questions that require comparing two entities

Difficulty Levels:
- Easy
- Medium
- Hard

Citation:
Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
EMNLP 2018.

Downloaded: $(date)
EOF

echo "=========================================="
echo "Dataset download complete!"
echo "Location: $DATASET_DIR"
echo "See dataset_info.txt for details"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
