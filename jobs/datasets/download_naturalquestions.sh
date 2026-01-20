#!/bin/bash
#SBATCH --job-name=download_nq
#SBATCH --partition=cbuild
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/download_nq_%j.out
#SBATCH --error=jobs/logs/download_nq_%j.err

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
source activate multi_agentic_rag

# Set dataset directory
DATASET_DIR="$HOME/datasets/natural_questions"
mkdir -p $DATASET_DIR

cd $DATASET_DIR

echo "Downloading Natural Questions dataset from Hugging Face..."

# Create download script
cat > download_nq.py << 'EOF'
from datasets import load_dataset
import json
import os

print("Loading Natural Questions dataset from Hugging Face...")
print("Note: This is a large dataset and may take significant time")

# Load dataset
dataset = load_dataset("google-research-datasets/natural_questions")

print(f"\nDataset splits: {list(dataset.keys())}")

# Save each split
for split_name, split_data in dataset.items():
    print(f"\nProcessing {split_name} split...")
    print(f"Number of examples: {len(split_data)}")
    
    # Convert to list of dictionaries
    data_list = []
    for idx, item in enumerate(split_data):
        # Extract key fields for RAG
        data_item = {
            'id': item.get('id', f'{split_name}_{idx}'),
            'question': item.get('question', {}).get('text', ''),
            'annotations': item.get('annotations', []),
            'document_title': item.get('document', {}).get('title', ''),
            'document_url': item.get('document', {}).get('url', ''),
        }
        data_list.append(data_item)
        
        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} examples...")
    
    # Save as JSON
    output_file = f"natural_questions_{split_name}.json"
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=2)
    
    print(f"Saved to: {output_file}")
    
    # Print sample
    if data_list:
        print(f"\nSample from {split_name}:")
        sample = data_list[0]
        print(f"Question: {sample.get('question', 'N/A')[:100]}...")
        print(f"Document Title: {sample.get('document_title', 'N/A')}")

# Create smaller splits for testing
print("\n========================================")
print("Creating test splits...")
print("========================================")

# Use validation split for creating smaller versions
if 'validation' in dataset:
    split_to_use = 'validation'
elif 'train' in dataset:
    split_to_use = 'train'
else:
    split_to_use = list(dataset.keys())[0]

data = []
for idx, item in enumerate(dataset[split_to_use]):
    if idx >= 1000:  # Only take first 1000
        break
    data_item = {
        'id': item.get('id', ''),
        'question': item.get('question', {}).get('text', ''),
        'annotations': item.get('annotations', []),
        'document_title': item.get('document', {}).get('title', ''),
    }
    data.append(data_item)

# Small split (100 examples)
small_split = data[:100]
with open('natural_questions_small.json', 'w') as f:
    json.dump(small_split, f, indent=2)
print(f"Created small split: 100 examples")

# Tiny split (10 examples)
tiny_split = data[:10]
with open('natural_questions_tiny.json', 'w') as f:
    json.dump(tiny_split, f, indent=2)
print(f"Created tiny split: 10 examples")

print("\n========================================")
print("Natural Questions download complete!")
print("========================================")
EOF

# Run download script
python download_nq.py

# Create dataset info file
cat > dataset_info.txt << EOF
Natural Questions Dataset Information
======================================

Dataset Location: $DATASET_DIR

Files:
- natural_questions_train.json: Training set
- natural_questions_validation.json: Validation set
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
