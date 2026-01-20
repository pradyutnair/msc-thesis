#!/bin/bash
#SBATCH --job-name=download_triviaqa
#SBATCH --partition=cbuild
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/download_triviaqa_%j.out
#SBATCH --error=jobs/logs/download_triviaqa_%j.err

# TriviaQA Dataset Download and Preparation
# Dataset: Large-scale reading comprehension dataset

echo "=========================================="
echo "Starting TriviaQA Dataset Download"
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
DATASET_DIR="$HOME/datasets/triviaqa"
mkdir -p $DATASET_DIR

cd $DATASET_DIR

echo "Downloading TriviaQA dataset from Hugging Face..."

# Create download script
cat > download_triviaqa.py << 'EOF'
from datasets import load_dataset
import json
import os

print("Loading TriviaQA dataset from Hugging Face...")
print("Note: This may take a while as TriviaQA is a large dataset")

# Load dataset (unfiltered version for RAG)
dataset = load_dataset("trivia_qa", "unfiltered")

print(f"\nDataset splits: {list(dataset.keys())}")

# Save each split
for split_name, split_data in dataset.items():
    print(f"\nProcessing {split_name} split...")
    print(f"Number of examples: {len(split_data)}")
    
    # Convert to list of dictionaries
    data_list = []
    for idx, item in enumerate(split_data):
        # Extract key fields
        data_item = {
            'question_id': item.get('question_id', f'{split_name}_{idx}'),
            'question': item.get('question', ''),
            'answer': item.get('answer', {}),
            'question_source': item.get('question_source', ''),
            'entity_pages': item.get('entity_pages', {}),
            'search_results': item.get('search_results', {})
        }
        data_list.append(data_item)
        
        # Progress indicator for large datasets
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} examples...")
    
    # Save as JSON
    output_file = f"triviaqa_{split_name}.json"
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=2)
    
    print(f"Saved to: {output_file}")
    
    # Print sample
    if data_list:
        print(f"\nSample from {split_name}:")
        sample = data_list[0]
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Answer: {sample.get('answer', 'N/A')}")

# Create smaller splits for testing
print("\n========================================")
print("Creating test splits...")
print("========================================")

# Use validation split for creating smaller versions
if 'validation' in dataset:
    data = []
    for item in dataset['validation']:
        data_item = {
            'question_id': item.get('question_id', ''),
            'question': item.get('question', ''),
            'answer': item.get('answer', {}),
            'question_source': item.get('question_source', ''),
        }
        data.append(data_item)
else:
    # Fallback to train split
    data = []
    for idx, item in enumerate(dataset['train']):
        if idx >= 1000:  # Only take first 1000 for creating splits
            break
        data_item = {
            'question_id': item.get('question_id', ''),
            'question': item.get('question', ''),
            'answer': item.get('answer', {}),
        }
        data.append(data_item)

# Small split (100 examples)
small_split = data[:100]
with open('triviaqa_small.json', 'w') as f:
    json.dump(small_split, f, indent=2)
print(f"Created small split: 100 examples")

# Tiny split (10 examples)
tiny_split = data[:10]
with open('triviaqa_tiny.json', 'w') as f:
    json.dump(tiny_split, f, indent=2)
print(f"Created tiny split: 10 examples")

print("\n========================================")
print("TriviaQA download complete!")
print("========================================")
EOF

# Run download script
python download_triviaqa.py

# Create dataset info file
cat > dataset_info.txt << EOF
TriviaQA Dataset Information
============================

Dataset Location: $DATASET_DIR

Files:
- triviaqa_train.json: Training set
- triviaqa_validation.json: Validation set
- triviaqa_test.json: Test set (if available)
- triviaqa_small.json: Small split (100 examples)
- triviaqa_tiny.json: Tiny split (10 examples)

Dataset Description:
TriviaQA is a large-scale reading comprehension dataset containing over 650K 
question-answer-evidence triples. Questions are authored by trivia enthusiasts 
and independently gathered evidence documents provide distant supervision.

Features:
- 650K+ question-answer pairs
- Multiple evidence documents per question
- Web and Wikipedia evidence sources
- Unfiltered version includes all evidence documents

Citation:
Joshi et al. (2017). TriviaQA: A Large Scale Distantly Supervised Challenge Dataset 
for Reading Comprehension. ACL 2017.

Downloaded: $(date)
EOF

echo "=========================================="
echo "Dataset download complete!"
echo "Location: $DATASET_DIR"
echo "See dataset_info.txt for details"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
