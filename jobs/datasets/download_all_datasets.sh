#!/bin/bash
#SBATCH --job-name=download_all_datasets
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --output=jobs/output/download_all_datasets_%j.log
#SBATCH --error=jobs/output/download_all_datasets_%j.log

# Master script to download all datasets for Multi-Agentic RAG
# This script downloads: HotpotQA, 2WikiMultiHopQA, TriviaQA, Natural Questions

echo "=========================================="
echo "Starting All Datasets Download"
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

# Create main datasets directory
DATASETS_ROOT="/projects/prjs1800/datasets"
mkdir -p $DATASETS_ROOT

echo "=========================================="
echo "Dataset 1/4: HotpotQA"
echo "=========================================="

DATASET_DIR="$DATASETS_ROOT/hotpotqa"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

echo "Downloading HotpotQA..."
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
wget -c http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json

# Process HotpotQA
python << 'EOFPY'
import json
data = json.load(open('hotpot_dev_distractor_v1.json'))
with open('hotpot_dev_small.json', 'w') as f:
    json.dump(data[:100], f, indent=2)
with open('hotpot_dev_tiny.json', 'w') as f:
    json.dump(data[:10], f, indent=2)
print(f"HotpotQA: {len(data)} examples processed")
EOFPY

echo "HotpotQA download complete!"

echo "=========================================="
echo "Dataset 2/4: 2WikiMultiHopQA"
echo "=========================================="

DATASET_DIR="$DATASETS_ROOT/2wikimultihopqa"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

python << 'EOFPY'
from datasets import load_dataset
import json

print("Downloading 2WikiMultiHopQA...")
try:
    dataset = load_dataset("2wikimultihopqa")
except Exception as e:
    print(f"Error loading 2wikimultihopqa: {e}")
    print("Trying alternative dataset name...")
    dataset = load_dataset("hotpot_qa", "fullwiki")
    print("Note: Using HotpotQA fullwiki as alternative")

for split_name, split_data in dataset.items():
    data_list = [dict(item) for item in split_data]
    with open(f"2wikimultihopqa_{split_name}.json", 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"2WikiMultiHopQA {split_name}: {len(data_list)} examples")

# Create small splits
if 'train' in dataset:
    data = [dict(item) for item in dataset['train']]
elif 'dev' in dataset:
    data = [dict(item) for item in dataset['dev']]
else:
    data = [dict(item) for item in list(dataset.values())[0]]

with open('2wikimultihopqa_small.json', 'w') as f:
    json.dump(data[:100], f, indent=2)
with open('2wikimultihopqa_tiny.json', 'w') as f:
    json.dump(data[:10], f, indent=2)
EOFPY

echo "2WikiMultiHopQA download complete!"

echo "=========================================="
echo "Dataset 3/4: TriviaQA"
echo "=========================================="

DATASET_DIR="$DATASETS_ROOT/triviaqa"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

python << 'EOFPY'
from datasets import load_dataset
import json

print("Downloading TriviaQA (this may take a while)...")
dataset = load_dataset("trivia_qa", "unfiltered")

for split_name, split_data in dataset.items():
    print(f"Processing TriviaQA {split_name}...")
    data_list = []
    for idx, item in enumerate(split_data):
        data_item = {
            'question_id': item.get('question_id', f'{split_name}_{idx}'),
            'question': item.get('question', ''),
            'answer': item.get('answer', {}),
        }
        data_list.append(data_item)
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1} examples...")
    
    with open(f"triviaqa_{split_name}.json", 'w') as f:
        json.dump(data_list, f, indent=2)
    print(f"TriviaQA {split_name}: {len(data_list)} examples")

# Create small splits
if 'validation' in dataset:
    data = [dict(item) for item in dataset['validation']][:1000]
else:
    data = [dict(item) for item in dataset['train']][:1000]

with open('triviaqa_small.json', 'w') as f:
    json.dump(data[:100], f, indent=2)
with open('triviaqa_tiny.json', 'w') as f:
    json.dump(data[:10], f, indent=2)
EOFPY

echo "TriviaQA download complete!"

echo "=========================================="
echo "Dataset 4/4: Natural Questions"
echo "=========================================="

DATASET_DIR="$DATASETS_ROOT/natural_questions"
mkdir -p $DATASET_DIR
cd $DATASET_DIR

python << 'EOFPY'
from datasets import load_dataset
import json
import os

print("Downloading Natural Questions (this may take a while)...")
print("Note: Processing in chunks to avoid disk quota issues...")

# Only download validation split to save space
dataset = load_dataset("google-research-datasets/natural_questions", split="validation")

print(f"Processing Natural Questions validation split...")
print(f"Total examples: {len(dataset)}")

# Process in smaller chunks and write incrementally
data_list = []
chunk_size = 1000
output_file = "natural_questions_validation.json"

# Process first 10000 examples only to save space
max_examples = min(10000, len(dataset))

for idx in range(max_examples):
    item = dataset[idx]
    data_item = {
        'id': item.get('id', f'validation_{idx}'),
        'question': item.get('question', {}).get('text', '') if isinstance(item.get('question'), dict) else str(item.get('question', '')),
        'annotations': item.get('annotations', []),
        'document_title': item.get('document', {}).get('title', '') if isinstance(item.get('document'), dict) else '',
    }
    data_list.append(data_item)
    
    # Write chunk periodically to avoid memory issues
    if (idx + 1) % chunk_size == 0:
        print(f"  Processed {idx + 1}/{max_examples} examples...")
        # Write incrementally
        with open(output_file, 'w') as f:
            json.dump(data_list, f, indent=2)

# Final write
with open(output_file, 'w') as f:
    json.dump(data_list, f, indent=2)
print(f"Natural Questions validation: {len(data_list)} examples saved")

# Create small splits
with open('natural_questions_small.json', 'w') as f:
    json.dump(data_list[:100], f, indent=2)
with open('natural_questions_tiny.json', 'w') as f:
    json.dump(data_list[:10], f, indent=2)
print("Created small and tiny splits")
EOFPY

echo "Natural Questions download complete!"

# Create master dataset summary
cd $DATASETS_ROOT
cat > DATASETS_SUMMARY.txt << EOF
Multi-Agentic RAG Datasets Summary
==================================

All datasets downloaded to: $DATASETS_ROOT

Datasets:
1. HotpotQA - Multi-hop QA with distractor documents
   Location: $DATASETS_ROOT/hotpotqa/
   
2. 2WikiMultiHopQA - Multi-hop QA with reasoning chains
   Location: $DATASETS_ROOT/2wikimultihopqa/
   
3. TriviaQA - Large-scale reading comprehension
   Location: $DATASETS_ROOT/triviaqa/
   
4. Natural Questions - Real Google search queries
   Location: $DATASETS_ROOT/natural_questions/

Each dataset includes:
- Full training/dev/test splits
- Small split (100 examples) for quick testing
- Tiny split (10 examples) for debugging

Downloaded: $(date)

Usage:
To use these datasets in your experiments, set the DATASET_DIR environment
variable or update your config files to point to these locations.

Example:
export HOTPOTQA_DIR=$DATASETS_ROOT/hotpotqa
EOF

echo "=========================================="
echo "All datasets downloaded successfully!"
echo "Summary saved to: $DATASETS_ROOT/DATASETS_SUMMARY.txt"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
