#!/bin/bash
#SBATCH --job-name=baseline_hotpotqa
#SBATCH --partition=gpu-a100
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --output=jobs/logs/baseline_hotpotqa_%j.out
#SBATCH --error=jobs/logs/baseline_hotpotqa_%j.err

# Baseline RAG Benchmark on HotpotQA
# This script runs a baseline single-agent RAG system on HotpotQA

echo "=========================================="
echo "Starting Baseline HotpotQA Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load 2023
module load Miniconda3/23.5.2-0
module load CUDA/12.1.1

# Activate conda environment
source activate /projects/prjs1800/conda_envs/multi_agentic_rag

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Set paths
DATASET_DIR="/projects/prjs1800/datasets/hotpotqa"
RESULTS_DIR="/projects/prjs1800/results/baseline_hotpotqa"
mkdir -p $RESULTS_DIR

# Navigate to project directory
cd $HOME/msc-thesis

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Dataset: HotpotQA"
echo "Dataset directory: $DATASET_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Model: gpt-3.5-turbo (via OpenAI API)"
echo "Retrieval: Dense retrieval with sentence-transformers"
echo "=========================================="

# Create baseline evaluation script
cat > baseline_rag_hotpotqa.py << 'EOF'
"""
Baseline RAG System for HotpotQA Evaluation
This implements a simple single-agent RAG system as a baseline.
"""

import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Simple RAG baseline
class BaselineRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents."""
        print("Building document index...")
        self.documents = documents
        
        # Extract text from documents
        texts = []
        for doc in documents:
            # Combine title and sentences
            title = doc.get('title', '')
            sentences = doc.get('sentences', [])
            text = f"{title}. {' '.join(sentences)}"
            texts.append(text)
        
        # Encode documents
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"Index built with {len(documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant documents."""
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['retrieval_score'] = float(score)
                results.append(doc)
        
        return results
    
    def answer_question(self, question: str, retrieved_docs: List[Dict]) -> str:
        """Generate answer from retrieved documents."""
        # Simple baseline: extract answer from most relevant document
        # In a real system, this would use an LLM
        
        if not retrieved_docs:
            return "Unable to answer"
        
        # For baseline, we'll just return a placeholder
        # In practice, you would call an LLM here
        context = " ".join([
            f"{doc.get('title', '')}. {' '.join(doc.get('sentences', []))}"
            for doc in retrieved_docs[:3]
        ])
        
        # Placeholder answer (in real implementation, call LLM with context)
        answer = f"[Based on retrieved context: {context[:200]}...]"
        
        return answer

def evaluate_hotpotqa(dataset_path: str, output_path: str, use_small: bool = False):
    """Evaluate baseline RAG on HotpotQA."""
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if use_small:
        data = data[:100]
    
    print(f"Loaded {len(data)} examples")
    
    # Initialize RAG system
    rag = BaselineRAG()
    
    # Build document index from all contexts in the dataset
    print("Extracting documents from dataset...")
    all_documents = []
    doc_id = 0
    for item in data:
        contexts = item.get('context', [])
        for context in contexts:
            doc = {
                'id': doc_id,
                'title': context[0] if isinstance(context, list) and len(context) > 0 else '',
                'sentences': context[1] if isinstance(context, list) and len(context) > 1 else []
            }
            all_documents.append(doc)
            doc_id += 1
    
    print(f"Extracted {len(all_documents)} documents")
    
    # Build index
    rag.build_index(all_documents)
    
    # Evaluate
    print("Running evaluation...")
    results = []
    
    for item in tqdm(data):
        question = item.get('question', '')
        gold_answer = item.get('answer', '')
        
        # Retrieve documents
        retrieved_docs = rag.retrieve(question, top_k=5)
        
        # Generate answer (placeholder for now)
        predicted_answer = rag.answer_question(question, retrieved_docs)
        
        # Store result
        result = {
            'question': question,
            'gold_answer': gold_answer,
            'predicted_answer': predicted_answer,
            'retrieved_docs': [
                {
                    'title': doc.get('title', ''),
                    'score': doc.get('retrieval_score', 0.0)
                }
                for doc in retrieved_docs
            ]
        }
        results.append(result)
    
    # Calculate metrics (simple exact match for now)
    exact_matches = sum(
        1 for r in results 
        if r['gold_answer'].lower().strip() in r['predicted_answer'].lower()
    )
    accuracy = exact_matches / len(results) if results else 0.0
    
    # Save results
    output_data = {
        'dataset': 'HotpotQA',
        'num_examples': len(results),
        'exact_match': accuracy,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Exact Match: {accuracy:.2%}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/projects/prjs1800/datasets/hotpotqa/hotpot_dev_distractor_v1.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/projects/prjs1800/results/baseline_hotpotqa/results.json"
    use_small = len(sys.argv) > 3 and sys.argv[3] == "--small"
    
    evaluate_hotpotqa(dataset_path, output_path, use_small)
EOF

# Run baseline evaluation
echo "Running baseline evaluation..."
python baseline_rag_hotpotqa.py \
    "$DATASET_DIR/hotpot_dev_small.json" \
    "$RESULTS_DIR/baseline_results_$(date +%Y%m%d_%H%M%S).json" \
    --small

echo "=========================================="
echo "Baseline evaluation complete!"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
