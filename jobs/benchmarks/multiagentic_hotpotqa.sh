#!/bin/bash
#SBATCH --job-name=multiagentic_hotpotqa
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=jobs/output/multiagentic_hotpotqa_%j.log
#SBATCH --error=jobs/output/multiagentic_hotpotqa_%j.log

# Multi-Agentic RAG Benchmark on HotpotQA
# This script runs the multi-agentic RAG framework on HotpotQA

echo "=========================================="
echo "Starting Multi-Agentic HotpotQA Benchmark"
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

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    echo "Please set it before running this job:"
    echo "export OPENAI_API_KEY='your-api-key'"
    exit 1
fi

# Set paths
DATASET_DIR="/projects/prjs1800/datasets/hotpotqa"
RESULTS_DIR="/projects/prjs1800/results/multiagentic_hotpotqa"
mkdir -p $RESULTS_DIR

# Navigate to project directory
cd $HOME/msc-thesis

echo "=========================================="
echo "Configuration"
echo "=========================================="
echo "Dataset: HotpotQA"
echo "Dataset directory: $DATASET_DIR"
echo "Results directory: $RESULTS_DIR"
echo "Framework: Multi-Agentic RAG"
echo "Orchestration: Hierarchical"
echo "LLM: gpt-4-turbo"
echo "=========================================="

# Create multi-agentic evaluation script
cat > multiagentic_rag_hotpotqa.py << 'EOF'
"""
Multi-Agentic RAG System for HotpotQA Evaluation
This implements the multi-agentic RAG framework for evaluation.
"""

import json
import os
import sys
from typing import List, Dict, Any
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# Add the multi_agentic_rag module to path
sys.path.insert(0, os.path.expanduser('~/msc-thesis'))

from multi_agentic_rag.agents.base_agent import BaseAgent
from multi_agentic_rag.agents.decomposition_agent import DecompositionAgent
from multi_agentic_rag.agents.retrieval_agent import RetrievalAgent
from multi_agentic_rag.agents.synthesis_agent import SynthesisAgent
from multi_agentic_rag.orchestrator import MultiAgenticOrchestrator, OrchestrationStrategy
from multi_agentic_rag.memory import AgenticMemory

# Simple LLM wrapper for OpenAI
class OpenAILLM:
    def __init__(self, model: str = "gpt-4-turbo"):
        self.client = OpenAI()
        self.model = model
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "Error generating response"

# Simple retrieval method
class VectorRetrieval:
    def __init__(self, documents: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        self.name = "vector_retrieval"
        self.encoder = SentenceTransformer(model_name)
        self.documents = documents
        self.index = None
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index."""
        print("Building vector index...")
        texts = []
        for doc in self.documents:
            title = doc.get('title', '')
            sentences = doc.get('sentences', [])
            text = f"{title}. {' '.join(sentences)}"
            texts.append(text)
        
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print(f"Index built with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5, filters: Dict = None) -> List[Dict[str, Any]]:
        """Retrieve top-k documents."""
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                doc['content'] = f"{doc.get('title', '')}. {' '.join(doc.get('sentences', []))}"
                doc['source'] = doc.get('title', 'Unknown')
                results.append(doc)
        
        return results

def evaluate_multiagentic_hotpotqa(dataset_path: str, output_path: str, use_small: bool = False):
    """Evaluate multi-agentic RAG on HotpotQA."""
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    if use_small:
        data = data[:10]  # Use even smaller set for testing
    
    print(f"Loaded {len(data)} examples")
    
    # Extract documents
    print("Extracting documents...")
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
    
    # Initialize components
    print("Initializing multi-agentic RAG system...")
    llm = OpenAILLM(model="gpt-3.5-turbo")  # Use cheaper model for testing
    memory = AgenticMemory()
    
    # Create retrieval method
    vector_retrieval = VectorRetrieval(all_documents)
    
    # Create agents
    decomposition_agent = DecompositionAgent(
        name="decomposer",
        llm=llm,
        memory=memory
    )
    
    retrieval_agent = RetrievalAgent(
        name="retriever",
        llm=llm,
        retrieval_methods=[vector_retrieval],
        memory=memory
    )
    
    synthesis_agent = SynthesisAgent(
        name="synthesizer",
        llm=llm,
        memory=memory
    )
    
    # Create orchestrator
    orchestrator = MultiAgenticOrchestrator(
        decomposition_agent=decomposition_agent,
        retrieval_agents=[retrieval_agent],
        synthesis_agent=synthesis_agent,
        strategy=OrchestrationStrategy.HIERARCHICAL
    )
    
    print("Multi-agentic RAG system initialized!")
    
    # Evaluate
    print("Running evaluation...")
    results = []
    
    for item in tqdm(data):
        question = item.get('question', '')
        gold_answer = item.get('answer', '')
        
        try:
            # Process query through multi-agent system
            result = orchestrator.process_query(question)
            
            predicted_answer = result.get('answer', '')
            confidence = result.get('confidence', 0.0)
            sources = result.get('sources', [])
            
            # Store result
            eval_result = {
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': predicted_answer,
                'confidence': confidence,
                'sources': sources,
                'metadata': result.get('metadata', {})
            }
            results.append(eval_result)
            
        except Exception as e:
            print(f"Error processing question: {e}")
            results.append({
                'question': question,
                'gold_answer': gold_answer,
                'predicted_answer': 'ERROR',
                'error': str(e)
            })
    
    # Calculate metrics
    exact_matches = sum(
        1 for r in results 
        if r.get('gold_answer', '').lower().strip() in r.get('predicted_answer', '').lower()
    )
    accuracy = exact_matches / len(results) if results else 0.0
    
    avg_confidence = np.mean([r.get('confidence', 0.0) for r in results])
    
    # Save results
    output_data = {
        'dataset': 'HotpotQA',
        'system': 'Multi-Agentic RAG',
        'orchestration': 'Hierarchical',
        'num_examples': len(results),
        'exact_match': accuracy,
        'avg_confidence': float(avg_confidence),
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Exact Match: {accuracy:.2%}")
    print(f"Average Confidence: {avg_confidence:.2f}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "/projects/prjs1800/datasets/hotpotqa/hotpot_dev_tiny.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "/projects/prjs1800/results/multiagentic_hotpotqa/results.json"
    use_small = len(sys.argv) > 3 and sys.argv[3] == "--small"
    
    evaluate_multiagentic_hotpotqa(dataset_path, output_path, use_small)
EOF

# Run multi-agentic evaluation
echo "Running multi-agentic evaluation..."
python multiagentic_rag_hotpotqa.py \
    "$DATASET_DIR/hotpot_dev_tiny.json" \
    "$RESULTS_DIR/multiagentic_results_$(date +%Y%m%d_%H%M%S).json" \
    --small

echo "=========================================="
echo "Multi-agentic evaluation complete!"
echo "Results directory: $RESULTS_DIR"
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
