"""
Main script for running naive vector search baseline.

Usage:
    python run_baseline.py --config configs/hotpotqa.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Set CUDA device if not already set
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import yaml
from tqdm import tqdm

# Support both package (python -m) and script invocation
try:
    from .dataset_loader import DatasetLoader
    from .retriever import NaiveVectorRetriever
    from .evaluator import Evaluator
except ImportError:
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from baselines.naive_vector_search.dataset_loader import DatasetLoader
    from baselines.naive_vector_search.retriever import NaiveVectorRetriever
    from baselines.naive_vector_search.evaluator import Evaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: Dict[str, Any], output_path: Path):
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def run_baseline(config: Dict[str, Any]):
    """
    Run naive vector search baseline.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 80)
    logger.info("Starting Naive Vector Search Baseline")
    logger.info("=" * 80)
    
    # Extract config
    dataset_name = config['dataset']['name']
    dataset_dir = Path(config['dataset']['dir'])
    split = config['dataset'].get('split', 'dev')
    limit = config['dataset'].get('limit', None)
    
    model_name = config['retriever']['model_name']
    device = config['retriever'].get('device', 'cuda')
    top_k = config['retriever'].get('top_k', 5)
    batch_size = config['retriever'].get('batch_size', 32)
    
    output_dir = Path(config['output']['dir'])
    save_index = config['output'].get('save_index', False)
    index_dir = Path(config['output'].get('index_dir', output_dir / 'index'))
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Split: {split}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Top-k: {top_k}")
    logger.info(f"Device: {device}")
    
    # Load dataset
    logger.info("-" * 80)
    logger.info("Loading dataset...")
    loader = DatasetLoader(dataset_name, dataset_dir)
    
    questions = loader.load_questions(split=split, limit=limit)
    logger.info(f"Loaded {len(questions)} questions")
    
    corpus = loader.load_corpus()
    logger.info(f"Loaded {len(corpus)} documents")
    
    # Initialize retriever
    logger.info("-" * 80)
    logger.info("Initializing retriever...")
    retriever = NaiveVectorRetriever(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    
    # Build or load index
    if index_dir.exists() and (index_dir / 'faiss.index').exists():
        logger.info(f"Loading existing index from {index_dir}")
        retriever.load_index(index_dir)
    else:
        logger.info("Building index from corpus...")
        start_time = time.time()
        retriever.build_index(corpus, show_progress=True)
        build_time = time.time() - start_time
        logger.info(f"Index built in {build_time:.2f} seconds")
        
        if save_index:
            logger.info(f"Saving index to {index_dir}")
            retriever.save_index(index_dir)
    
    # Print retriever stats
    stats = retriever.get_stats()
    logger.info(f"Retriever stats: {stats}")
    
    # Run retrieval
    logger.info("-" * 80)
    logger.info("Running retrieval...")
    
    all_results = []
    queries = [q.question for q in questions]
    
    start_time = time.time()
    retrieved_docs_batch = retriever.batch_retrieve(
        queries=queries,
        top_k=top_k,
        show_progress=True,
    )
    retrieval_time = time.time() - start_time
    
    logger.info(f"Retrieval completed in {retrieval_time:.2f} seconds")
    logger.info(f"Average time per query: {retrieval_time / len(queries):.4f} seconds")
    
    # Process results
    logger.info("-" * 80)
    logger.info("Processing results...")
    
    predictions = []
    ground_truths = []
    retrieved_titles_list = []
    supporting_facts_list = []
    
    for question, retrieved_docs in tqdm(zip(questions, retrieved_docs_batch), total=len(questions)):
        # Extract retrieved titles
        retrieved_titles = [doc.title for doc, _ in retrieved_docs]
        
        # For baseline, we use a simple heuristic:
        # Return the answer if it appears in retrieved documents
        # Otherwise, return empty string
        # In a real system, this would use an LLM to generate the answer
        
        # Simple answer extraction (baseline heuristic)
        predicted_answer = ""
        for doc, score in retrieved_docs:
            # Check if answer appears in document
            if question.answer.lower() in doc.text.lower():
                predicted_answer = question.answer
                break
        
        # If no match found, use a placeholder
        if not predicted_answer:
            predicted_answer = "no_answer"
        
        predictions.append(predicted_answer)
        ground_truths.append(question.answer)
        retrieved_titles_list.append(retrieved_titles)
        
        if question.supporting_facts:
            supporting_facts_list.append(question.supporting_facts)
        else:
            supporting_facts_list.append([])
        
        # Store detailed result
        result = {
            'question_id': question.id,
            'question': question.question,
            'predicted_answer': predicted_answer,
            'ground_truth': question.answer,
            'retrieved_docs': [
                {
                    'title': doc.title,
                    'score': score,
                    'text': doc.text[:200] + '...' if len(doc.text) > 200 else doc.text,
                }
                for doc, score in retrieved_docs
            ],
        }
        all_results.append(result)
    
    # Compute metrics
    logger.info("-" * 80)
    logger.info("Computing metrics...")
    
    metrics = Evaluator.compute_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        retrieved_docs_list=retrieved_titles_list,
        supporting_facts_list=supporting_facts_list,
    )
    
    Evaluator.print_metrics(metrics)
    
    # Save results
    logger.info("-" * 80)
    output_data = {
        'config': config,
        'metrics': metrics,
        'stats': {
            'num_questions': len(questions),
            'num_documents': len(corpus),
            'retrieval_time': retrieval_time,
            'avg_time_per_query': retrieval_time / len(queries),
        },
        'results': all_results,
    }
    
    output_path = output_dir / f"{dataset_name}_{split}_results.json"
    save_results(output_data, output_path)
    
    logger.info("=" * 80)
    logger.info("Baseline evaluation complete!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run naive vector search baseline for multi-hop QA"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    
    # Run baseline
    try:
        run_baseline(config)
    except Exception as e:
        logger.error(f"Error running baseline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
