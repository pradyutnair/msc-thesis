"""
Naive Vector Search Baseline for Multi-Hop QA.

This module implements a simple baseline using:
- Sentence-transformers for embeddings
- FAISS for cosine similarity search
- No query decomposition or multi-hop reasoning
"""

from .retriever import NaiveVectorRetriever
from .evaluator import Evaluator
from .dataset_loader import DatasetLoader

__all__ = ['NaiveVectorRetriever', 'Evaluator', 'DatasetLoader']
