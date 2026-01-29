"""
Naive vector retriever using sentence-transformers and FAISS.

Implements simple cosine similarity search without any multi-hop reasoning.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .dataset_loader import Document

logger = logging.getLogger(__name__)


class NaiveVectorRetriever:
    """
    Naive vector retriever using cosine similarity.
    
    This is a simple baseline that:
    - Encodes documents using sentence-transformers
    - Builds a FAISS index for efficient search
    - Retrieves top-k documents based on cosine similarity
    - Does NOT perform any query decomposition or multi-hop reasoning
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        Initialize the retriever.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use (cuda/cpu)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        logger.info(f"Loading sentence-transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=device)
        
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.dimension: Optional[int] = None
    
    def build_index(self, documents: List[Document], show_progress: bool = True):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of Document objects
            show_progress: Whether to show progress bar
        """
        logger.info(f"Building index from {len(documents)} documents")
        self.documents = documents
        
        # Extract text for encoding
        texts = [self._format_document(doc) for doc in documents]
        
        # Encode documents
        logger.info("Encoding documents...")
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        # Build FAISS index with cosine similarity
        self.dimension = embeddings.shape[1]
        logger.info(f"Building FAISS index (dimension={self.dimension})")
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"Index built successfully with {self.index.ntotal} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples, sorted by score (descending)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            min(top_k, self.index.ntotal)
        )
        
        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        show_progress: bool = True,
    ) -> List[List[Tuple[Document, float]]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query
            show_progress: Whether to show progress bar
            
        Returns:
            List of retrieval results, one per query
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        logger.info(f"Batch retrieving for {len(queries)} queries")
        
        # Encode all queries
        query_embeddings = self.encoder.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embeddings)
        
        # Batch search
        scores_batch, indices_batch = self.index.search(
            query_embeddings.astype(np.float32),
            min(top_k, self.index.ntotal)
        )
        
        # Format results
        all_results = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for idx, score in zip(indices, scores):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            all_results.append(results)
        
        return all_results
    
    def save_index(self, save_dir: Path):
        """
        Save index and documents to disk.
        
        Args:
            save_dir: Directory to save index
        """
        if self.index is None:
            raise RuntimeError("No index to save")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save documents
        docs_path = save_dir / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved documents to {docs_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'num_documents': len(self.documents),
        }
        metadata_path = save_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_index(self, load_dir: Path):
        """
        Load index and documents from disk.
        
        Args:
            load_dir: Directory containing saved index
        """
        load_dir = Path(load_dir)
        
        # Load metadata
        metadata_path = load_dir / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        logger.info(f"Loading index with {metadata['num_documents']} documents")
        
        # Load FAISS index
        index_path = load_dir / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load documents
        docs_path = load_dir / "documents.pkl"
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        logger.info(f"Loaded {len(self.documents)} documents from {docs_path}")
        
        self.dimension = metadata['dimension']
    
    def _format_document(self, doc: Document) -> str:
        """
        Format document for encoding.
        
        Args:
            doc: Document object
            
        Returns:
            Formatted text string
        """
        # Simple format: title + text
        if doc.title:
            return f"{doc.title}. {doc.text}"
        return doc.text
    
    def get_stats(self) -> dict:
        """
        Get retriever statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'num_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0,
            'device': self.device,
        }
