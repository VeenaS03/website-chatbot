"""
Embeddings module for generating and storing embeddings using FAISS.
Handles embedding generation, indexing, and persistence.
"""

import logging
import os
import json
from typing import List, Dict, Tuple
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation and FAISS vector database operations.
    
    Features:
    - Generate embeddings using SentenceTransformers
    - Store embeddings in FAISS index
    - Persist and reload from disk
    - Semantic search capabilities
    - Metadata tracking
    """
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight, efficient
    FAISS_INDEX_FILE = "faiss_index.bin"
    METADATA_FILE = "metadata.json"
    
    def __init__(self, storage_path: str = "data/faiss_index"):
        """
        Initialize embedding manager.
        
        Args:
            storage_path: Directory to store FAISS index and metadata
        """
        self.storage_path = storage_path
        self.model = None
        self.faiss_index = None
        self.metadata = []
        self.dimension = None
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return np.array([])
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            )
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def create_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings
            
        Returns:
            FAISS Index object
        """
        try:
            dimension = embeddings.shape[1]
            logger.info(f"Creating FAISS index with dimension {dimension}")
            
            # Use L2 distance (Euclidean) for similarity search
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            
            logger.info(f"Index created with {index.ntotal} vectors")
            return index
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
            raise
    
    def index_chunks(self, chunks: List[Dict], website_url: str, page_title: str):
        """
        Generate embeddings for chunks and create FAISS index.
        
        Args:
            chunks: List of chunk dictionaries with 'content' key
            website_url: Source website URL
            page_title: Page title for metadata
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return
        
        # Extract text content
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.faiss_index = self.create_index(embeddings)
        
        # Store metadata
        self.metadata = [
            {
                'content': chunk['content'],
                'chunk_index': chunk['chunk_index'],
                'length': chunk['length'],
                'source_url': website_url,
                'page_title': page_title
            }
            for chunk in chunks
        ]
        
        logger.info(f"Indexed {len(self.metadata)} chunks from {website_url}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of similar chunk dictionaries with scores
        """
        if self.faiss_index is None or not self.metadata:
            logger.warning("FAISS index not initialized")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Search in FAISS index
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'),
                min(k, len(self.metadata))
            )
            
            # Retrieve results with scores
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    # Convert distance to similarity score (lower distance = higher similarity)
                    result['similarity_score'] = float(1 / (1 + distance))
                    result['distance'] = float(distance)
                    results.append(result)
            
            logger.info(f"Search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def save_to_disk(self):
        """Save FAISS index and metadata to disk."""
        if self.faiss_index is None:
            logger.warning("No index to save")
            return
        
        try:
            # Save FAISS index
            index_path = os.path.join(self.storage_path, self.FAISS_INDEX_FILE)
            faiss.write_index(self.faiss_index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
            
            # Save metadata
            metadata_path = os.path.join(self.storage_path, self.METADATA_FILE)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_from_disk(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if load was successful, False otherwise
        """
        try:
            index_path = os.path.join(self.storage_path, self.FAISS_INDEX_FILE)
            metadata_path = os.path.join(self.storage_path, self.METADATA_FILE)
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.info("No saved index found")
                return False
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} metadata entries")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def clear_index(self):
        """Clear the current index and metadata."""
        self.faiss_index = None
        self.metadata = []
        logger.info("Index cleared")
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.faiss_index is None:
            return {'initialized': False, 'vector_count': 0}
        
        return {
            'initialized': True,
            'vector_count': self.faiss_index.ntotal,
            'dimension': self.dimension,
            'metadata_count': len(self.metadata),
            'storage_path': self.storage_path
        }


def create_embedding_manager(storage_path: str = "data/faiss_index") -> EmbeddingManager:
    """Factory function to create an EmbeddingManager instance."""
    return EmbeddingManager(storage_path)