"""
Text processing module for cleaning and chunking content.
Handles text normalization, deduplication, and semantic chunking.
"""

import logging
import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Responsible for cleaning, normalizing, and chunking text content.
    
    Features:
    - Text normalization and cleaning
    - Duplicate removal
    - Semantic chunking with configurable size and overlap
    - Metadata preservation
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize text processor with chunking configuration.
        
        Args:
            chunk_size: Size of each text chunk (characters)
            chunk_overlap: Overlap between consecutive chunks (characters)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Remove consecutive punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€\x9d', '"')
        
        # Remove standalone numbers and special markers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    @staticmethod
    def remove_duplicates(chunks: List[str]) -> List[str]:
        """
        Remove duplicate chunks while preserving order.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of unique chunks
        """
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            # Use normalized text for comparison
            normalized = chunk.lower().strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into semantic chunks with metadata.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of dictionaries with 'content' and metadata
        """
        # Clean text first
        cleaned_text = self.clean_text(text)
        
        if not cleaned_text:
            logger.warning("Text is empty after cleaning")
            return []
        
        # Split into chunks
        chunks = self.splitter.split_text(cleaned_text)
        
        # Remove duplicates
        chunks = self.remove_duplicates(chunks)
        
        # Create chunk objects with metadata
        chunk_objects = [
            {
                'content': chunk,
                'chunk_index': idx,
                'length': len(chunk)
            }
            for idx, chunk in enumerate(chunks)
        ]
        
        logger.info(f"Created {len(chunk_objects)} chunks from text")
        return chunk_objects
    
    def update_chunk_config(self, chunk_size: int, chunk_overlap: int):
        """
        Update chunking configuration dynamically.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        if chunk_overlap >= chunk_size:
            logger.warning(f"Overlap {chunk_overlap} >= size {chunk_size}, adjusting")
            chunk_overlap = chunk_size // 4
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"Updated chunk config: size={chunk_size}, overlap={chunk_overlap}")


def create_text_processor(chunk_size: int = 500, chunk_overlap: int = 100) -> TextProcessor:
    """Factory function to create a TextProcessor instance."""
    return TextProcessor(chunk_size, chunk_overlap)