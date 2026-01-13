"""
Text chunking para RAG. Chunks de 512 tokens con overlap.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import Config
import logging

logger = logging.getLogger(__name__)


class TextSplitter:
    """Smart text splitter for RAG documents."""
    
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50
    
    def __init__(self, config: Config):
        """Initialize text splitter with configuration.
        
        Args:
            config: Project config
        """
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE or self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP or self.DEFAULT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,  # token-aware más adelante
        )
        logger.info("Splitter: %d tokens, overlap %d", self.splitter._chunk_size, self.splitter._chunk_overlap)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks.
        
        Args:
            documents: Raw documents from loader
            
        Returns:
            Chunked documents ready for embedding
        """
        chunks = self.splitter.split_documents(documents)
        logger.info("✅ Split %d docs → %d chunks", len(documents), len(chunks))
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """Stats sobre chunks generados.
        
        Args:
            chunks: List of chunked documents
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
            }
        
        lengths = [len(chunk.page_content) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }


if __name__ == "__main__":
    from src.config import Config
    from src.document_loader import DocumentLoader
    
    config = Config()
    loader = DocumentLoader(config)
    splitter = TextSplitter(config)
    
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    print(splitter.get_chunk_stats(chunks))