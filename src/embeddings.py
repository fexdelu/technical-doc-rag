"""
OpenAI embeddings para RAG chunks.
"""

from typing import List, Any
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from src.config import Config
import logging

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """Generate embeddings con OpenAI."""
    
    def __init__(self, config: Config):
        """Initialize OpenAI embeddings.
        
        Args:
            config: Configuration object with API keys and model settings
        """
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.OPENAI_MODEL or "text-embedding-3-small",
            api_key=config.OPENAI_API_KEY,
            dimensions=1536,  # Default para embedding-3-small
        )
        logger.info(f"Embeddings: {self.embeddings.model}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.
        
        Args:
            texts: Raw chunk texts
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embeddings.embed_documents(texts)
        logger.info(f"✅ Embedded {len(embeddings)} texts")
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Embed single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector for the query
        """
        embedding = self.embeddings.embed_query(query)
        return embedding


if __name__ == "__main__":
    # Test main
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Requires OPENAI_API_KEY in environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY not set - skipping test")
    else:
        config = Config()
        embedder = EmbeddingsGenerator(config)
        
        # Test embedding documents
        test_texts = ["Hello world", "RAG system test"]
        embeddings = embedder.embed_documents(test_texts)
        logger.info(f"Document embeddings shape: {len(embeddings)}x{len(embeddings[0])}")
        
        # Test embedding query
        query_embedding = embedder.embed_query("test query")
        logger.info(f"Query embedding shape: {len(query_embedding)}")