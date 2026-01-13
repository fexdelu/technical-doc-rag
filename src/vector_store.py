"""
Pinecone vector store integration.
"""

from typing import List, Any
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from src.config import Config
from src.embeddings import EmbeddingsGenerator
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Pinecone integration para RAG."""
    
    def __init__(self, config: Config):
        """Initialize Pinecone vector store.
        
        Args:
            config: Configuration object with API keys and index settings
        """
        self.config = config
        self.embeddings = EmbeddingsGenerator(config)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = self._get_or_create_index()
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings.embeddings,
        )
    
    def _get_or_create_index(self) -> Any:
        """Create index si no existe.
        
        Returns:
            Pinecone index object
        """
        if self.config.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.config.PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-west-2"
                    }
                }
            )
            logger.info(f"✅ Created Pinecone index: {self.config.PINECONE_INDEX_NAME}")
        return self.pc.Index(self.config.PINECONE_INDEX_NAME)
    
    def upsert(self, chunks: List[Document]) -> None:
        """Upsert chunks to Pinecone.
        
        Args:
            chunks: List of document chunks to upsert
        """
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        self.vectorstore.add_documents(chunks)
        logger.info(f"✅ Upserted {len(chunks)} chunks")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Semantic search.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of most similar documents
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return results


if __name__ == "__main__":
    # Test main
    import os
    from dotenv import load_dotenv
    from src.document_loader import DocumentLoader
    from src.text_splitter import TextSplitter
    
    load_dotenv()
    
    # Requires OPENAI_API_KEY and PINECONE_API_KEY in environment
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY or PINECONE_API_KEY not set - skipping test")
    else:
        config = Config()
        
        # Load and split documents
        loader = DocumentLoader(config)
        splitter = TextSplitter(config)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        
        # Initialize vector store and upsert
        vector_store = VectorStore(config)
        vector_store.upsert(chunks[:5])  # Test with first 5 chunks
        
        # Test similarity search
        results = vector_store.similarity_search("test query", k=3)
        logger.info(f"Search results: {len(results)} documents found")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")