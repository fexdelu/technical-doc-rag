#!/usr/bin/env python
"""
End-to-end test: load → split → stats
"""

from src.config import Config
from src.document_loader import DocumentLoader
from src.text_splitter import TextSplitter


def main() -> None:
    """Run end-to-end pipeline test."""
    config = Config()
    
    # Load
    loader = DocumentLoader(config)
    docs = loader.load()
    
    # Split  
    splitter = TextSplitter(config)
    chunks = splitter.split_documents(docs)
    
    # Stats
    stats = splitter.get_chunk_stats(chunks)
    print("PIPELINE STATS:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print(f"\n{len(chunks)} chunks ready for embedding")


if __name__ == "__main__":
    main()