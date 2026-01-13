"""
Document loading module for RAG system.
Supports PDF, TXT, MD with metadata.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.config import Config

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load documents from configured path with metadata."""

    SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]

    def __init__(self, config: Config):
        """Initialize the document loader.

        Args:
            config: Configuration object containing DOCS_PATH.
        """
        self.config = config
        self.docs_path = Path(config.DOCS_PATH)
        self.docs_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loader init: {self.docs_path}")

    def load(self, extensions: Optional[List[str]] = None) -> List[Document]:
        """Load documents from the documents directory.

        Args:
            extensions: List of file extensions to load.
                Defaults to SUPPORTED_EXTENSIONS if None.

        Returns:
            List of loaded documents with metadata.
        """
        if extensions is None:
            extensions = self.SUPPORTED_EXTENSIONS

        docs: List[Document] = []

        for ext in sorted(extensions):
            try:
                loader_cls = self._get_loader_class(ext)
            except ValueError:
                logger.warning(f"Extension {ext} not supported, skipping.")
                continue

            files = list(self.docs_path.glob(f"*{ext}"))
            if not files:
                continue

            count = 0
            for file_path in files:
                try:
                    loader = loader_cls(str(file_path))
                    file_docs = loader.load()

                    for doc in file_docs:
                        doc.metadata["source_file"] = file_path.name
                        doc.metadata["doc_type"] = ext.lstrip(".")

                    docs.extend(file_docs)
                    count += 1
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

            logger.info(f"Loaded {count} {ext} files")

        return docs

    def _get_loader_class(self, ext: str) -> type:
        """Get the appropriate loader class for the extension.

        Args:
            ext: File extension (e.g. '.pdf').

        Returns:
            The LangChain loader class.

        Raises:
            ValueError: If the extension is not supported.
        """
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported extension: {ext}")
        return loaders[ext]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = Config()
    loader = DocumentLoader(config)
    docs = loader.load()
    print(f"Loaded {len(docs)} docs")
