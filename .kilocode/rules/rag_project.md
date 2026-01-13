# rag_project.md

RAG Technical Documentation System - Federico Lumma

## Guidelines

### 🎯 Architecture
config → document_loader → text_splitter → embeddings → vector_store → rag_pipeline

- Single responsibility per module
- Dependency injection via Config
- LangChain abstractions + custom logic

### 📝 Code Style
- Type hints 100%
- Google docstrings
def func(param: str) -> List[Dict[str, Any]]:
"""Summary line.

Args:
    param: Description
    
Returns:
    List of results
"""

- Imports: LangChain first, stdlib, 3rd party alphabetical
- NO print() → logging.getLogger(__name__)

### 🧪 Testing
tests/test_*.py → pytest -v --cov=src
test_document_loader.py → 95% coverage mínimo
Mock external APIs (Pinecone, OpenAI)


### 🔄 Commits
feat(loader): add PDF/TXT support [95% test cov]
docs(readme): update usage examples
test(splitter): add chunking tests
chore(deps): bump langchain to 0.2.16
fix(embed): handle empty docs edgecase


### 📁 File Structure
src/
├── init.py (version="0.1.0")
├── config.py
├── document_loader.py
├── text_splitter.py
├── embeddings.py
├── vector_store.py
└── rag_pipeline.py

docs/plan.md
tests/test_*.py
.kilocode/rules/rag_project.md


### 🚀 Defaults
CHUNK_SIZE=512
CHUNK_OVERLAP=50
EMBEDDING_MODEL="text-embedding-3-small"
PINECONE_METRIC="cosine"
PINECONE_DIMENSION=1536


### ✅ Success Criteria
- main.py → load PDF → split → embed → upsert → query
- pytest tests/ → 90%+ coverage
- black . && ruff check .
- Fresh clone → pip install → python main.py OK
