# Week 2: Document Processing Pipeline
**Tuesday Jan 21, 2026** - 6 horas

## 🎯 Objectives
1. Load PDFs/TXT desde ./documents/
2. Extraer texto limpio
3. Chunking inteligente (512 tokens)
4. Primera conexión Pinecone
5. Test end-to-end

## 📋 Tasks (6 commits)

### Commit #1: DocumentLoader class
src/document_loader.py

PDF/TXT loader con LangChain

metadata (filename, page)

test con sample PDF

### Commit #2: Text Splitter
src/text_splitter.py

RecursiveCharacterTextSplitter

512 tokens/chunk

overlap 50 tokens

### Commit #3: Embeddings + Pinecone
src/embeddings.py
src/vector_store.py

OpenAI embeddings

Pinecone upsert

### Commit #4: RAG Pipeline skeleton
src/rag_pipeline.py

load → split → embed → store

query skeleton

### Commit #5: main.py + tests
main.py
tests/test_loader.py

end-to-end test

pytest suite

### Commit #6: README + docs update
README.md
docs/plan.md

Usage examples

API docs

## 📁 Sample Data
documents/
├── sample-manual.pdf
├── api-docs.txt
└── quickstart.md

## ✅ Success Criteria
- [ ] main.py carga PDF → 10 chunks → Pinecone
- [ ] Query funciona (aunque básica)
- [ ] 100% test coverage loader
- [ ] README con screenshots