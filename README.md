# Technical Documentation RAG System

## 🎯 Overview
Retrieval-Augmented Generation system para buscar y responder preguntas sobre documentación técnica usando LLMs.

## 🤔 Problem Statement
- Documentación técnica dispersa en múltiples fuentes
- Búsquedas lentas y poco precisas
- Ingenieros pierden horas buscando respuestas

## 💡 Solution
1. Indexar docs con embeddings vectoriales (Pinecone)
2. Query natural language → retrieval → LLM response
3. API REST para integración fácil

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4 Turbo |
| Vector DB | Pinecone |
| Framework | LangChain + FastAPI |
| Language | Python 3.11+ |
| Tests | pytest |

## 🚀 Quick Start
```bash
git clone https://github.com/fexdelu/technical-doc-rag.git
cd technical-doc-rag
pip install -r requirements.txt
cp .env.example .env  # Edit con tus API keys
python src/rag_pipeline.py
```

## 📁 Project Structure
```text
technical-doc-rag/
├── src/
│   ├── config.py
│   ├── document_loader.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── tests/
├── documents/     # Input docs here
├── output/        # Generated results
├── requirements.txt
└── README.md
```

## 🔄 Usage
```python
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()
response = rag.query("¿Cómo configuro Pinecone?")
print(response)
```

## 👨‍💻 Author
Federico Lumma - AI Engineer | Python | LLMs
Buenos Aires, Argentina

## 📄 License
MIT
