# Day 29 — Multi-Document Retrieval-Augmented Generation

A Day 29 upgrade to the RAG chatbot: search across multiple TXT/Markdown documents with Sentence Transformer embeddings and FAISS, preserve source metadata, and optionally generate answers with a local Ollama model.

## Features
- Multi-document TXT/Markdown ingestion
- Chunking with overlap
- Sentence Transformer embeddings
- FAISS semantic search
- Top-K retrieval and source attribution
- Optional local Ollama generation
- Command-line chatbot

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python build_index.py
python chat.py
```

Optional Ollama:
```bash
ollama pull llama3.2:3b
python chat.py --llm
```

## Structure
```text
day-29-multi-document-rag/
├── data/documents/
├── index/
├── src/
│   ├── document_loader.py
│   ├── chunker.py
│   ├── vector_store.py
│   └── rag_pipeline.py
├── build_index.py
├── chat.py
├── requirements.txt
└── README.md
```

## Learning
Multi-document retrieval, semantic search, vector indexes, metadata, source attribution, and Retrieval-Augmented Generation.
