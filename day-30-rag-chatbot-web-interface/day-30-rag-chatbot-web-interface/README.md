# Day 30 — Retrieval-Augmented Generation Web Interface

A browser-based upgrade to the RAG chatbot. This project uses Flask, Sentence Transformers, and FAISS for semantic retrieval, with optional local Ollama generation.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python build_index.py
python app.py
```

Open `http://127.0.0.1:5000`.

Optional:
```bash
ollama pull llama3.2:3b
```

## Features
- Flask web interface
- TXT/Markdown knowledge base
- Sentence Transformer embeddings
- FAISS semantic search
- Top-K retrieval
- Source attribution
- Optional Ollama generation

## Structure

```text
day-30-rag-chatbot-web-interface/
├── app.py
├── build_index.py
├── rag_engine.py
├── requirements.txt
├── data/documents/
├── index/
├── templates/index.html
├── static/style.css
└── README.md
```
