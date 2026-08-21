# Day 31 — RAG Chat History and Evaluation

An incremental upgrade to the Day 30 RAG web application.

## Features
- Flask RAG web app
- Multi-document TXT/Markdown knowledge base
- Sentence Transformer embeddings
- FAISS semantic retrieval
- Session-based conversation history
- Source attribution and retrieval scores
- Retrieval evaluation with expected-source hit rate
- Optional local Ollama generation

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

Evaluate retrieval:
```bash
python evaluate.py
```
