# Day 32 — Hybrid Retrieval-Augmented Generation Search

Day 32 extends the RAG application with hybrid retrieval: semantic vector search plus TF-IDF keyword search.

## Features
- Flask RAG web interface
- Multi-document TXT/Markdown knowledge base
- Sentence Transformer semantic retrieval
- TF-IDF lexical retrieval
- Weighted hybrid ranking
- Source attribution and retrieval scores
- Conversation history
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

Hybrid score:
`0.65 × semantic score + 0.35 × keyword score`
