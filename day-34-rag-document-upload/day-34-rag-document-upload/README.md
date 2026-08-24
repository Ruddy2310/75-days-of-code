# Day 34 — Retrieval-Augmented Generation Document Upload

Day 34 upgrades the RAG assistant into a practical knowledge-base app.
Upload TXT/Markdown files in the browser, rebuild the index automatically,
and ask questions against the uploaded documents.

## Features
- Browser document upload
- TXT and Markdown support
- Secure filenames
- 2 MB upload limit
- Automatic indexing
- Sentence Transformer semantic retrieval
- TF-IDF keyword retrieval
- Hybrid ranking
- Cross-Encoder reranking
- Source attribution
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

Pipeline:
Upload → Index → Hybrid Retrieval → Cross-Encoder Reranking → RAG Answer
