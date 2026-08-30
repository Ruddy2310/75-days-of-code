# Day 38 — RAG Evaluation Dashboard

A retrieval evaluation layer for the conversational RAG project.

## Features
- Hit@1, Hit@3 and Mean Reciprocal Rank (MRR)
- Per-question retrieval results
- Browser evaluation dashboard
- Evaluation REST API
- Conversational RAG
- Retrieval caching
- Hybrid semantic + keyword search
- Cross-Encoder reranking
- Optional Ollama generation

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python build_index.py
python app.py
```
Open `http://127.0.0.1:5000`.

Run CLI evaluation:
```bash
python evaluate_retrieval.py
```

API:
- `POST /api/chat`
- `GET /api/evaluate`
- `GET /api/cache`
- `GET /api/health`
