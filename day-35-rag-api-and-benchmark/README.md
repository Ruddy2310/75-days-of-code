# Day 35 — Retrieval-Augmented Generation API and Benchmark

Day 35 turns the document RAG application into a reusable API and adds a
small retrieval benchmark.

## Features
- REST API for RAG questions
- Browser dashboard
- Hybrid semantic + keyword retrieval
- Cross-Encoder reranking
- Source attribution and retrieval scores
- Conversation history
- Retrieval benchmark with Hit@1, Hit@3 and MRR
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

## API
POST `/api/ask`
```json
{"question":"What is retrieval augmented generation?","use_llm":false}
```

GET `/api/documents` returns indexed documents.

## Benchmark
```bash
python evaluate_retrieval.py
```

Metrics:
- Hit@1: expected source ranked first
- Hit@3: expected source appears in top three
- MRR: reciprocal rank of the first correct source
