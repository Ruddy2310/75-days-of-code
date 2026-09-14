# Day 44 — RAG Query Explanation

Explainable query routing for the RAG pipeline.

## Features
- Query intent detection
- Factual, technical, comparison and general routes
- Retrieval strategy selection
- Confidence score
- Matched keywords
- Decision trace
- Flask dashboard and JSON API

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## API
POST `/api/explain`

Example:
```json
{"query":"Compare vector search and keyword search"}
```

## Day 44 Goal
Make RAG routing decisions transparent by explaining why a query was assigned to a particular retrieval strategy.
