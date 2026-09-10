# Day 40 — RAG Feedback Learning

A feedback loop for a Retrieval-Augmented Generation system.

## Features
- Helpful / not-helpful feedback
- Persistent JSON feedback storage
- Quality-score tracking
- Feedback statistics dashboard
- Recent feedback history
- Flask UI and JSON API

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## API
`POST /api/feedback`
```json
{"query":"What is RAG?","answer":"RAG retrieves relevant context before generation.","rating":"up","quality_score":0.9,"sources":["rag_notes.txt"]}
```

`GET /api/stats`

## Day 40 Goal
Add a practical feedback loop so user ratings become measurable quality signals for the RAG system.
