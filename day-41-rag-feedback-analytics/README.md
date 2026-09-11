# Day 41 — RAG Feedback Analytics

Analytics dashboard for the RAG feedback loop from Day 40.

## Features
- Helpful vs unhelpful feedback rate
- Average answer quality score
- Daily feedback trend
- Low-rated query detection
- Recent feedback history
- Flask JSON APIs
- Local JSON persistence

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## APIs
- GET /api/analytics
- GET /api/feedback
- POST /api/feedback

## Goal
Turn raw user feedback into actionable signals for improving a RAG system.
