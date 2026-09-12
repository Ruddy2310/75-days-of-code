# Day 42 — RAG Feedback Insights

Actionable insights dashboard for RAG feedback.

## Features
- Helpful vs unhelpful rate
- Average quality score
- Quality buckets
- Weak-query detection
- Weak-topic frequency
- Improvement recommendations
- Flask JSON API
- Local JSON persistence

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## APIs
- GET /api/insights
- GET /api/feedback

## Goal
Move from measuring feedback to identifying patterns in weak answers and generating practical improvement suggestions.
