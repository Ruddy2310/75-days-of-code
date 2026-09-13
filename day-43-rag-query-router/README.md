# Day 43 — RAG Query Router

A lightweight query-routing layer for a Retrieval-Augmented Generation pipeline.

## Features
- Query intent classification
- Factual, technical, comparison, and general routes
- Retrieval strategy selection
- Confidence score and route explanation
- JSON API and browser dashboard
- No external model/API required

## Run
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## API
POST `/api/route` with `{"query":"Compare vector search and keyword search"}`

## Day 43 Goal
Route different question types to suitable retrieval strategies before retrieval begins.
