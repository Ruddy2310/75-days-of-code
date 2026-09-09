# Day 39 — RAG Answer Quality Evaluation

Phase 2 Big Project — Incremental Commit #14

This milestone extends the RAG project from retrieval evaluation into **answer-quality evaluation**.

## Features
- Reference-answer based evaluation
- Semantic similarity using Sentence Transformers
- Keyword coverage
- Answer length and basic completeness checks
- Combined quality score
- Per-question evaluation details
- Flask dashboard and JSON API
- Sample evaluation dataset
- Clean modular structure

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`

## Project Structure

```text
day-39-rag-answer-quality-evaluation/
├── app.py
├── evaluator.py
├── requirements.txt
├── COMMIT_MESSAGE.txt
├── data/
│   └── evaluation_dataset.json
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Evaluation

For each question, the system compares a generated answer against a reference answer using:
- semantic similarity
- keyword coverage
- completeness

The final score is a weighted combination of these signals.

> Note: This is an educational evaluation milestone, not a replacement for human evaluation or LLM-as-a-judge systems.

## Day 39 Goal

Build a measurable answer-quality layer on top of the RAG pipeline so future milestones can compare improvements objectively.
