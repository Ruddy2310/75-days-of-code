# Day 37 — Streaming Retrieval-Augmented Generation with Caching

Day 37 makes the conversational RAG assistant faster and more production-ready.

## Features
- Retrieval-result caching
- Query normalization
- Server-Sent Events streaming API
- Conversational query rewriting
- Hybrid semantic + keyword retrieval
- Cross-Encoder reranking
- Source attribution
- Cache statistics
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

Optional:
```bash
ollama pull llama3.2:3b
```

## API
`POST /api/chat`
```json
{"message":"What is RAG?","history":[],"use_llm":false}
```

`POST /api/chat/stream` returns Server-Sent Events.

`GET /api/cache` returns cache statistics.

`GET /api/health` returns service health.

## Pipeline
```text
Question → Normalize → Cache Lookup
                         ↓ miss
                 Hybrid Retrieval
                         ↓
                Cross-Encoder Rerank
                         ↓
                       Cache
                         ↓
                    RAG Answer
                         ↓
                      Stream
```
