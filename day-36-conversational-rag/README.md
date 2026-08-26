# Day 36 — Conversational Retrieval-Augmented Generation

Day 36 upgrades the RAG assistant into a conversational RAG system. It keeps recent chat history, rewrites follow-up questions into retrieval-friendly queries, retrieves evidence, reranks it with a Cross-Encoder, and optionally generates the final answer with Ollama.

## Pipeline
```text
Conversation History → Follow-up Query Rewriting → Hybrid Retrieval → Cross-Encoder Reranking → Grounded Answer
```

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python build_index.py
python app.py
```
Open `http://127.0.0.1:5000`.

Optional local generation: `ollama pull llama3.2:3b`

## API
`POST /api/chat`
```json
{"message":"Why is it useful?","history":[{"role":"user","content":"What is RAG?"}],"use_llm":false}
```
`GET /api/health` returns service status.
