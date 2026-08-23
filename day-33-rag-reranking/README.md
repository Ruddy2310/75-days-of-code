# Day 33 — Retrieval-Augmented Generation with Cross-Encoder Reranking

Day 33 adds a second-stage Cross-Encoder reranker to the Day 32 hybrid RAG pipeline.

Pipeline: Question → Hybrid Semantic + Keyword Retrieval → Candidate Chunks → Cross-Encoder Reranking → RAG Answer.

## Run
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python build_index.py
python app.py
```
Open http://127.0.0.1:5000

Optional local LLM: `ollama pull llama3.2:3b`

Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
