# 📚 StudyMate

A RAG (Retrieval-Augmented Generation) chatbot that answers questions
grounded in your own study material — PDFs, notes, slides — instead of
generic web knowledge. Ask it something, and it retrieves the actual
relevant passages from your documents before answering, with citations
back to the source.

Part of Phase 2 ("Big Projects") of my
[#75DaysOfCode](https://github.com/Ruddy2310/75-days-of-code) challenge,
built incrementally across Days 26-32+.

## Why this project

I have a pile of course PDFs (Software Engineering, Computer Networks,
DAA, etc.) that I currently search through manually when studying.
StudyMate is meant to actually be useful day-to-day: point it at a
folder of my notes, then ask "what's the difference between a Git
merge and rebase?" and get an answer sourced from my own material.

## How RAG works (the short version)

1. **Load & chunk** documents into small, focused pieces of text.
2. **Embed** each chunk into a vector (a list of numbers capturing its meaning).
3. Store those vectors in a **vector database** for fast similarity search.
4. When a question comes in, embed the question too, and retrieve the
   most similar chunks.
5. Feed those chunks + the question to an LLM, which answers using
   only that retrieved context (instead of guessing from general
   training knowledge).

This grounds answers in your actual documents and lets you trace every
answer back to a source, instead of trusting an LLM's memory.

## Build roadmap (incremental commits)

- [x] **Day 26**: Document loading (PDF/TXT/MD) + text chunking pipeline
- [ ] **Day 27**: Embeddings + vector store (sentence-transformers + FAISS)
- [ ] **Day 28**: Retrieval + LLM-based question answering
- [ ] **Day 29**: Streamlit chat UI
- [ ] **Day 30**: Source citations + chat history
- [ ] **Day 31**: Evaluation (does retrieval actually find the right chunks?)
- [ ] **Day 32**: Polish, error handling, deployment prep

## Current status: Day 26 - Document Loading & Chunking

### What's here

- **`document_loader.py`** — loads `.pdf`, `.txt`, and `.md` files from a
  folder, extracting clean text. Skips unsupported/unreadable files
  gracefully instead of crashing the whole batch.
- **`chunker.py`** — splits loaded documents into overlapping chunks
  (default: 800 characters, 150 character overlap), trying to break on
  sentence/paragraph boundaries rather than cutting mid-word. Overlap
  prevents answers from losing context right at a chunk boundary.

### Try it yourself

```bash
pip install -r requirements.txt

# Drop a few PDFs/notes into sample_docs/, then:
python document_loader.py ./sample_docs
python chunker.py ./sample_docs
```

### Design decisions

- **Character-based chunking, not token-based (for now)**: simpler to
  reason about and doesn't require loading a tokenizer yet. This may
  get upgraded to token-aware chunking once the embedding model is
  wired in (Day 27), since embedding models have token limits, not
  character limits.
- **Boundary-aware splitting**: chunks try to end on `\n\n`, `. `, or
  `\n` rather than an arbitrary character count, so retrieval doesn't
  return a chunk that starts or ends mid-sentence.
- **Metadata carried through**: every chunk remembers its source
  filename and character range, which is what makes citations possible
  later (Day 30).

## Tech stack (planned, full pipeline)

- **pypdf** — PDF text extraction
- **sentence-transformers** — local embedding model (no API cost)
- **FAISS** — fast vector similarity search
- **OpenAI API** (or a local LLM) — answer generation
- **Streamlit** — chat UI

---
Part of [Rudra's](https://github.com/Ruddy2310) AI/ML portfolio work.
