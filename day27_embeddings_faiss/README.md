# Day 27 — StudyMate: Embeddings + Vector Store (FAISS)

Part of [#75DaysOfCode](../). This is the day StudyMate goes from "split text
into chunks" to **actually searching by meaning** instead of by keyword.

## What this does

1. **Chunk** — `src/chunker.py` splits raw notes into overlapping,
   sentence-aware chunks (so a search hit still has enough surrounding
   context to be useful).
2. **Embed** — `src/vector_store.py` turns each chunk into a dense vector
   using `sentence-transformers` (`all-MiniLM-L6-v2` by default — small and
   fast, good baseline for semantic search).
3. **Index** — vectors go into a FAISS `IndexFlatIP` index (cosine
   similarity via normalized inner product), so a query embedding can be
   compared against every chunk almost instantly.
4. **Search** — a natural-language question gets embedded the same way and
   matched against the index, returning the most *semantically* relevant
   chunks — even if they don't share any exact keywords with the query.

## Project structure

```
day27_embeddings_faiss/
├── data/
│   └── sample_notes.txt      # sample study notes to index
├── src/
│   ├── chunker.py            # TextChunker + Chunk dataclass
│   └── vector_store.py       # VectorStore: embed + FAISS index + save/load
├── index/                    # built index lands here (git-ignored)
├── build_index.py            # chunk -> embed -> build & save FAISS index
├── search_demo.py            # load index, run semantic queries
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Build the index from the sample notes (or point `DATA_FILE` in
`build_index.py` at your own notes):

```bash
python build_index.py
```

Then search it:

```bash
python search_demo.py                                   # runs example queries
python search_demo.py "how does backpropagation work?"  # your own question
```

Example output:

```
Query: how does a model learn from its mistakes?
------------------------------------------------------------
  #1  score=0.612  Backpropagation is the algorithm used to train neural
      networks by computing the gradient of the loss function...
  #2  score=0.487  Gradient descent is an optimization algorithm used to
      minimize a loss function by iteratively moving...
```

Note the query never says "backpropagation," "gradient," or "loss" — the
match comes from meaning, not keyword overlap. That's the difference
between this and the Day-26-style plain text search.

## Why FAISS + sentence-transformers

- **sentence-transformers** produces fixed-size embeddings where
  semantically similar sentences land close together in vector space —
  exactly what's needed to match a question to the right notes.
- **FAISS** (Facebook AI Research) makes comparing a query vector against
  thousands (or millions) of chunk vectors fast. This project uses an exact
  `IndexFlatIP` since the dataset is small; swapping in `IndexIVFFlat` or
  `IndexHNSWFlat` is a drop-in change once StudyMate's notes corpus grows
  large enough that exact search becomes a bottleneck.

## Next up

Day 28+: wire this into StudyMate's Q&A flow — retrieve the top-k chunks
for a user's question, then feed them to an LLM as context (retrieval-
augmented generation).
