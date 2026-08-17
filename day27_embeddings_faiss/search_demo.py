"""
search_demo.py
---------------
Day 27 demo: loads the FAISS index built by build_index.py and lets you
search the study notes by meaning, not just keywords.

Run:
    python search_demo.py                # runs a few example queries
    python search_demo.py "your question here"
"""

import sys
from src.vector_store import VectorStore

INDEX_PATH = "index/studymate"

EXAMPLE_QUERIES = [
    "how does a model learn from its mistakes?",
    "what stops a model from memorizing the training data?",
    "how do transformers pay attention to different words?",
    "library for fast vector similarity search",
]


def run_query(store: VectorStore, query: str, top_k: int = 3):
    print(f"\nQuery: {query}")
    print("-" * 60)
    results = store.search(query, top_k=top_k)
    for rank, (chunk, score) in enumerate(results, start=1):
        preview = chunk.text.strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "..."
        print(f"  #{rank}  score={score:.3f}  {preview}")


def main():
    print(f"Loading index from {INDEX_PATH} ...")
    store = VectorStore.load(INDEX_PATH)
    print(f"Loaded {len(store)} chunks.")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(store, query)
    else:
        for q in EXAMPLE_QUERIES:
            run_query(store, q)


if __name__ == "__main__":
    main()
