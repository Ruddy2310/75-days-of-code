"""
build_index.py
---------------
Day 27: reads study notes, chunks them, embeds each chunk with
sentence-transformers, and builds a FAISS index for semantic search.

Run:
    python build_index.py
"""

from src.chunker import TextChunker
from src.vector_store import VectorStore

DATA_FILE = "data/sample_notes.txt"
INDEX_PATH = "index/studymate"


def main():
    print(f"Reading notes from {DATA_FILE} ...")
    chunker = TextChunker(chunk_size=400, overlap=80)
    chunks = chunker.chunk_file(DATA_FILE)
    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model and building FAISS index (first run downloads the model)...")
    store = VectorStore()
    store.add(chunks)
    print(f"Indexed {len(store)} vectors of dimension {store.dim}.")

    store.save(INDEX_PATH)
    print(f"Saved index to {INDEX_PATH}.faiss and {INDEX_PATH}.pkl")


if __name__ == "__main__":
    main()
