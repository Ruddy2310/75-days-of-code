"""
vector_store.py
----------------
Day 27 of #75DaysOfCode -- StudyMate goes from "split text" to
"actually search by meaning."

Wraps:
  - sentence-transformers  -> turns text chunks into dense embeddings
  - FAISS                  -> fast approximate/exact nearest-neighbour search

Usage:
    store = VectorStore()
    store.add(chunks)                # list[Chunk] from chunker.py
    store.save("index/studymate")
    ...
    store = VectorStore.load("index/studymate")
    results = store.search("what is backpropagation?", top_k=3)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import Chunk

DEFAULT_MODEL = "all-MiniLM-L6-v2"  # small, fast, strong baseline for semantic search


class VectorStore:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        # Inner-product index over L2-normalised vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks: List[Chunk] = []

    # ---------- building the index ----------

    def _embed(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # required for cosine via inner product
        )
        return vectors.astype("float32")

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        vectors = self._embed([c.text for c in chunks])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    # ---------- querying ----------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        query_vec = self._embed([query])
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """Saves the FAISS index (.faiss) and chunk metadata (.pkl) side by side."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, f"{out}.faiss")
        with open(f"{out}.pkl", "wb") as f:
            pickle.dump(
                {"model_name": self.model_name, "chunks": self.chunks}, f
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(f"{path}.pkl", "rb") as f:
            data = pickle.load(f)

        store = cls(model_name=data["model_name"])
        store.index = faiss.read_index(f"{path}.faiss")
        store.chunks = data["chunks"]
        return store

    def __len__(self) -> int:
        return self.index.ntotal

    def export_chunks_json(self, path: str) -> None:
        """Optional human-readable dump for debugging what got indexed."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=2)
