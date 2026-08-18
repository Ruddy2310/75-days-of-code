"""Day 28 - Lightweight local RAG engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class RAGEngine:
    """Load documents, build a semantic index, and retrieve relevant chunks."""

    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 650,
        overlap: int = 100,
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedder = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None

    def load_documents(self) -> list[Chunk]:
        """Read .txt and .md files and split them into overlapping chunks."""
        chunks: list[Chunk] = []

        for path in sorted(self.data_dir.glob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue

            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            # Normalize whitespace while preserving readable sentences.
            text = re.sub(r"\s+", " ", text)

            start = 0
            chunk_id = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()

                if chunk_text:
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            source=path.name,
                            chunk_id=chunk_id,
                        )
                    )
                    chunk_id += 1

                if end == len(text):
                    break

                start = end - self.overlap

        self.chunks = chunks
        return chunks

    def build_index(self) -> None:
        """Create a normalized FAISS inner-product index."""
        if not self.chunks:
            self.load_documents()

        if not self.chunks:
            raise RuntimeError(f"No .txt or .md documents found in {self.data_dir}")

        texts = [chunk.text for chunk in self.chunks]
        vectors = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the most semantically similar chunks."""
        if self.index is None:
            self.build_index()

        query_vector = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(score)))

        return results

    @staticmethod
    def fallback_answer(query: str, results: list[tuple[Chunk, float]]) -> str:
        """Answer without an external LLM by returning the best retrieved evidence."""
        if not results:
            return "I could not find relevant information in the knowledge base."

        lines = [
            "I don't have a local language model enabled, so here is the most "
            "relevant information I retrieved:"
        ]

        for i, (chunk, score) in enumerate(results, start=1):
            lines.append(f"\n[{i}] {chunk.text}")

        return "\n".join(lines)
