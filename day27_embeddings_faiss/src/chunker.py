"""
chunker.py
-----------
Splits raw study material (notes, textbook text, PDFs already converted to
plain text, etc.) into overlapping chunks that are small enough to embed
and search over meaningfully.

This is intentionally dependency-free (pure Python) so it can plug into
whatever text-extraction step earlier days of StudyMate already handle.
"""

from dataclasses import dataclass, field
from typing import List
import re


@dataclass
class Chunk:
    """A single chunk of text plus light metadata."""
    id: int
    text: str
    source: str = "unknown"
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """
    Splits text into overlapping, sentence-aware chunks.

    chunk_size: target number of characters per chunk
    overlap:    number of characters shared between consecutive chunks,
                so context isn't lost at chunk boundaries
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        # naive sentence splitter -- good enough for study notes/textbooks
        self._sentence_re = re.compile(r"(?<=[.!?])\s+")

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        return self._sentence_re.split(text)

    def chunk_text(self, text: str, source: str = "unknown") -> List[Chunk]:
        sentences = self._split_sentences(text)
        chunks: List[Chunk] = []

        current = ""
        current_start = 0
        char_cursor = 0
        chunk_id = 0

        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.chunk_size:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(
                        Chunk(
                            id=chunk_id,
                            text=current,
                            source=source,
                            start_char=current_start,
                            end_char=current_start + len(current),
                        )
                    )
                    chunk_id += 1
                # start new chunk, carrying over the overlap tail
                overlap_text = current[-self.overlap:] if current else ""
                current_start = char_cursor - len(overlap_text)
                current = f"{overlap_text} {sentence}".strip()

            char_cursor += len(sentence) + 1

        if current:
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=current,
                    source=source,
                    start_char=current_start,
                    end_char=current_start + len(current),
                )
            )

        return chunks

    def chunk_file(self, path: str) -> List[Chunk]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.chunk_text(text, source=path)
