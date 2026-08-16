"""
chunker.py
Splits loaded documents into smaller overlapping chunks. This matters
because:
  1. Embedding models and LLMs have limited context windows - you can't
     just embed an entire 40-page PDF as one vector and expect it to
     capture everything usefully.
  2. Retrieval works better on focused chunks - if a question is about
     one specific topic, you want to retrieve the paragraph that
     actually discusses it, not an entire document.
  3. Overlap between chunks prevents an answer from being awkwardly
     split across a chunk boundary and losing context.
"""

from dataclasses import dataclass, field

from document_loader import LoadedDocument


@dataclass
class Chunk:
    """A single retrievable piece of text, with metadata pointing back to its source."""
    text: str
    source_filename: str
    chunk_index: int  # position of this chunk within its source document
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[tuple[str, int, int]]:
    """
    Split text into overlapping chunks by character count, trying to
    break on sentence/paragraph boundaries when possible instead of
    cutting words in half mid-sentence.

    Args:
        text: the full text to split.
        chunk_size: target number of characters per chunk.
        overlap: number of characters shared between consecutive chunks.

    Returns:
        List of (chunk_text, start_index, end_index) tuples.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Try to break on a paragraph or sentence boundary near the end,
        # instead of cutting mid-word/mid-sentence
        if end < text_length:
            for boundary in ["\n\n", ". ", "\n"]:
                boundary_pos = text.rfind(boundary, start, end)
                if boundary_pos != -1 and boundary_pos > start + chunk_size // 2:
                    end = boundary_pos + len(boundary)
                    break

        chunk_text_piece = text[start:end].strip()
        if chunk_text_piece:  # skip empty chunks (e.g. from excessive whitespace)
            chunks.append((chunk_text_piece, start, end))

        # Move forward, but overlap with the previous chunk
        next_start = end - overlap
        start = next_start if next_start > start else end  # guard against infinite loop on tiny texts

    return chunks


def chunk_document(document: LoadedDocument, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunk a single loaded document, attaching source metadata to each chunk."""
    raw_chunks = chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)

    return [
        Chunk(
            text=text,
            source_filename=document.filename,
            chunk_index=i,
            char_start=start,
            char_end=end,
            metadata={"source_path": document.source_path},
        )
        for i, (text, start, end) in enumerate(raw_chunks)
    ]


def chunk_documents(documents: list[LoadedDocument], chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Chunk a whole batch of documents into a single flat list of chunks."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks


if __name__ == "__main__":
    # Quick manual test
    import sys
    from document_loader import load_documents_from_folder

    test_folder = sys.argv[1] if len(sys.argv) > 1 else "./sample_docs"
    docs = load_documents_from_folder(test_folder)

    if not docs:
        print(f"No documents found in {test_folder}. Add a .pdf/.txt/.md file there and re-run.")
    else:
        chunks = chunk_documents(docs)
        print(f"Loaded {len(docs)} document(s) -> split into {len(chunks)} chunk(s)\n")

        for chunk in chunks[:3]:
            print(f"[{chunk.source_filename} | chunk {chunk.chunk_index} | "
                  f"chars {chunk.char_start}-{chunk.char_end}]")
            print(f"{chunk.text[:200]}...\n")
