"""
document_loader.py
Loads documents from a folder and extracts raw text, regardless of
format. This is step 1 of the RAG pipeline: get clean text out of
whatever files the user has (PDFs, notes, markdown), before chunking
and embedding happen in later commits.
"""

import os
from pathlib import Path
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class LoadedDocument:
    """A single loaded document, before chunking."""
    source_path: str
    filename: str
    text: str
    num_pages: int = 1  # for PDFs; 1 for plain text/markdown files


def load_pdf(path: str) -> LoadedDocument:
    """Extract text from every page of a PDF and join it together."""
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n\n".join(pages_text)
    return LoadedDocument(
        source_path=path,
        filename=os.path.basename(path),
        text=full_text,
        num_pages=len(reader.pages),
    )


def load_text_file(path: str) -> LoadedDocument:
    """Load a plain .txt or .md file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return LoadedDocument(
        source_path=path,
        filename=os.path.basename(path),
        text=text,
        num_pages=1,
    )


def load_document(path: str) -> LoadedDocument:
    """Dispatch to the right loader based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in (".txt", ".md"):
        return load_text_file(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (only .pdf, .txt, .md are supported)")


def load_documents_from_folder(folder_path: str) -> list[LoadedDocument]:
    """
    Load every supported document in a folder (non-recursive).
    Skips unsupported file types with a warning instead of crashing,
    so one bad file doesn't stop the whole batch.
    """
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue
        try:
            doc = load_document(str(file_path))
            if doc.text.strip():  # skip files that produced no extractable text
                documents.append(doc)
            else:
                print(f"  Skipped (no extractable text): {file_path.name}")
        except ValueError:
            print(f"  Skipped (unsupported type): {file_path.name}")
        except Exception as e:
            print(f"  Skipped (error reading file): {file_path.name} - {e}")

    return documents


if __name__ == "__main__":
    # Quick manual test - point this at a folder of your own notes/PDFs
    import sys

    test_folder = sys.argv[1] if len(sys.argv) > 1 else "./sample_docs"
    print(f"Loading documents from: {test_folder}\n")

    docs = load_documents_from_folder(test_folder)

    print(f"\nLoaded {len(docs)} document(s):")
    for doc in docs:
        preview = doc.text[:150].replace("\n", " ")
        print(f"  - {doc.filename} ({doc.num_pages} page(s), {len(doc.text)} chars)")
        print(f"    Preview: {preview}...")
