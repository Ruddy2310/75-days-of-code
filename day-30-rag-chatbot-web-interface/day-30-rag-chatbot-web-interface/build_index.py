from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/documents")
INDEX_DIR = Path("index")

def load_documents():
    docs = []
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append({"source": str(path).replace("\\", "/"), "text": text})
    return docs

def chunk_documents(documents, size=700, overlap=120):
    chunks = []
    for doc in documents:
        start = 0
        chunk_id = 0
        while start < len(doc["text"]):
            end = min(start + size, len(doc["text"]))
            text = doc["text"][start:end].strip()
            if text:
                chunks.append({"source": doc["source"], "chunk_id": chunk_id, "text": text})
            if end == len(doc["text"]):
                break
            start = max(end - overlap, start + 1)
            chunk_id += 1
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    if not chunks:
        raise SystemExit("No documents found in data/documents.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    INDEX_DIR.mkdir(exist_ok=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_DIR/"documents.faiss"))
    (INDEX_DIR/"metadata.json").write_text(
        json.dumps(chunks, indent=2), encoding="utf-8"
    )
    print(f"Indexed {len(docs)} documents and {len(chunks)} chunks.")
