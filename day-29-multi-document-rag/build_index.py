from src.document_loader import load_documents
from src.chunker import chunk_documents
from src.vector_store import VectorStore

def main():
    docs = load_documents("data/documents")
    if not docs:
        raise SystemExit("No TXT/Markdown documents found.")
    chunks = chunk_documents(docs)
    VectorStore("index").build(chunks)
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print("FAISS index saved in index/")

if __name__ == "__main__":
    main()
